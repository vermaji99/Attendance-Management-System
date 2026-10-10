"""
Employee Attendance & Analytics API

FastAPI application for HROne Software Engineer Trainee Technical Assignment.
Provides employee management, attendance tracking (punch-in, punch-out, regularization with audit trail),
MongoDB aggregation analytics pipelines, and MongoDB execution plan explain capabilities.
"""
from __future__ import annotations

import calendar
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

# Load local .env if present; actual environment variables take precedence
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

# PyMongo client configured with tz_aware=True to ensure UTC datetimes
client = MongoClient(MONGO_URI, tz_aware=True)
db = client[MONGO_DB]

# Indian Standard Time (UTC+05:30) per R1
IST = timezone(timedelta(hours=5, minutes=30))


# --------------------------------------------------------------------------- #
# Application Lifespan & Index Creation
# --------------------------------------------------------------------------- #
def ensure_index(collection, keys, **kwargs):
    """Create index only if an index on the exact same key pattern does not already exist."""
    try:
        existing = collection.index_information()
        for idx in existing.values():
            if idx.get("key") == keys:
                return
        collection.create_index(keys, **kwargs)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ensure all required indexes are created idempotently at application startup.
    Guarantees uniqueness constraints and optimal IXSCAN query plans.
    """
    # Unique index on employees.emp_code
    ensure_index(db.employees, [("emp_code", 1)], unique=True)
    # Supporting indexes for employees queries and department groupings
    ensure_index(db.employees, [("department", 1), ("joined_on", 1)])

    # Natural compound unique index on attendance_logs (emp_code, date)
    ensure_index(db.attendance_logs, [("emp_code", 1), ("date", 1)], unique=True)
    # Compound index for paginated list query sorted by date desc, emp_code asc
    ensure_index(db.attendance_logs, [("date", -1), ("emp_code", 1)])
    # Supporting index for punch-out finding most recent open punch-in
    ensure_index(db.attendance_logs, [("emp_code", 1), ("punch_in", -1)])
    # Supporting index for late leaderboard and status queries
    ensure_index(db.attendance_logs, [("date", 1), ("late_minutes", -1), ("emp_code", 1)])
    yield


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan
)


# --------------------------------------------------------------------------- #
# Date, Time, and Mathematical Rounding Helpers (R1 - R8)
# --------------------------------------------------------------------------- #
def truncate_to_seconds(ms: int) -> int:
    """R1: Instants are truncated to whole seconds before any calculation."""
    return ms - (ms % 1000)


def ms_to_utc_dt(ms: int) -> datetime:
    """Convert epoch milliseconds to timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Convert BSON datetime to epoch milliseconds integer for the API boundary."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def round_half_up_2(val: Optional[float]) -> Optional[float]:
    """R8: 2 decimals half-up rounding (quantize half-up, not banker's rounding)."""
    if val is None:
        return None
    return float(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def round_half_up_4(val: Optional[float]) -> Optional[float]:
    """R8: 4 decimals half-up rounding for rates."""
    if val is None:
        return None
    return float(Decimal(str(val)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def compute_attendance_date(punch_in_utc: datetime, shift_start: str, shift_end: str) -> str:
    """
    R1 Time zone and attendance date.
    A record's date is the IST calendar date of its punch_in instant.
    Overnight shifts (shift_end <= shift_start): a punch-in earlier than shift_end
    belongs to the shift that started the previous day.
    """
    punch_in_ist = punch_in_utc.astimezone(IST)
    sh_h, sh_m = map(int, shift_start.split(":"))
    eh_h, eh_m = map(int, shift_end.split(":"))

    is_overnight = (eh_h, eh_m) <= (sh_h, sh_m)
    if is_overnight:
        # If punch-in IST time of day is strictly earlier than shift_end time
        punch_time = (punch_in_ist.hour, punch_in_ist.minute, punch_in_ist.second)
        shift_end_time = (eh_h, eh_m, 0)
        if punch_time < shift_end_time:
            # Belongs to shift that started previous calendar day
            return (punch_in_ist.date() - timedelta(days=1)).isoformat()

    return punch_in_ist.date().isoformat()


def compute_late_minutes(punch_in_utc: datetime, shift_start: str, attendance_date_str: str) -> int:
    """
    R2 Late.
    Grace period 10 minutes. Late only if punch-in is strictly more than 10:00 (600 seconds)
    after shift_start; then late_minutes = whole minutes (floor) since shift_start.
    """
    punch_in_ist = punch_in_utc.astimezone(IST)
    h, m = map(int, shift_start.split(":"))
    att_date = date.fromisoformat(attendance_date_str)
    shift_start_ist = datetime(att_date.year, att_date.month, att_date.day, h, m, 0, tzinfo=IST)

    elapsed_seconds = (punch_in_ist - shift_start_ist).total_seconds()
    if elapsed_seconds > 600:
        return int(elapsed_seconds // 60)
    return 0


def compute_work_hours(punch_in_utc: datetime, punch_out_utc: datetime) -> float:
    """
    R4 Work hours.
    (punch_out - punch_in) in seconds / 3600, rounded to 2 decimals half-up.
    """
    duration_seconds = (punch_out_utc - punch_in_utc).total_seconds()
    hours = Decimal(str(duration_seconds)) / Decimal("3600")
    return float(hours.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_overtime(punch_out_utc: datetime, shift_end: str, attendance_date_str: str, shift_start: str) -> int:
    """
    R3 Overtime.
    Whole minutes (floor) from shift_end to punch_out, counted only if at least 30; otherwise 0.
    For overnight shifts (shift_end <= shift_start), shift_end falls on next calendar day.
    """
    punch_out_ist = punch_out_utc.astimezone(IST)
    sh_h, sh_m = map(int, shift_start.split(":"))
    eh_h, eh_m = map(int, shift_end.split(":"))
    att_date = date.fromisoformat(attendance_date_str)

    is_overnight = (eh_h, eh_m) <= (sh_h, sh_m)
    if is_overnight:
        end_date = att_date + timedelta(days=1)
    else:
        end_date = att_date

    shift_end_ist = datetime(end_date.year, end_date.month, end_date.day, eh_h, eh_m, 0, tzinfo=IST)
    elapsed_seconds = (punch_out_ist - shift_end_ist).total_seconds()
    if elapsed_seconds >= 1800:  # at least 30 minutes
        return int(elapsed_seconds // 60)
    return 0


# --------------------------------------------------------------------------- #
# Schemas & Serialization Models
# --------------------------------------------------------------------------- #
class Status(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    LEAVE = "LEAVE"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class PresenceStatus(str, Enum):
    PRESENT = "PRESENT"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class EmployeeCreate(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str = Field(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")

    @model_validator(mode="after")
    def validate_shift(self) -> "EmployeeCreate":
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(default=None, ge=100000000000, le=4102444800000)
    status: PresenceStatus = Field(default=PresenceStatus.PRESENT)


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(default=None, ge=100000000000, le=4102444800000)


class RegularizeRequest(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[int] = Field(default=None, ge=100000000000, le=4102444800000)
    punch_out: Optional[int] = Field(default=None, ge=100000000000, le=4102444800000)
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)


def serialize_employee(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize employee document to Employee response schema."""
    created_at = doc.get("created_at")
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc.get("shift_start", "09:30"),
        "shift_end": doc.get("shift_end", "18:30"),
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(created_at) if isinstance(created_at, datetime) else created_at,
    }


def serialize_history_entry(h: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize single audit history entry."""
    at_val = h.get("at")
    changes_out: Dict[str, Any] = {}
    for k, v in h.get("changes", {}).items():
        from_val = v.get("from")
        to_val = v.get("to")
        if k in ("punch_in", "punch_out"):
            if isinstance(from_val, datetime):
                from_val = dt_to_ms(from_val)
            if isinstance(to_val, datetime):
                to_val = dt_to_ms(to_val)
        changes_out[k] = {"from": from_val, "to": to_val}

    return {
        "at": dt_to_ms(at_val) if isinstance(at_val, datetime) else at_val,
        "by": h.get("by"),
        "reason": h.get("reason"),
        "changes": changes_out,
    }


def serialize_attendance(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize attendance document to AttendanceRecord response schema."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": [serialize_history_entry(h) for h in doc.get("history", [])],
    }


# --------------------------------------------------------------------------- #
# System & Employee Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse, tags=["system"], include_in_schema=False)
def index_ui():
    """Interactive HR Attendance & Analytics Web UI Dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HROne Attendance & Analytics Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .modal-backdrop { background-color: rgba(15, 23, 42, 0.6); backdrop-filter: blur(2px); }
  </style>
</head>
<body class="bg-slate-100 text-slate-800 antialiased min-h-screen">
  <!-- Top Navbar -->
  <header class="bg-indigo-700 text-white shadow-md sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 py-3 flex flex-wrap justify-between items-center gap-4">
      <div class="flex items-center space-x-3">
        <div class="bg-white text-indigo-700 w-10 h-10 rounded-lg flex items-center justify-center font-bold text-xl shadow">HR</div>
        <div>
          <h1 class="font-bold text-lg leading-tight">HROne Attendance & Analytics Portal</h1>
          <p class="text-xs text-indigo-200">FastAPI &bull; MongoDB 8.2 Engine</p>
        </div>
      </div>
      <div class="flex items-center space-x-2">
        <button onclick="openModal('modal_add_emp')" class="px-3 py-1.5 bg-indigo-800 hover:bg-indigo-900 rounded-md text-xs font-semibold tracking-wide transition flex items-center gap-1.5 border border-indigo-400">
          <i class="fa-solid fa-user-plus"></i> Add Employee
        </button>
        <a href="/docs" target="_blank" class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 rounded-md text-xs font-semibold tracking-wide transition flex items-center gap-1.5 border border-indigo-400">
          <i class="fa-solid fa-code"></i> Swagger (/docs)
        </a>
        <a href="/health" target="_blank" class="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 rounded-md text-xs font-semibold tracking-wide transition flex items-center gap-1.5">
          <i class="fa-solid fa-heart-pulse"></i> Health
        </a>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 py-6 space-y-6">
    <!-- Top Action Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Punch Terminal Widget -->
      <div class="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
        <h2 class="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
          <i class="fa-solid fa-fingerprint text-indigo-600"></i> Punch Action Terminal
        </h2>
        <div class="space-y-3 text-sm">
          <div>
            <label class="block text-xs font-medium text-slate-500 mb-1">Select Employee</label>
            <select id="punch_emp" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none">
              <!-- Dynamically populated from API -->
            </select>
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-500 mb-1">Status (for Punch In)</label>
            <select id="punch_status" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none">
              <option value="PRESENT">PRESENT (On Site)</option>
              <option value="WFH">WFH (Work From Home)</option>
              <option value="ON_DUTY">ON_DUTY (Official Duty)</option>
            </select>
          </div>
          <div>
            <div class="flex justify-between items-center mb-1">
              <label class="block text-xs font-medium text-slate-500">Timestamp</label>
              <div class="flex items-center gap-2 text-xs">
                <label class="cursor-pointer flex items-center gap-1"><input type="radio" name="time_mode" value="now" checked onchange="toggleTimeMode()"> Current Time</label>
                <label class="cursor-pointer flex items-center gap-1"><input type="radio" name="time_mode" value="custom" onchange="toggleTimeMode()"> Custom</label>
              </div>
            </div>
            <input type="datetime-local" id="custom_time_input" class="hidden w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none" value="2026-07-07T18:30">
          </div>
          <div class="grid grid-cols-2 gap-3 pt-1">
            <button onclick="handlePunch('in')" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 rounded-lg shadow-sm transition flex items-center justify-center gap-1.5 text-xs">
              <i class="fa-solid fa-right-to-bracket"></i> Punch In
            </button>
            <button onclick="handlePunch('out')" class="w-full bg-slate-800 hover:bg-slate-900 text-white font-medium py-2 rounded-lg shadow-sm transition flex items-center justify-center gap-1.5 text-xs">
              <i class="fa-solid fa-right-from-bracket"></i> Punch Out
            </button>
          </div>
          <div id="punch_msg" class="hidden p-3 rounded-lg text-xs font-mono break-all whitespace-pre-wrap"></div>
        </div>
      </div>

      <!-- Quick Metrics Summary -->
      <div class="bg-white p-5 rounded-xl shadow-sm border border-slate-200 lg:col-span-2 flex flex-col justify-between">
        <div>
          <div class="flex justify-between items-center mb-3">
            <h2 class="text-base font-bold text-slate-800 flex items-center gap-2">
              <i class="fa-solid fa-chart-pie text-indigo-600"></i> System Metrics Overview
            </h2>
            <span class="text-xs px-2.5 py-1 bg-emerald-100 text-emerald-800 rounded-full font-semibold">MongoDB 8.2 Connected</span>
          </div>
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
            <div class="bg-indigo-50 border border-indigo-100 p-3 rounded-lg">
              <p class="text-xs text-indigo-600 font-medium">Total Employees</p>
              <p id="stat_emp_count" class="text-xl font-extrabold text-indigo-900 mt-1">-</p>
            </div>
            <div class="bg-emerald-50 border border-emerald-100 p-3 rounded-lg">
              <p class="text-xs text-emerald-600 font-medium">Departments</p>
              <p id="stat_dept_count" class="text-xl font-extrabold text-emerald-900 mt-1">-</p>
            </div>
            <div class="bg-amber-50 border border-amber-100 p-3 rounded-lg">
              <p class="text-xs text-amber-600 font-medium">Overnight Shift</p>
              <p class="text-xl font-extrabold text-amber-900 mt-1">22:00-06:00</p>
            </div>
            <div class="bg-sky-50 border border-sky-100 p-3 rounded-lg">
              <p class="text-xs text-sky-600 font-medium">Timezone</p>
              <p class="text-xl font-extrabold text-sky-900 mt-1">IST (UTC+5:30)</p>
            </div>
          </div>
        </div>
        <div class="mt-4 pt-3 border-t border-slate-100 text-xs text-slate-500 flex flex-wrap justify-between items-center gap-2">
          <span><i class="fa-solid fa-shield-halved text-indigo-600"></i> Concurrency-Safe (Unique indexes on <code>(emp_code, date)</code>)</span>
          <span><i class="fa-solid fa-database text-emerald-600"></i> Pure MongoDB Aggregation Pipelines</span>
        </div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="flex border-b border-slate-200 space-x-2 sm:space-x-4 text-xs sm:text-sm font-semibold overflow-x-auto">
      <button onclick="switchTab('tab_logs')" id="btn_tab_logs" class="py-2.5 px-3 sm:px-4 border-b-2 border-indigo-600 text-indigo-600 transition flex items-center gap-2 whitespace-nowrap">
        <i class="fa-solid fa-calendar-check"></i> Attendance Logs
      </button>
      <button onclick="switchTab('tab_dept')" id="btn_tab_dept" class="py-2.5 px-3 sm:px-4 border-b-2 border-transparent text-slate-500 hover:text-slate-800 transition flex items-center gap-2 whitespace-nowrap">
        <i class="fa-solid fa-building"></i> Department Summary
      </button>
      <button onclick="switchTab('tab_leader')" id="btn_tab_leader" class="py-2.5 px-3 sm:px-4 border-b-2 border-transparent text-slate-500 hover:text-slate-800 transition flex items-center gap-2 whitespace-nowrap">
        <i class="fa-solid fa-trophy"></i> Late Leaderboard (Ties)
      </button>
      <button onclick="switchTab('tab_trend')" id="btn_tab_trend" class="py-2.5 px-3 sm:px-4 border-b-2 border-transparent text-slate-500 hover:text-slate-800 transition flex items-center gap-2 whitespace-nowrap">
        <i class="fa-solid fa-chart-line"></i> Trend ($densify)
      </button>
      <button onclick="switchTab('tab_emps')" id="btn_tab_emps" class="py-2.5 px-3 sm:px-4 border-b-2 border-transparent text-slate-500 hover:text-slate-800 transition flex items-center gap-2 whitespace-nowrap">
        <i class="fa-solid fa-users"></i> Employee Directory
      </button>
    </div>

    <!-- Tab 1: Attendance Logs Table -->
    <section id="tab_logs" class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div class="p-4 bg-slate-50 border-b border-slate-200 flex flex-wrap justify-between items-center gap-3">
        <div class="flex items-center gap-3">
          <h3 class="font-bold text-sm text-slate-700">Attendance Records</h3>
          <span class="text-xs text-slate-400">Natural key: (emp_code, date)</span>
        </div>
        <button onclick="loadLogs()" class="px-3 py-1 bg-white border border-slate-300 rounded text-xs font-semibold hover:bg-slate-100 transition shadow-sm">
          <i class="fa-solid fa-rotate-right"></i> Refresh
        </button>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse text-xs">
          <thead>
            <tr class="bg-slate-100 text-slate-600 border-b border-slate-200 font-semibold uppercase tracking-wider text-[11px]">
              <th class="p-3">Emp Code</th>
              <th class="p-3">Date (IST)</th>
              <th class="p-3">Status</th>
              <th class="p-3">Punch In</th>
              <th class="p-3">Punch Out</th>
              <th class="p-3">Work Hours</th>
              <th class="p-3">Late Mins</th>
              <th class="p-3">Overtime</th>
              <th class="p-3">Half Day</th>
              <th class="p-3 text-center">Action</th>
            </tr>
          </thead>
          <tbody id="logs_table_body" class="divide-y divide-slate-100">
            <tr><td colspan="10" class="p-4 text-center text-slate-400">Loading attendance records...</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- Tab 2: Department Summary -->
    <section id="tab_dept" class="hidden bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div class="p-4 bg-slate-50 border-b border-slate-200 flex flex-wrap justify-between items-center gap-3">
        <div class="flex items-center gap-3">
          <h3 class="font-bold text-sm text-slate-700">Department Summary</h3>
          <div class="flex items-center gap-2">
            <label class="text-xs text-slate-500 font-medium">Month:</label>
            <input type="month" id="dept_month_input" value="2026-07" onchange="loadDeptSummary()" class="border border-slate-300 rounded px-2 py-1 text-xs bg-white">
          </div>
        </div>
        <button onclick="loadDeptSummary()" class="px-3 py-1 bg-white border border-slate-300 rounded text-xs font-semibold hover:bg-slate-100 transition shadow-sm">
          <i class="fa-solid fa-rotate-right"></i> Refresh
        </button>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse text-xs">
          <thead>
            <tr class="bg-slate-100 text-slate-600 border-b border-slate-200 font-semibold uppercase tracking-wider text-[11px]">
              <th class="p-3">Department</th>
              <th class="p-3">Headcount (R9)</th>
              <th class="p-3">Present Days</th>
              <th class="p-3">Avg Work Hours</th>
              <th class="p-3">Late Count</th>
              <th class="p-3">Total Late Mins</th>
              <th class="p-3">Leave Count</th>
              <th class="p-3">On Duty Count</th>
            </tr>
          </thead>
          <tbody id="dept_table_body" class="divide-y divide-slate-100">
            <tr><td colspan="8" class="p-4 text-center text-slate-400">Loading department summary...</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- Tab 3: Late Leaderboard -->
    <section id="tab_leader" class="hidden bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div class="p-4 bg-slate-50 border-b border-slate-200 flex flex-wrap justify-between items-center gap-3">
        <div class="flex items-center gap-3">
          <h3 class="font-bold text-sm text-slate-700">Late-Comers Leaderboard</h3>
          <div class="flex items-center gap-2">
            <label class="text-xs text-slate-500 font-medium">Month:</label>
            <input type="month" id="leader_month_input" value="2026-07" onchange="loadLeaderboard()" class="border border-slate-300 rounded px-2 py-1 text-xs bg-white">
            <label class="text-xs text-slate-500 font-medium">Limit:</label>
            <select id="leader_limit_select" onchange="loadLeaderboard()" class="border border-slate-300 rounded px-2 py-1 text-xs bg-white">
              <option value="5">5</option>
              <option value="10" selected>10</option>
              <option value="25">25</option>
            </select>
          </div>
        </div>
        <button onclick="loadLeaderboard()" class="px-3 py-1 bg-white border border-slate-300 rounded text-xs font-semibold hover:bg-slate-100 transition shadow-sm">
          <i class="fa-solid fa-rotate-right"></i> Refresh
        </button>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse text-xs">
          <thead>
            <tr class="bg-slate-100 text-slate-600 border-b border-slate-200 font-semibold uppercase tracking-wider text-[11px]">
              <th class="p-3">Rank (1, 2, 2, 4)</th>
              <th class="p-3">Emp Code</th>
              <th class="p-3">Name</th>
              <th class="p-3">Department</th>
              <th class="p-3">Total Late Mins</th>
              <th class="p-3">Late Incidents</th>
            </tr>
          </thead>
          <tbody id="leader_table_body" class="divide-y divide-slate-100">
            <tr><td colspan="6" class="p-4 text-center text-slate-400">Loading leaderboard...</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- Tab 4: Department Trend ($densify) -->
    <section id="tab_trend" class="hidden bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div class="p-4 bg-slate-50 border-b border-slate-200 flex flex-wrap justify-between items-center gap-3">
        <div class="flex items-center flex-wrap gap-3">
          <h3 class="font-bold text-sm text-slate-700">Daily Trend Series</h3>
          <div class="flex items-center gap-2">
            <label class="text-xs text-slate-500 font-medium">Dept:</label>
            <select id="trend_dept_select" onchange="loadTrend()" class="border border-slate-300 rounded px-2 py-1 text-xs bg-white">
              <option value="Engineering">Engineering</option>
              <option value="Sales">Sales</option>
              <option value="Support">Support</option>
            </select>
            <label class="text-xs text-slate-500 font-medium">From:</label>
            <input type="date" id="trend_from_input" value="2026-07-05" onchange="loadTrend()" class="border border-slate-300 rounded px-2 py-1 text-xs bg-white">
            <label class="text-xs text-slate-500 font-medium">To:</label>
            <input type="date" id="trend_to_input" value="2026-07-10" onchange="loadTrend()" class="border border-slate-300 rounded px-2 py-1 text-xs bg-white">
          </div>
        </div>
        <button onclick="loadTrend()" class="px-3 py-1 bg-white border border-slate-300 rounded text-xs font-semibold hover:bg-slate-100 transition shadow-sm">
          <i class="fa-solid fa-rotate-right"></i> Refresh
        </button>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse text-xs">
          <thead>
            <tr class="bg-slate-100 text-slate-600 border-b border-slate-200 font-semibold uppercase tracking-wider text-[11px]">
              <th class="p-3">Date (Gap-Filled)</th>
              <th class="p-3">Working Day</th>
              <th class="p-3">Headcount</th>
              <th class="p-3">Present Count</th>
              <th class="p-3">Late Count</th>
              <th class="p-3">Attendance Rate</th>
              <th class="p-3">7-Day Moving Avg</th>
            </tr>
          </thead>
          <tbody id="trend_table_body" class="divide-y divide-slate-100">
            <tr><td colspan="7" class="p-4 text-center text-slate-400">Loading department trend...</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- Tab 5: Employee Directory -->
    <section id="tab_emps" class="hidden bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div class="p-4 bg-slate-50 border-b border-slate-200 flex justify-between items-center">
        <h3 class="font-bold text-sm text-slate-700">Employee Directory</h3>
        <button onclick="openModal('modal_add_emp')" class="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded text-xs font-semibold transition shadow-sm flex items-center gap-1.5">
          <i class="fa-solid fa-plus"></i> New Employee
        </button>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse text-xs">
          <thead>
            <tr class="bg-slate-100 text-slate-600 border-b border-slate-200 font-semibold uppercase tracking-wider text-[11px]">
              <th class="p-3">Emp Code</th>
              <th class="p-3">Name</th>
              <th class="p-3">Email</th>
              <th class="p-3">Department</th>
              <th class="p-3">Shift Hours</th>
              <th class="p-3">Joined Date</th>
            </tr>
          </thead>
          <tbody id="emps_table_body" class="divide-y divide-slate-100">
            <tr><td colspan="6" class="p-4 text-center text-slate-400">Loading employee directory...</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>

  <!-- Modal: Add Employee -->
  <div id="modal_add_emp" class="hidden fixed inset-0 z-50 flex items-center justify-center p-4 modal-backdrop">
    <div class="bg-white rounded-xl max-w-md w-full p-6 shadow-2xl border border-slate-200">
      <div class="flex justify-between items-center mb-4">
        <h3 class="font-bold text-base text-slate-800 flex items-center gap-2">
          <i class="fa-solid fa-user-plus text-indigo-600"></i> Create New Employee
        </h3>
        <button onclick="closeModal('modal_add_emp')" class="text-slate-400 hover:text-slate-600"><i class="fa-solid fa-xmark text-lg"></i></button>
      </div>
      <form onsubmit="handleCreateEmployee(event)" class="space-y-3 text-xs">
        <div>
          <label class="block text-slate-600 font-medium mb-1">Employee Code (pattern: EMP followed by 4-6 digits)</label>
          <input type="text" id="add_emp_code" placeholder="EMP0007" required class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>
        <div>
          <label class="block text-slate-600 font-medium mb-1">Full Name</label>
          <input type="text" id="add_name" placeholder="John Doe" required class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>
        <div>
          <label class="block text-slate-600 font-medium mb-1">Email</label>
          <input type="email" id="add_email" placeholder="john@example.com" required class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>
        <div>
          <label class="block text-slate-600 font-medium mb-1">Department</label>
          <input type="text" id="add_dept" placeholder="Engineering" required class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-slate-600 font-medium mb-1">Shift Start</label>
            <input type="text" id="add_shift_start" value="09:30" required class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
          </div>
          <div>
            <label class="block text-slate-600 font-medium mb-1">Shift End</label>
            <input type="text" id="add_shift_end" value="18:30" required class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
          </div>
        </div>
        <div>
          <label class="block text-slate-600 font-medium mb-1">Joined Date</label>
          <input type="date" id="add_joined_on" value="2026-07-01" required class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>
        <div id="add_emp_msg" class="hidden p-2 rounded text-xs"></div>
        <div class="flex justify-end gap-2 pt-2">
          <button type="button" onclick="closeModal('modal_add_emp')" class="px-4 py-2 border border-slate-300 rounded text-slate-600 hover:bg-slate-50 font-medium">Cancel</button>
          <button type="submit" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded font-medium">Create</button>
        </div>
      </form>
    </div>
  </div>

  <!-- Modal: Regularize Attendance -->
  <div id="modal_regularize" class="hidden fixed inset-0 z-50 flex items-center justify-center p-4 modal-backdrop">
    <div class="bg-white rounded-xl max-w-md w-full p-6 shadow-2xl border border-slate-200">
      <div class="flex justify-between items-center mb-4">
        <h3 class="font-bold text-base text-slate-800 flex items-center gap-2">
          <i class="fa-solid fa-pen-to-square text-indigo-600"></i> Regularize Attendance
        </h3>
        <button onclick="closeModal('modal_regularize')" class="text-slate-400 hover:text-slate-600"><i class="fa-solid fa-xmark text-lg"></i></button>
      </div>
      <form onsubmit="handleRegularizeSubmit(event)" class="space-y-3 text-xs">
        <div class="grid grid-cols-2 gap-3 bg-slate-50 p-2.5 rounded border border-slate-200">
          <div><span class="text-slate-400 block">Employee</span> <strong id="reg_emp_code" class="text-slate-700"></strong></div>
          <div><span class="text-slate-400 block">Date</span> <strong id="reg_date" class="text-slate-700"></strong></div>
        </div>
        <div>
          <label class="block text-slate-600 font-medium mb-1">Status</label>
          <select id="reg_status" class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
            <option value="PRESENT">PRESENT</option>
            <option value="WFH">WFH</option>
            <option value="ON_DUTY">ON_DUTY</option>
            <option value="ABSENT">ABSENT</option>
            <option value="LEAVE">LEAVE</option>
          </select>
        </div>
        <div>
          <label class="block text-slate-600 font-medium mb-1">Punch In Instant</label>
          <input type="datetime-local" id="reg_punch_in" class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>
        <div>
          <label class="block text-slate-600 font-medium mb-1">Punch Out Instant</label>
          <input type="datetime-local" id="reg_punch_out" class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>
        <div>
          <label class="block text-slate-600 font-medium mb-1">Reason (min 5 chars)</label>
          <input type="text" id="reg_reason" placeholder="Biometric glitch at entry" required minlength="5" class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>
        <div>
          <label class="block text-slate-600 font-medium mb-1">Regularized By</label>
          <input type="text" id="reg_by" value="hr.admin" required class="w-full border border-slate-300 rounded px-3 py-2 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>
        <div id="reg_msg" class="hidden p-2 rounded text-xs font-mono break-all whitespace-pre-wrap"></div>
        <div class="flex justify-end gap-2 pt-2">
          <button type="button" onclick="closeModal('modal_regularize')" class="px-4 py-2 border border-slate-300 rounded text-slate-600 hover:bg-slate-50 font-medium">Cancel</button>
          <button type="submit" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded font-medium">Save Regularization</button>
        </div>
      </form>
    </div>
  </div>

  <footer class="max-w-7xl mx-auto px-4 py-8 text-center text-xs text-slate-400">
    HROne Software Engineer Trainee Engineering Assessment &bull; FastAPI + MongoDB
  </footer>

  <script>
    function openModal(id) { document.getElementById(id).classList.remove('hidden'); }
    function closeModal(id) { document.getElementById(id).classList.add('hidden'); }

    function toggleTimeMode() {
      const mode = document.querySelector('input[name="time_mode"]:checked').value;
      const input = document.getElementById('custom_time_input');
      if (mode === 'custom') {
        input.classList.remove('hidden');
      } else {
        input.classList.add('hidden');
      }
    }

    function switchTab(tabId) {
      ['tab_logs', 'tab_dept', 'tab_leader', 'tab_trend', 'tab_emps'].forEach(id => {
        document.getElementById(id).classList.add('hidden');
        document.getElementById('btn_' + id).className = 'py-2.5 px-3 sm:px-4 border-b-2 border-transparent text-slate-500 hover:text-slate-800 transition flex items-center gap-2 whitespace-nowrap';
      });
      document.getElementById(tabId).classList.remove('hidden');
      document.getElementById('btn_' + tabId).className = 'py-2.5 px-3 sm:px-4 border-b-2 border-indigo-600 text-indigo-600 transition flex items-center gap-2 whitespace-nowrap';

      if (tabId === 'tab_logs') loadLogs();
      if (tabId === 'tab_dept') loadDeptSummary();
      if (tabId === 'tab_leader') loadLeaderboard();
      if (tabId === 'tab_trend') loadTrend();
      if (tabId === 'tab_emps') loadEmployees();
    }

    function formatTime(ms) {
      if (!ms) return '-';
      const d = new Date(ms);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function isoToEpochMs(val) {
      if (!val) return null;
      return new Date(val).getTime();
    }

    function msToIsoDatetimeLocal(ms) {
      if (!ms) return '';
      const d = new Date(ms);
      const pad = n => String(n).padStart(2, '0');
      return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + 'T' + pad(d.getHours()) + ':' + pad(d.getMinutes());
    }

    async function loadEmployees() {
      try {
        const res = await fetch('/employees?page=1&page_size=100');
        const data = await res.json();
        
        // Populate Directory Table
        const tbody = document.getElementById('emps_table_body');
        tbody.innerHTML = data.items.map(e => `
          <tr class="hover:bg-slate-50 transition">
            <td class="p-3 font-semibold text-slate-800">${e.emp_code}</td>
            <td class="p-3 font-medium">${e.name}</td>
            <td class="p-3 text-slate-500">${e.email}</td>
            <td class="p-3"><span class="px-2 py-0.5 bg-slate-100 rounded text-slate-700 font-semibold">${e.department}</span></td>
            <td class="p-3 text-slate-600 font-mono">${e.shift_start} - ${e.shift_end}</td>
            <td class="p-3 text-slate-500">${e.joined_on}</td>
          </tr>
        `).join('');

        // Populate Punch Dropdown
        const punchSelect = document.getElementById('punch_emp');
        punchSelect.innerHTML = data.items.map(e => `
          <option value="${e.emp_code}">${e.emp_code} - ${e.name} (${e.department})</option>
        `).join('');

        // Stats
        document.getElementById('stat_emp_count').textContent = data.total;
        const depts = new Set(data.items.map(e => e.department));
        document.getElementById('stat_dept_count').textContent = depts.size;
      } catch (err) {
        console.error(err);
      }
    }

    async function handleCreateEmployee(e) {
      e.preventDefault();
      const msg = document.getElementById('add_emp_msg');
      const payload = {
        emp_code: document.getElementById('add_emp_code').value.trim(),
        name: document.getElementById('add_name').value.trim(),
        email: document.getElementById('add_email').value.trim(),
        department: document.getElementById('add_dept').value.trim(),
        shift_start: document.getElementById('add_shift_start').value.trim(),
        shift_end: document.getElementById('add_shift_end').value.trim(),
        joined_on: document.getElementById('add_joined_on').value.trim(),
      };

      try {
        const res = await fetch('/employees', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok) {
          msg.className = 'p-2 rounded text-xs bg-emerald-50 text-emerald-800 block';
          msg.textContent = 'Employee created successfully!';
          setTimeout(() => {
            closeModal('modal_add_emp');
            loadEmployees();
            msg.className = 'hidden';
          }, 1000);
        } else {
          msg.className = 'p-2 rounded text-xs bg-rose-50 text-rose-800 block';
          msg.textContent = 'Error: ' + JSON.stringify(data.detail);
        }
      } catch (err) {
        msg.className = 'p-2 rounded text-xs bg-rose-50 text-rose-800 block';
        msg.textContent = 'Request failed: ' + err.message;
      }
    }

    async function handlePunch(type) {
      const emp = document.getElementById('punch_emp').value;
      const status = document.getElementById('punch_status').value;
      const mode = document.querySelector('input[name="time_mode"]:checked').value;
      const msg = document.getElementById('punch_msg');

      msg.className = 'p-3 rounded-lg text-xs font-mono break-all whitespace-pre-wrap block bg-slate-100 text-slate-600';
      msg.textContent = 'Processing request...';

      let punched_at = null;
      if (mode === 'custom') {
        const customVal = document.getElementById('custom_time_input').value;
        if (customVal) {
          punched_at = new Date(customVal).getTime();
        }
      }

      try {
        let res;
        const payload = { emp_code: emp };
        if (punched_at) payload.punched_at = punched_at;

        if (type === 'in') {
          payload.status = status;
          res = await fetch('/attendance/punch-in', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
        } else {
          res = await fetch('/attendance/punch-out', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
        }
        const data = await res.json();
        if (res.ok) {
          msg.className = 'p-3 rounded-lg text-xs font-mono break-all whitespace-pre-wrap block bg-emerald-50 text-emerald-800 border border-emerald-200';
          msg.textContent = 'Success (' + res.status + '):\n' + JSON.stringify(data, null, 2);
          loadLogs();
        } else {
          msg.className = 'p-3 rounded-lg text-xs font-mono break-all whitespace-pre-wrap block bg-rose-50 text-rose-800 border border-rose-200';
          msg.textContent = 'Error (' + res.status + '):\n' + JSON.stringify(data, null, 2);
        }
      } catch (err) {
        msg.className = 'p-3 rounded-lg text-xs font-mono break-all whitespace-pre-wrap block bg-rose-50 text-rose-800 border border-rose-200';
        msg.textContent = 'Fetch failed: ' + err.message;
      }
    }

    async function loadLogs() {
      try {
        const res = await fetch('/attendance?page=1&page_size=30');
        const data = await res.json();
        const tbody = document.getElementById('logs_table_body');
        if (!data.items || data.items.length === 0) {
          tbody.innerHTML = '<tr><td colspan="10" class="p-4 text-center text-slate-400">No attendance logs found.</td></tr>';
          return;
        }
        tbody.innerHTML = data.items.map(r => `
          <tr class="hover:bg-slate-50 transition">
            <td class="p-3 font-semibold text-slate-700">${r.emp_code}</td>
            <td class="p-3 text-slate-600 font-mono">${r.date}</td>
            <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${r.status === 'PRESENT' ? 'bg-emerald-100 text-emerald-700' : r.status === 'WFH' ? 'bg-sky-100 text-sky-700' : r.status === 'ON_DUTY' ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-100 text-slate-600'}">${r.status}</span></td>
            <td class="p-3 text-slate-600">${formatTime(r.punch_in)}</td>
            <td class="p-3 text-slate-600">${formatTime(r.punch_out)}</td>
            <td class="p-3 font-semibold">${r.work_hours != null ? r.work_hours.toFixed(2) + ' h' : '-'}</td>
            <td class="p-3 ${r.late_minutes > 0 ? 'text-amber-600 font-bold' : 'text-slate-400'}">${r.late_minutes} m</td>
            <td class="p-3 ${r.overtime_minutes > 0 ? 'text-indigo-600 font-bold' : 'text-slate-400'}">${r.overtime_minutes} m</td>
            <td class="p-3">${r.half_day ? '<span class="text-rose-600 font-bold">Yes</span>' : '<span class="text-slate-400">No</span>'}</td>
            <td class="p-3 text-center">
              <button onclick='openRegularizeModal(${JSON.stringify(r)})' class="px-2 py-1 bg-slate-100 hover:bg-indigo-100 text-slate-700 hover:text-indigo-700 rounded text-[11px] font-medium transition">
                <i class="fa-solid fa-pen"></i> Regularize
              </button>
            </td>
          </tr>
        `).join('');
      } catch (err) {
        console.error(err);
      }
    }

    function openRegularizeModal(r) {
      document.getElementById('reg_emp_code').textContent = r.emp_code;
      document.getElementById('reg_date').textContent = r.date;
      document.getElementById('reg_status').value = r.status;
      document.getElementById('reg_punch_in').value = msToIsoDatetimeLocal(r.punch_in);
      document.getElementById('reg_punch_out').value = msToIsoDatetimeLocal(r.punch_out);
      document.getElementById('reg_reason').value = '';
      document.getElementById('reg_msg').className = 'hidden';
      openModal('modal_regularize');
    }

    async function handleRegularizeSubmit(e) {
      e.preventDefault();
      const emp_code = document.getElementById('reg_emp_code').textContent;
      const date = document.getElementById('reg_date').textContent;
      const status = document.getElementById('reg_status').value;
      const punch_in_val = document.getElementById('reg_punch_in').value;
      const punch_out_val = document.getElementById('reg_punch_out').value;
      const reason = document.getElementById('reg_reason').value.trim();
      const regularized_by = document.getElementById('reg_by').value.trim();
      const msg = document.getElementById('reg_msg');

      const payload = {
        reason: reason,
        regularized_by: regularized_by,
        status: status
      };
      if (punch_in_val) payload.punch_in = isoToEpochMs(punch_in_val);
      if (punch_out_val) payload.punch_out = isoToEpochMs(punch_out_val);

      try {
        const res = await fetch('/attendance/' + emp_code + '/' + date, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok) {
          msg.className = 'p-2 rounded text-xs bg-emerald-50 text-emerald-800 block';
          msg.textContent = 'Record regularized successfully!';
          setTimeout(() => {
            closeModal('modal_regularize');
            loadLogs();
          }, 1000);
        } else {
          msg.className = 'p-2 rounded text-xs bg-rose-50 text-rose-800 block';
          msg.textContent = 'Error: ' + JSON.stringify(data.detail);
        }
      } catch (err) {
        msg.className = 'p-2 rounded text-xs bg-rose-50 text-rose-800 block';
        msg.textContent = 'Request failed: ' + err.message;
      }
    }

    async function loadDeptSummary() {
      try {
        const m = document.getElementById('dept_month_input').value || '2026-07';
        const res = await fetch('/analytics/departments/summary?month=' + m);
        const data = await res.json();
        const tbody = document.getElementById('dept_table_body');
        if (!data.items || data.items.length === 0) {
          tbody.innerHTML = '<tr><td colspan="8" class="p-4 text-center text-slate-400">No departments found for ' + m + '.</td></tr>';
          return;
        }
        tbody.innerHTML = data.items.map(it => `
          <tr class="hover:bg-slate-50 transition">
            <td class="p-3 font-bold text-slate-800">${it.department}</td>
            <td class="p-3 font-semibold text-indigo-700">${it.headcount}</td>
            <td class="p-3">${it.present_days}</td>
            <td class="p-3 font-medium">${it.avg_work_hours != null ? it.avg_work_hours.toFixed(2) + ' h' : '-'}</td>
            <td class="p-3">${it.late_count}</td>
            <td class="p-3">${it.total_late_minutes} m</td>
            <td class="p-3">${it.leave_count}</td>
            <td class="p-3">${it.on_duty_count}</td>
          </tr>
        `).join('');
      } catch (err) {
        console.error(err);
      }
    }

    async function loadLeaderboard() {
      try {
        const m = document.getElementById('leader_month_input').value || '2026-07';
        const lim = document.getElementById('leader_limit_select').value || '10';
        const res = await fetch('/analytics/leaderboard/late?month=' + m + '&limit=' + lim);
        const data = await res.json();
        const tbody = document.getElementById('leader_table_body');
        if (!data.items || data.items.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" class="p-4 text-center text-slate-400">No late incidents in ' + m + '.</td></tr>';
          return;
        }
        tbody.innerHTML = data.items.map(it => `
          <tr class="hover:bg-slate-50 transition">
            <td class="p-3 font-extrabold text-indigo-700">#${it.rank}</td>
            <td class="p-3 font-semibold">${it.emp_code}</td>
            <td class="p-3">${it.name}</td>
            <td class="p-3 text-slate-500">${it.department}</td>
            <td class="p-3 font-bold text-amber-600">${it.total_late_minutes} mins</td>
            <td class="p-3">${it.late_count}</td>
          </tr>
        `).join('');
      } catch (err) {
        console.error(err);
      }
    }

    async function loadTrend() {
      try {
        const dept = document.getElementById('trend_dept_select').value || 'Engineering';
        const fromDate = document.getElementById('trend_from_input').value || '2026-07-05';
        const toDate = document.getElementById('trend_to_input').value || '2026-07-10';
        const res = await fetch('/analytics/departments/' + dept + '/trend?from=' + fromDate + '&to=' + toDate);
        const data = await res.json();
        const tbody = document.getElementById('trend_table_body');
        if (!data.items || data.items.length === 0) {
          tbody.innerHTML = '<tr><td colspan="7" class="p-4 text-center text-slate-400">No trend data available.</td></tr>';
          return;
        }
        tbody.innerHTML = data.items.map(it => `
          <tr class="hover:bg-slate-50 transition">
            <td class="p-3 font-semibold font-mono">${it.date}</td>
            <td class="p-3">${it.is_working_day ? '<span class="text-emerald-600 font-bold">Mon-Fri</span>' : '<span class="text-slate-400">Weekend</span>'}</td>
            <td class="p-3">${it.headcount}</td>
            <td class="p-3">${it.present_count}</td>
            <td class="p-3">${it.late_count}</td>
            <td class="p-3 font-semibold">${it.attendance_rate != null ? (it.attendance_rate * 100).toFixed(2) + '%' : '-'}</td>
            <td class="p-3 font-bold text-indigo-700">${it.moving_avg_7d != null ? (it.moving_avg_7d * 100).toFixed(2) + '%' : '-'}</td>
          </tr>
        `).join('');
      } catch (err) {
        console.error(err);
      }
    }

    // Initialize application on load
    loadEmployees();
    loadLogs();
  </script>
</body>
</html>"""


@app.get("/health", tags=["system"])
def health():
    """Readiness probe. Returns 200 only if MongoDB answers a ping, otherwise 503."""
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(e)}")


@app.post("/employees", status_code=201, tags=["employees"])
def create_employee(body: EmployeeCreate):
    """
    Create an employee. Enforced by unique index on emp_code.
    Concurrent duplicates yield 409. created_at is set by server clock.
    """
    doc = body.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists")

    return serialize_employee(doc)


@app.get("/employees", tags=["employees"])
def list_employees(
    department: Optional[str] = Query(None, description="Exact, case-sensitive match."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List employees sorted by emp_code ascending. Total reflects filtered query."""
    q: Dict[str, Any] = {}
    if department:
        q["department"] = department

    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    items_cursor = db.employees.find(q, {"_id": 0}).sort("emp_code", 1).skip(skip).limit(page_size)
    items = [serialize_employee(emp) for emp in items_cursor]

    return {"items": items, "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# Attendance Endpoints (Punch-in, Punch-out, List, Regularization)
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201, tags=["attendance"])
def punch_in(body: PunchInRequest):
    """
    Record a punch-in. Creates the day's record (R1, R2).
    Race-safe: enforced by unique compound index on (emp_code, date).
    """
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if body.punched_at is not None:
        raw_ms = body.punched_at
    else:
        raw_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    truncated_ms = truncate_to_seconds(raw_ms)
    ts_utc = ms_to_utc_dt(truncated_ms)

    # Determine attendance date according to R1
    attendance_date = compute_attendance_date(ts_utc, emp["shift_start"], emp["shift_end"])
    late_mins = compute_late_minutes(ts_utc, emp["shift_start"], attendance_date)

    doc = {
        "emp_code": body.emp_code,
        "date": attendance_date,
        "status": body.status.value,
        "punch_in": ts_utc,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_mins,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="already punched in for this date")

    return serialize_attendance(doc)


@app.post("/attendance/punch-out", status_code=200, tags=["attendance"])
def punch_out(body: PunchOutRequest):
    """
    Record a punch-out. Closes employee's most recent record whose punch_in is <= punched_at.
    Computes work_hours (R4), overtime_minutes (R3), and half_day (R5).
    Race-safe: atomic update ensures concurrent punch-outs yield exactly one 200 and rest 409.
    """
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if body.punched_at is not None:
        raw_ms = body.punched_at
    else:
        raw_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    truncated_ms = truncate_to_seconds(raw_ms)
    ts_utc = ms_to_utc_dt(truncated_ms)

    # Find most recent record with punch_in <= punched_at
    target_rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts_utc}},
        sort=[("punch_in", -1)]
    )

    if not target_rec:
        raise HTTPException(status_code=404, detail="No punch-in found")

    if target_rec.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="that record is already punched out")

    punch_in_utc = target_rec["punch_in"]
    if punch_in_utc.tzinfo is None:
        punch_in_utc = punch_in_utc.replace(tzinfo=timezone.utc)

    # Validation: punched_at must be strictly greater than punch_in and <= 24 hours
    if ts_utc <= punch_in_utc:
        raise HTTPException(status_code=422, detail="punched_at must be strictly after punch_in")

    diff_seconds = (ts_utc - punch_in_utc).total_seconds()
    if diff_seconds > 24 * 3600:
        raise HTTPException(status_code=422, detail="punched_at cannot be more than 24 hours after punch_in")

    # Compute derived fields
    work_hrs = compute_work_hours(punch_in_utc, ts_utc)
    ot_mins = compute_overtime(ts_utc, emp["shift_end"], target_rec["date"], emp["shift_start"])
    half_d = bool(work_hrs < 4.50)

    # Atomic update to prevent race conditions on simultaneous punch-outs
    updated_doc = db.attendance_logs.find_one_and_update(
        {"_id": target_rec["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": ts_utc,
                "work_hours": work_hrs,
                "overtime_minutes": ot_mins,
                "half_day": half_d,
            }
        },
        return_document=ReturnDocument.AFTER
    )

    if not updated_doc:
        raise HTTPException(status_code=409, detail="that record is already punched out")

    return serialize_attendance(updated_doc)


@app.get("/attendance", tags=["attendance"])
def list_attendance(
    emp_code: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status: Optional[Status] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    List attendance records sorted by date descending, then emp_code ascending.
    Fast on 100k records; does not load collection into memory.
    """
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from cannot be greater than date_to")

    q: Dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if status:
        q["status"] = status.value
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    cursor = db.attendance_logs.find(q).sort([("date", -1), ("emp_code", 1)]).skip(skip).limit(page_size)
    items = [serialize_attendance(doc) for doc in cursor]

    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}", tags=["attendance"])
def regularize_attendance(
    emp_code: str = Path(...),
    date: str = Path(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
    body: RegularizeRequest = ...,
):
    """
    Regularize (manually correct) one day's record addressed by (emp_code, date).
    Recomputes derived fields, validates state transitions, and appends to history.
    """
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not rec:
        raise HTTPException(status_code=404, detail="Attendance record not found for that date")

    # Determine target status
    new_status = body.status.value if body.status is not None else rec["status"]

    # Rule: setting ABSENT or LEAVE clears both punch times; supplying punch_in or punch_out with it is 422
    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(status_code=422, detail="punch_in and punch_out cannot be supplied when status is ABSENT or LEAVE")
        new_punch_in = None
        new_punch_out = None
    else:
        # Presence status requires a punch_in (supplied or stored)
        if body.punch_in is not None:
            trunc_in = truncate_to_seconds(body.punch_in)
            new_punch_in = ms_to_utc_dt(trunc_in)
        else:
            new_punch_in = rec.get("punch_in")
            if new_punch_in is not None and new_punch_in.tzinfo is None:
                new_punch_in = new_punch_in.replace(tzinfo=timezone.utc)

        if new_punch_in is None:
            raise HTTPException(status_code=422, detail="A presence status requires punch_in")

        # Validate that punch_in stays on the record's attendance date per R1
        computed_date = compute_attendance_date(new_punch_in, emp["shift_start"], emp["shift_end"])
        if computed_date != date:
            raise HTTPException(status_code=422, detail="punch_in must stay on the record's attendance date")

        # Handle punch_out
        if body.punch_out is not None:
            trunc_out = truncate_to_seconds(body.punch_out)
            new_punch_out = ms_to_utc_dt(trunc_out)
        else:
            new_punch_out = rec.get("punch_out")
            if new_punch_out is not None and new_punch_out.tzinfo is None:
                new_punch_out = new_punch_out.replace(tzinfo=timezone.utc)

        # Validate punch_out relative to punch_in if present
        if new_punch_out is not None:
            if new_punch_out <= new_punch_in:
                raise HTTPException(status_code=422, detail="punch_out must be strictly after punch_in")
            if (new_punch_out - new_punch_in).total_seconds() > 24 * 3600:
                raise HTTPException(status_code=422, detail="punch_out cannot be more than 24 hours after punch_in")

    # Recompute derived fields
    if new_status in ("ABSENT", "LEAVE"):
        new_work_hours = None
        new_late_minutes = 0
        new_overtime_minutes = 0
        new_half_day = False
    else:
        new_late_minutes = compute_late_minutes(new_punch_in, emp["shift_start"], date)
        if new_punch_out is not None:
            new_work_hours = compute_work_hours(new_punch_in, new_punch_out)
            new_overtime_minutes = compute_overtime(new_punch_out, emp["shift_end"], date, emp["shift_start"])
            new_half_day = bool(new_work_hours < 4.50)
        else:
            new_work_hours = None
            new_overtime_minutes = 0
            new_half_day = False

    # Detect field differences
    changes: Dict[str, Dict[str, Any]] = {}
    candidate_updates = {
        "status": (rec["status"], new_status),
        "punch_in": (rec.get("punch_in"), new_punch_in),
        "punch_out": (rec.get("punch_out"), new_punch_out),
        "work_hours": (rec.get("work_hours"), new_work_hours),
        "late_minutes": (rec.get("late_minutes", 0), new_late_minutes),
        "overtime_minutes": (rec.get("overtime_minutes", 0), new_overtime_minutes),
        "half_day": (rec.get("half_day", False), new_half_day),
    }

    for field_name, (old_val, new_val) in candidate_updates.items():
        if field_name in ("punch_in", "punch_out"):
            # Compare instants by millisecond timestamp
            old_ms = dt_to_ms(old_val)
            new_ms = dt_to_ms(new_val)
            if old_ms != new_ms:
                changes[field_name] = {"from": old_val, "to": new_val}
        else:
            if old_val != new_val:
                changes[field_name] = {"from": old_val, "to": new_val}

    if not changes:
        raise HTTPException(status_code=422, detail="No changes detected in regularization request")

    # Audit history entry (punches stored as BSON dates in DB)
    now_utc = datetime.now(timezone.utc)
    history_entry = {
        "at": now_utc,
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    # Atomic update appending history to prevent lost updates
    update_set = {
        "status": new_status,
        "punch_in": new_punch_in,
        "punch_out": new_punch_out,
        "work_hours": new_work_hours,
        "late_minutes": new_late_minutes,
        "overtime_minutes": new_overtime_minutes,
        "half_day": new_half_day,
    }

    current_history_len = len(rec.get("history", []))
    res = db.attendance_logs.find_one_and_update(
        {
            "_id": rec["_id"],
            f"history.{current_history_len}": {"$exists": False}  # Concurrency check
        },
        {
            "$set": update_set,
            "$push": {"history": history_entry}
        },
        return_document=ReturnDocument.AFTER
    )

    if not res:
        raise HTTPException(status_code=409, detail="Concurrent modification conflict")

    return serialize_attendance(res)


# --------------------------------------------------------------------------- #
# Analytics Pipelines (Employee Monthly, Department Summary, Leaderboard, Trend)
# --------------------------------------------------------------------------- #
def build_employee_monthly_pipeline(emp_code: str, month: str) -> List[Dict[str, Any]]:
    """Build pure aggregation pipeline for monthly attendance summary of one employee."""
    year, m = map(int, month.split("-"))
    days_in_month = calendar.monthrange(year, m)[1]
    month_start = f"{month}-01"
    month_end = f"{month}-{days_in_month:02d}"

    return [
        {"$match": {"emp_code": emp_code}},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"emp": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$emp"]},
                                    {"$gte": ["$date", month_start]},
                                    {"$lte": ["$date", month_end]}
                                ]
                            }
                        }
                    },
                    {
                        "$addFields": {
                            "dow": {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                            "is_mon_fri": {
                                "$in": [
                                    {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                                    [2, 3, 4, 5, 6]
                                ]
                            },
                            "is_presence": {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]}
                        }
                    }
                ],
                "as": "logs"
            }
        },
        {
            "$addFields": {
                "working_days": {
                    "$size": {
                        "$filter": {
                            "input": {
                                "$map": {
                                    "input": {"$range": [1, days_in_month + 1]},
                                    "as": "day",
                                    "in": {
                                        "$let": {
                                            "vars": {
                                                "date_str": {
                                                    "$concat": [
                                                        month,
                                                        "-",
                                                        {"$cond": [{"$lt": ["$$day", 10]}, {"$concat": ["0", {"$toString": "$$day"}]}, {"$toString": "$$day"}]}
                                                    ]
                                                }
                                            },
                                            "in": {
                                                "date_str": "$$date_str",
                                                "dow": {"$dayOfWeek": {"$dateFromString": {"dateString": "$$date_str"}}}
                                            }
                                        }
                                    }
                                }
                            },
                            "as": "d",
                            "cond": {
                                "$and": [
                                    {"$gte": ["$$d.date_str", "$joined_on"]},
                                    {"$in": ["$$d.dow", [2, 3, 4, 5, 6]]}
                                ]
                            }
                        }
                    }
                },
                "present_days": {
                    "$sum": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$logs",
                                    "as": "l",
                                    "cond": {
                                        "$and": [
                                            "$$l.is_mon_fri",
                                            "$$l.is_presence"
                                        ]
                                    }
                                }
                            },
                            "as": "pl",
                            "in": {"$cond": [{"$ifNull": ["$$pl.half_day", False]}, 0.5, 1.0]}
                        }
                    }
                },
                "leave_days": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "LEAVE"]}
                        }
                    }
                },
                "late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$gt": [{"$ifNull": ["$$l.late_minutes", 0]}, 0]}
                        }
                    }
                },
                "total_late_minutes": {"$sum": "$logs.late_minutes"},
                "total_overtime_minutes": {"$sum": "$logs.overtime_minutes"}
            }
        },
        {
            "$project": {
                "_id": 0,
                "emp_code": 1,
                "month": {"$literal": month},
                "working_days": 1,
                "present_days": 1,
                "leave_days": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "total_overtime_minutes": 1,
                "attendance_pct": {
                    "$cond": [
                        {"$eq": ["$working_days", 0]},
                        None,
                        {
                            "$multiply": [
                                {"$divide": ["$present_days", "$working_days"]},
                                100
                            ]
                        }
                    ]
                }
            }
        }
    ]


@app.get("/analytics/employees/{emp_code}/monthly", tags=["analytics"])
def employee_monthly(
    emp_code: str = Path(...),
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    """Monthly attendance summary for one employee backed by MongoDB aggregation."""
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    pipeline = build_employee_monthly_pipeline(emp_code, month)
    res = list(db.employees.aggregate(pipeline))
    if not res:
        raise HTTPException(status_code=404, detail="Employee not found")

    data = res[0]
    data["present_days"] = float(data["present_days"])
    data["attendance_pct"] = round_half_up_2(data["attendance_pct"])
    return data


def build_department_summary_pipeline(month: str, department: Optional[str] = None) -> List[Dict[str, Any]]:
    """Build cross-collection aggregation for monthly department summary."""
    year, m = map(int, month.split("-"))
    days_in_month = calendar.monthrange(year, m)[1]
    month_start = f"{month}-01"
    month_end = f"{month}-{days_in_month:02d}"

    match_stage: Dict[str, Any] = {"joined_on": {"$lte": month_end}}
    if department:
        match_stage["department"] = department

    return [
        {"$match": match_stage},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"emp": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$emp"]},
                                    {"$gte": ["$date", month_start]},
                                    {"$lte": ["$date", month_end]}
                                ]
                            }
                        }
                    },
                    {
                        "$addFields": {
                            "dow": {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                            "is_mon_fri": {
                                "$in": [
                                    {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                                    [2, 3, 4, 5, 6]
                                ]
                            },
                            "is_presence": {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                            "has_work_hours": {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$ne": ["$work_hours", None]}
                                ]
                            }
                        }
                    }
                ],
                "as": "logs"
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "all_logs": {"$push": "$logs"}
            }
        },
        {
            "$project": {
                "department": "$_id",
                "_id": 0,
                "headcount": 1,
                "logs": {
                    "$reduce": {
                        "input": "$all_logs",
                        "initialValue": [],
                        "in": {"$concatArrays": ["$$value", "$$this"]}
                    }
                }
            }
        },
        {
            "$project": {
                "department": 1,
                "headcount": 1,
                "present_days": {
                    "$sum": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$logs",
                                    "as": "l",
                                    "cond": {
                                        "$and": [
                                            "$$l.is_mon_fri",
                                            "$$l.is_presence"
                                        ]
                                    }
                                }
                            },
                            "as": "pl",
                            "in": {"$cond": [{"$ifNull": ["$$pl.half_day", False]}, 0.5, 1.0]}
                        }
                    }
                },
                "avg_work_hours": {
                    "$avg": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$logs",
                                    "as": "l",
                                    "cond": "$$l.has_work_hours"
                                }
                            },
                            "as": "wl",
                            "in": "$$wl.work_hours"
                        }
                    }
                },
                "late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$gt": [{"$ifNull": ["$$l.late_minutes", 0]}, 0]}
                        }
                    }
                },
                "total_late_minutes": {"$sum": "$logs.late_minutes"},
                "leave_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "LEAVE"]}
                        }
                    }
                },
                "on_duty_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "ON_DUTY"]}
                        }
                    }
                }
            }
        },
        {"$sort": {"department": 1}}
    ]


@app.get("/analytics/departments/summary", tags=["analytics"])
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
):
    """Department summary for a month cross-collection aggregation."""
    pipeline = build_department_summary_pipeline(month, department)
    items = list(db.employees.aggregate(pipeline))
    for it in items:
        it["present_days"] = float(it["present_days"])
        it["avg_work_hours"] = round_half_up_2(it["avg_work_hours"])

    return {"month": month, "items": items}


def build_late_leaderboard_pipeline(
    month: str,
    limit: int = 10,
    department: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Build late-comers leaderboard aggregation pipeline using window function."""
    year, m = map(int, month.split("-"))
    days_in_month = calendar.monthrange(year, m)[1]
    month_start = f"{month}-01"
    month_end = f"{month}-{days_in_month:02d}"

    pipeline: List[Dict[str, Any]] = [
        {
            "$match": {
                "date": {"$gte": month_start, "$lte": month_end},
                "late_minutes": {"$gt": 0}
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1}
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp"
            }
        },
        {"$unwind": "$emp"},
        {
            "$project": {
                "_id": 0,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1
            }
        }
    ]

    if department:
        pipeline.append({"$match": {"department": department}})

    # Standard competition ranking: ties share rank, next rank skipped
    pipeline.extend([
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {
                    "rank": {"$rank": {}}
                }
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "emp_code": 1}}
    ])

    return pipeline


@app.get("/analytics/leaderboard/late", tags=["analytics"])
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = Query(None),
):
    """Late-comers ranking with ties using MongoDB $setWindowFields."""
    pipeline = build_late_leaderboard_pipeline(month, limit, department)
    items = list(db.attendance_logs.aggregate(pipeline))
    return {"month": month, "items": items}


def build_department_trend_pipeline(department: str, date_from: str, date_to: str) -> List[Dict[str, Any]]:
    """Build daily series trend aggregation pipeline with $densify gap-filling and 7d moving average."""
    dt_from = datetime.fromisoformat(date_from)
    dt_to = datetime.fromisoformat(date_to)
    dt_to_bound = dt_to + timedelta(days=1)

    return [
        {"$match": {"department": department}},
        {"$limit": 1},
        {
            "$project": {
                "_id": 0,
                "date_dt": {"$literal": dt_from}
            }
        },
        {
            "$densify": {
                "field": "date_dt",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [dt_from, dt_to_bound]
                }
            }
        },
        {
            "$project": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$date_dt"}}
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"cur_date": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {"$eq": ["$date", "$$cur_date"]}
                        }
                    },
                    {
                        "$lookup": {
                            "from": "employees",
                            "localField": "emp_code",
                            "foreignField": "emp_code",
                            "as": "emp"
                        }
                    },
                    {"$unwind": "$emp"},
                    {
                        "$match": {
                            "emp.department": department
                        }
                    }
                ],
                "as": "logs"
            }
        },
        {
            "$addFields": {
                "present_count": {
                    "$sum": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$logs",
                                    "as": "l",
                                    "cond": {"$in": ["$$l.status", ["PRESENT", "WFH", "ON_DUTY"]]}
                                }
                            },
                            "as": "pl",
                            "in": {"$cond": [{"$ifNull": ["$$pl.half_day", False]}, 0.5, 1.0]}
                        }
                    }
                },
                "late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$gt": [{"$ifNull": ["$$l.late_minutes", 0]}, 0]}
                        }
                    }
                },
                "is_working_day": {
                    "$in": [
                        {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                        [2, 3, 4, 5, 6]
                    ]
                }
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"cur_date": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$cur_date"]}
                                ]
                            }
                        }
                    },
                    {"$count": "cnt"}
                ],
                "as": "hc_docs"
            }
        },
        {
            "$addFields": {
                "headcount": {
                    "$ifNull": [{"$arrayElemAt": ["$hc_docs.cnt", 0]}, 0]
                }
            }
        },
        {
            "$addFields": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_working_day",
                                {"$gt": ["$headcount", 0]}
                            ]
                        },
                        {"$divide": ["$present_count", "$headcount"]},
                        None
                    ]
                }
            }
        },
        {"$sort": {"date": 1}},
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "moving_avg_7d": {
                        "$avg": "$attendance_rate",
                        "window": {
                            "documents": [-6, "current"]
                        }
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": 1
            }
        }
    ]


@app.get("/analytics/departments/{department}/trend", tags=["analytics"])
def department_trend(
    department: str = Path(...),
    date_from: str = Query(..., alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: str = Query(..., alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """Daily attendance trend for a department with $densify gap-filling and moving average."""
    dept_exists = db.employees.count_documents({"department": department}) > 0
    if not dept_exists:
        raise HTTPException(status_code=404, detail="Department not found")

    d_from = date.fromisoformat(date_from)
    d_to = date.fromisoformat(date_to)
    if d_to < d_from:
        raise HTTPException(status_code=422, detail="'to' date cannot be before 'from' date")

    diff_days = (d_to - d_from).days + 1
    if diff_days > 92:
        raise HTTPException(status_code=422, detail="Date range cannot exceed 92 days")

    pipeline = build_department_trend_pipeline(department, date_from, date_to)
    items = list(db.employees.aggregate(pipeline))

    for it in items:
        it["present_count"] = float(it["present_count"])
        it["attendance_rate"] = round_half_up_4(it["attendance_rate"])
        it["moving_avg_7d"] = round_half_up_4(it["moving_avg_7d"])

    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Admin Explain Endpoint
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}", tags=["admin"])
def explain_endpoint(
    endpoint: str = Path(..., pattern=r"^(attendance_list|employee_monthly|department_summary|late_leaderboard|department_trend)$"),
    emp_code: Optional[str] = Query(None),
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
    limit: Optional[int] = Query(10, ge=1, le=50),
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status: Optional[Status] = Query(None),
    from_date: Optional[str] = Query(None, alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: Optional[str] = Query(None, alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(20, ge=1, le=100),
):
    """
    MongoDB explain output with executionStats for an endpoint's main query or aggregation pipeline.
    Validates required parameters for the specified endpoint (422 if missing).
    """
    if endpoint == "attendance_list":
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from cannot be greater than date_to")

        q: Dict[str, Any] = {}
        if emp_code:
            q["emp_code"] = emp_code
        if status:
            q["status"] = status.value
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to

        skip = ((page or 1) - 1) * (page_size or 20)
        cursor = db.attendance_logs.find(q).sort([("date", -1), ("emp_code", 1)]).skip(skip).limit(page_size or 20)
        raw_explain = cursor.explain()
        clean_explain = json_util.loads(json_util.dumps(raw_explain))
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": clean_explain,
        }

    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(status_code=422, detail="emp_code and month are required for employee_monthly")
        pipeline = build_employee_monthly_pipeline(emp_code, month)
        raw_explain = db.command(
            "explain",
            {"aggregate": "employees", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats"
        )
        clean_explain = json_util.loads(json_util.dumps(raw_explain))
        return {
            "endpoint": endpoint,
            "collection": "employees",
            "explain": clean_explain,
        }

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(status_code=422, detail="month is required for department_summary")
        pipeline = build_department_summary_pipeline(month, department)
        raw_explain = db.command(
            "explain",
            {"aggregate": "employees", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats"
        )
        clean_explain = json_util.loads(json_util.dumps(raw_explain))
        return {
            "endpoint": endpoint,
            "collection": "employees",
            "explain": clean_explain,
        }

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(status_code=422, detail="month is required for late_leaderboard")
        pipeline = build_late_leaderboard_pipeline(month, limit or 10, department)
        raw_explain = db.command(
            "explain",
            {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats"
        )
        clean_explain = json_util.loads(json_util.dumps(raw_explain))
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": clean_explain,
        }

    elif endpoint == "department_trend":
        if not department or not from_date or not to_date:
            raise HTTPException(status_code=422, detail="department, from, and to are required for department_trend")
        d_from = date.fromisoformat(from_date)
        d_to = date.fromisoformat(to_date)
        if d_to < d_from:
            raise HTTPException(status_code=422, detail="'to' date cannot be before 'from' date")
        if (d_to - d_from).days + 1 > 92:
            raise HTTPException(status_code=422, detail="Date range cannot exceed 92 days")

        pipeline = build_department_trend_pipeline(department, from_date, to_date)
        raw_explain = db.command(
            "explain",
            {"aggregate": "employees", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats"
        )
        clean_explain = json_util.loads(json_util.dumps(raw_explain))
        return {
            "endpoint": endpoint,
            "collection": "employees",
            "explain": clean_explain,
        }

    raise HTTPException(status_code=422, detail="Unknown endpoint for explain")
