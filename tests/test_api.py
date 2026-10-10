import pytest
import concurrent.futures
import subprocess
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from app.main import app, db, IST

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Ensure indexes exist and lifespan runs."""
    with TestClient(app) as _:
        pass


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_create_and_list_employees():
    emp_code = f"EMP{int(datetime.now().timestamp() * 1000) % 90000 + 10000}"
    payload = {
        "emp_code": emp_code,
        "name": "Test User",
        "email": f"{emp_code.lower()}@example.com",
        "department": "Engineering",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    }
    # Create employee
    res = client.post("/employees", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["emp_code"] == emp_code
    assert isinstance(data["created_at"], int)

    # Duplicate should 409
    res_dup = client.post("/employees", json=payload)
    assert res_dup.status_code == 409

    # Validation: shift_start == shift_end should 422
    payload_invalid = dict(payload)
    payload_invalid["emp_code"] = f"EMP{int(datetime.now().timestamp() * 1000) % 90000 + 10001}"
    payload_invalid["shift_end"] = payload_invalid["shift_start"]
    res_inv = client.post("/employees", json=payload_invalid)
    assert res_inv.status_code == 422

    # List employees
    res_list = client.get("/employees?department=Engineering&page=1&page_size=10")
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert "items" in list_data
    assert list_data["total"] >= 1
    assert list_data["page"] == 1
    # Check sorted by emp_code asc
    emp_codes = [e["emp_code"] for e in list_data["items"]]
    assert emp_codes == sorted(emp_codes)


def test_punch_in_and_late_calculation():
    emp_code = f"EMP{int(datetime.now().timestamp() * 1000) % 90000 + 20000}"
    client.post("/employees", json={
        "emp_code": emp_code,
        "name": "Punch Tester",
        "email": f"{emp_code.lower()}@example.com",
        "department": "QA",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })

    # Exactly 09:40:00 IST -> 10 minutes elapsed -> Within grace period (R2) -> late_minutes = 0
    dt_grace = datetime(2026, 7, 20, 4, 10, 0, tzinfo=timezone.utc)
    ms_grace = int(dt_grace.timestamp() * 1000)

    res = client.post("/attendance/punch-in", json={
        "emp_code": emp_code,
        "punched_at": ms_grace,
        "status": "PRESENT"
    })
    assert res.status_code == 201
    data = res.json()
    assert data["date"] == "2026-07-20"
    assert data["late_minutes"] == 0
    assert data["half_day"] is False
    assert data["work_hours"] is None
    assert "id" not in data

    # Duplicate punch in for same day -> 409
    res_dup = client.post("/attendance/punch-in", json={
        "emp_code": emp_code,
        "punched_at": ms_grace + 60000,
        "status": "PRESENT"
    })
    assert res_dup.status_code == 409

    # Day 2: 09:40:01 IST -> strictly > 10 min grace period -> late_minutes = 10
    dt_late = datetime(2026, 7, 21, 4, 10, 1, tzinfo=timezone.utc)
    ms_late = int(dt_late.timestamp() * 1000)
    res_late = client.post("/attendance/punch-in", json={
        "emp_code": emp_code,
        "punched_at": ms_late,
        "status": "PRESENT"
    })
    assert res_late.status_code == 201
    assert res_late.json()["late_minutes"] == 10


def test_punch_out_and_work_hours_half_up():
    emp_code = f"EMP{int(datetime.now().timestamp() * 1000) % 90000 + 30000}"
    client.post("/employees", json={
        "emp_code": emp_code,
        "name": "Out Tester",
        "email": f"{emp_code.lower()}@example.com",
        "department": "QA",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })

    # Punch in at 09:30:00 IST (04:00:00 UTC)
    dt_in = datetime(2026, 7, 22, 4, 0, 0, tzinfo=timezone.utc)
    ms_in = int(dt_in.timestamp() * 1000)
    client.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": ms_in})

    # Duration: 3 hours and 30 seconds = 10830 seconds -> 10830 / 3600 = 3.008333... hours -> 3.01
    dt_out = datetime(2026, 7, 22, 7, 0, 30, tzinfo=timezone.utc)
    ms_out = int(dt_out.timestamp() * 1000)

    res_out = client.post("/attendance/punch-out", json={"emp_code": emp_code, "punched_at": ms_out})
    assert res_out.status_code == 200
    data = res_out.json()
    assert data["work_hours"] == 3.01
    assert data["half_day"] is True  # work_hours < 4.50
    assert data["overtime_minutes"] == 0

    # Duplicate punch out -> 409
    res_dup_out = client.post("/attendance/punch-out", json={"emp_code": emp_code, "punched_at": ms_out + 10000})
    assert res_dup_out.status_code == 409


def test_overtime_threshold_r3():
    emp_code = f"EMP{int(datetime.now().timestamp() * 1000) % 90000 + 40000}"
    client.post("/employees", json={
        "emp_code": emp_code,
        "name": "OT Tester",
        "email": f"{emp_code.lower()}@example.com",
        "department": "QA",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })

    # Punch in 09:30 IST
    dt_in = datetime(2026, 7, 23, 4, 0, 0, tzinfo=timezone.utc)
    client.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": int(dt_in.timestamp() * 1000)})

    # Punch out at 18:55 IST (25 min past 18:30) -> Under 30 min minimum -> overtime_minutes = 0
    dt_out_under = datetime(2026, 7, 23, 13, 25, 0, tzinfo=timezone.utc)
    res_under = client.post("/attendance/punch-out", json={"emp_code": emp_code, "punched_at": int(dt_out_under.timestamp() * 1000)})
    assert res_under.status_code == 200
    assert res_under.json()["overtime_minutes"] == 0

    # Next day: punch in 09:30 IST, punch out 19:15 IST (45 min past 18:30) -> overtime_minutes = 45
    dt_in2 = datetime(2026, 7, 24, 4, 0, 0, tzinfo=timezone.utc)
    client.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": int(dt_in2.timestamp() * 1000)})
    dt_out_over = datetime(2026, 7, 24, 13, 45, 0, tzinfo=timezone.utc)
    res_over = client.post("/attendance/punch-out", json={"emp_code": emp_code, "punched_at": int(dt_out_over.timestamp() * 1000)})
    assert res_over.status_code == 200
    assert res_over.json()["overtime_minutes"] == 45


def test_overnight_shift_r1():
    emp_code = f"EMP{int(datetime.now().timestamp() * 1000) % 90000 + 50000}"
    client.post("/employees", json={
        "emp_code": emp_code,
        "name": "Night Owl",
        "email": f"{emp_code.lower()}@example.com",
        "department": "Operations",
        "shift_start": "22:00",
        "shift_end": "06:00",
        "joined_on": "2026-01-01"
    })

    # Punch in on 2026-07-26 at 00:30 IST (which is < shift_end 06:00)
    # Belongs to shift that started previous day: 2026-07-25!
    dt_in = datetime(2026, 7, 25, 19, 0, 0, tzinfo=timezone.utc)
    res_in = client.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": int(dt_in.timestamp() * 1000)})
    assert res_in.status_code == 201
    assert res_in.json()["date"] == "2026-07-25"


def test_regularize_attendance_and_history():
    emp_code = f"EMP{int(datetime.now().timestamp() * 1000) % 90000 + 60000}"
    client.post("/employees", json={
        "emp_code": emp_code,
        "name": "Regularize Tester",
        "email": f"{emp_code.lower()}@example.com",
        "department": "Engineering",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })

    # Punch in at 10:15 IST (45 min late)
    dt_in = datetime(2026, 7, 27, 4, 45, 0, tzinfo=timezone.utc)
    ms_in = int(dt_in.timestamp() * 1000)
    client.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": ms_in})

    # Regularize punch_in to 09:28 IST (03:58 UTC) -> on time (late_minutes = 0)
    dt_corr = datetime(2026, 7, 27, 3, 58, 0, tzinfo=timezone.utc)
    ms_corr = int(dt_corr.timestamp() * 1000)

    patch_payload = {
        "punch_in": ms_corr,
        "reason": "Biometric device malfunction",
        "regularized_by": "manager.bob"
    }
    res_patch = client.patch(f"/attendance/{emp_code}/2026-07-27", json=patch_payload)
    assert res_patch.status_code == 200
    data = res_patch.json()
    assert data["late_minutes"] == 0
    assert len(data["history"]) == 1
    h = data["history"][0]
    assert h["by"] == "manager.bob"
    assert h["reason"] == "Biometric device malfunction"
    assert "punch_in" in h["changes"]
    assert "late_minutes" in h["changes"]
    assert h["changes"]["late_minutes"]["from"] == 45
    assert h["changes"]["late_minutes"]["to"] == 0

    # Sending with no changes should return 422
    res_no_change = client.patch(f"/attendance/{emp_code}/2026-07-27", json=patch_payload)
    assert res_no_change.status_code == 422

    # Regularize to ABSENT with punch_in supplied -> must 422
    res_bad_absent = client.patch(f"/attendance/{emp_code}/2026-07-27", json={
        "status": "ABSENT",
        "punch_in": ms_corr,
        "reason": "Invalid correction",
        "regularized_by": "hr.admin"
    })
    assert res_bad_absent.status_code == 422


def test_concurrency_race_punch_in():
    """Verify that under concurrent punch-ins, exactly one gets 201 and the other gets 409."""
    emp_code = f"EMP{int(datetime.now().timestamp() * 1000) % 90000 + 70000}"
    client.post("/employees", json={
        "emp_code": emp_code,
        "name": "Race Tester",
        "email": f"{emp_code.lower()}@example.com",
        "department": "QA",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })

    dt = datetime(2026, 7, 28, 4, 0, 0, tzinfo=timezone.utc)
    ms = int(dt.timestamp() * 1000)

    def do_punch():
        c = TestClient(app)
        return c.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": ms})

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(do_punch) for _ in range(5)]
        results = [f.result() for f in futures]

    statuses = [r.status_code for r in results]
    assert statuses.count(201) == 1
    assert statuses.count(409) == 4


def test_leaderboard_competition_ranking_ties():
    """Verify standard competition ranking with ties (1, 2, 2, 4) and limit applied after."""
    # Seed 4 employees with specific late minutes
    subprocess.run([".venv/Scripts/python", "sample_seed.py"], check=True)

    # Let's add two employees tied on late minutes
    emp_a = "EMP9001"
    emp_b = "EMP9002"
    client.post("/employees", json={
        "emp_code": emp_a,
        "name": "Tied A",
        "email": "tieda@example.com",
        "department": "TiedDept",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })
    client.post("/employees", json={
        "emp_code": emp_b,
        "name": "Tied B",
        "email": "tiedb@example.com",
        "department": "TiedDept",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })

    # Both punch in at 10:00 (30 min late) on 2026-07-15
    ms_late = int(datetime(2026, 7, 15, 4, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)
    client.post("/attendance/punch-in", json={"emp_code": emp_a, "punched_at": ms_late})
    client.post("/attendance/punch-in", json={"emp_code": emp_b, "punched_at": ms_late})

    res = client.get("/analytics/leaderboard/late?month=2026-07&department=TiedDept&limit=1")
    assert res.status_code == 200
    items = res.json()["items"]
    # Both share rank 1! Because limit=1 is applied AFTER ranking (rank <= limit), BOTH must be returned!
    assert len(items) == 2
    assert items[0]["rank"] == 1
    assert items[1]["rank"] == 1
    assert items[0]["total_late_minutes"] == 30
    assert items[1]["total_late_minutes"] == 30
    assert items[0]["emp_code"] < items[1]["emp_code"]


def test_list_attendance_date_validation():
    res = client.get("/attendance?date_from=2026-07-30&date_to=2026-07-01")
    assert res.status_code == 422


def test_analytics_endpoints():
    subprocess.run([".venv/Scripts/python", "sample_seed.py"], check=True)

    # 1. Employee Monthly
    res_m = client.get("/analytics/employees/EMP0001/monthly?month=2026-07")
    assert res_m.status_code == 200
    data_m = res_m.json()
    assert data_m["emp_code"] == "EMP0001"
    assert data_m["working_days"] == 23
    assert data_m["present_days"] == 2.0
    assert isinstance(data_m["attendance_pct"], float)

    # 2. Department Summary
    res_ds = client.get("/analytics/departments/summary?month=2026-07")
    assert res_ds.status_code == 200
    data_ds = res_ds.json()
    assert len(data_ds["items"]) >= 3
    eng = next(it for it in data_ds["items"] if it["department"] == "Engineering")
    assert eng["headcount"] == 2

    # 3. Late Leaderboard
    res_lb = client.get("/analytics/leaderboard/late?month=2026-07&limit=5")
    assert res_lb.status_code == 200
    data_lb = res_lb.json()
    assert len(data_lb["items"]) >= 1
    assert data_lb["items"][0]["rank"] == 1
    assert data_lb["items"][0]["total_late_minutes"] > 0

    # 4. Department Trend
    res_tr = client.get("/analytics/departments/Engineering/trend?from=2026-07-05&to=2026-07-10")
    assert res_tr.status_code == 200
    data_tr = res_tr.json()
    assert len(data_tr["items"]) == 6
    dates = [row["date"] for row in data_tr["items"]]
    assert dates == ["2026-07-05", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10"]


def test_admin_explain():
    endpoints = [
        ("attendance_list", {"date_from": "2026-07-01", "date_to": "2026-07-31"}),
        ("employee_monthly", {"emp_code": "EMP0001", "month": "2026-07"}),
        ("department_summary", {"month": "2026-07"}),
        ("late_leaderboard", {"month": "2026-07", "limit": 10}),
        ("department_trend", {"department": "Engineering", "from": "2026-07-01", "to": "2026-07-10"}),
    ]

    for ep_name, params in endpoints:
        res = client.get(f"/admin/explain/{ep_name}", params=params)
        assert res.status_code == 200, f"Explain failed for {ep_name}: {res.text}"
        data = res.json()
        assert data["endpoint"] == ep_name
        assert "explain" in data
