# Employee Attendance & Analytics API

A robust, production-grade FastAPI and MongoDB service for employee attendance tracking and analytics, built for the HROne Software Engineer Trainee technical assessment.

---

## 1. Prerequisites

- Python 3.11 or newer
- MongoDB 6.0 or newer (MongoDB 7.0 / 8.0+ supported, local or MongoDB Atlas)
- Git

---

## 2. Environment Configuration

The application reads MongoDB connection settings from environment variables with fallback defaults:

| Variable | Default Value | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection URI string |
| `MONGO_DB` | `attendance_db` | MongoDB database name |

Real environment variables take precedence over any local `.env` file. Do not commit `.env` or secrets.

---

## 3. Installation & Setup

1. **Clone the repository and navigate to root:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt pytest httpx
   ```

4. **Seed sample data (optional for local testing):**
   ```bash
   python sample_seed.py
   ```

---

## 4. Running the Application

Start the FastAPI application with Uvicorn from the repository root:

```bash
uvicorn app.main:app --port 8000 --reload
```

- **Readiness probe:** `http://localhost:8000/health` (returns `200 {"status": "ok"}` only when MongoDB answers a ping)
- **Interactive OpenAPI Documentation (Swagger UI):** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`

---

## 5. Running the Test Suite

Execute the automated test suite with pytest:

```bash
pytest -v
```

Tests cover:
- Employee creation, validation, and listing
- Punch-in and late grace period calculations (R1, R2)
- Punch-out, half-up work hours rounding, and overtime threshold (R3, R4, R5)
- Overnight shift handling across calendar midnight (R1)
- Regularization state transitions and immutable history audit trail
- Concurrency race conditions (simultaneous punch-in idempotency)
- Late leaderboard standard competition ranking with ties (1, 2, 2, 4)
- Department monthly summary and daily trend gap-filling (`$densify`)
- Admin explain execution plan validation

---

## 6. Endpoints Summary

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Readiness probe; verifies MongoDB ping |
| `POST` | `/employees` | Creates employee with unique `emp_code` constraint |
| `GET` | `/employees` | Paginated listing sorted by `emp_code` asc, optional department filter |
| `POST` | `/attendance/punch-in` | Creates day's record (R1, R2); atomic concurrency safety |
| `POST` | `/attendance/punch-out` | Closes latest punch-in; computes work hours, overtime, half day |
| `GET` | `/attendance` | Paginated attendance logs sorted by `date` desc, `emp_code` asc |
| `PATCH` | `/attendance/{emp_code}/{date}` | Regularizes record, recomputes derived fields, appends audit history |
| `GET` | `/analytics/employees/{emp_code}/monthly` | Monthly employee summary via MongoDB aggregation |
| `GET` | `/analytics/departments/summary` | Cross-collection monthly department aggregation (counts 0-log staff) |
| `GET` | `/analytics/leaderboard/late` | Competition ranking leaderboard using `$setWindowFields` |
| `GET` | `/analytics/departments/{department}/trend` | Daily series with `$densify` gap-filling and 7-day moving average |
| `GET` | `/admin/explain/{endpoint}` | MongoDB `explain` with `executionStats` for query plan inspection |

---

## 7. Submission Artifacts

- `app/main.py`: Complete application implementation in a single entry point.
- `requirements.txt`: Minimal runtime dependencies.
- `REVIEW.md`: Comprehensive defect log of the starter code with verified fixes.
- `DECISIONS.md`: Concise answers to the five core architectural questions.
- `README.md`: Setup, verification, and endpoint documentation.
