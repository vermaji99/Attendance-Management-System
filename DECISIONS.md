# DECISIONS.md

1. **Indexes.**
   Created `employees(emp_code)` (unique) for employee business key uniqueness and `employees(department, joined_on)` for department filtering and headcount scans. On `attendance_logs`, created `(emp_code, date)` (unique) to enforce one record per employee-day and eliminate punch-in races; `(date: -1, emp_code: 1)` for paginated attendance listing; `(emp_code: 1, punch_in: -1)` for punch-out session lookup; and `(date: 1, late_minutes: -1, emp_code: 1)` for leaderboard aggregation. Considered and rejected indexing `history.at` because audit logs are append-only and never queried by timestamp.

2. **Punch-in race.**
   When two identical punch-ins arrive simultaneously for the same employee and date, both compute the IST attendance date and attempt `insert_one`. The first request succeeds (HTTP 201). The second request is rejected atomically by MongoDB's unique compound index on `(emp_code, date)`, raising `DuplicateKeyError`. Our code catches this error and returns HTTP 409 Conflict, avoiding racy check-then-insert checks.

3. **Ties.**
   The leaderboard uses MongoDB's window function `$setWindowFields` with standard competition ranking `$rank: {}` (1, 2, 2, 4) on total late minutes descending, filtering `rank <= limit` *after* ranking. Tied employees at the cutoff share the same rank and are both returned, sorted deterministically by `emp_code` ascending.

4. **Headcount.**
   The department summary pipeline originates on the `employees` collection (filtering `joined_on <= month_end`) and uses `$lookup` to join `attendance_logs`. Grouping by department counts every matched employee, ensuring zero-log employees are fully counted in headcount.

5. **One thing you would change for 100x the data.**
   At 100x scale (~10M logs), cross-collection lookups in real-time aggregations become costly. I would incrementally pre-aggregate monthly employee totals on punch-out/regularization and shard `attendance_logs` by `(emp_code, date)`.
