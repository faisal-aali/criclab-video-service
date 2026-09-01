# TASK-001 — Dedicated video workers + coaching split

**Feature:** FEAT-026 Dedicated video workers  
**Status:** Done  
**Priority:** P0

## Goal

Run Action and Ball-flight analysis in this repo so the website API only inserts `queued` jobs. Drill matching after CV lives here; Train / admin catalog HTTP stays on `criclab-web-backend`.

## Acceptance criteria

- [x] `python -m app.worker` claims Mongo jobs (one clip per process)
- [x] `app/coaching/` has matching (`weakness_tags`, `balltrack_tags`, hydrate) + a `drills.json` snapshot
- [x] Website `recommend.py` is catalog I/O only (`load_catalog` / `save_catalog`)
- [x] Failed Action jobs re-raise after writing `status=failed` (worker logs `job failed`)
- [x] This Memory Bank matches the backend/frontend layout (`productBrief` → `agent-rules`)

## Notes

Admin edits to the website `drills.json` do not auto-sync here. A later Mongo catalog can unify them (out of scope for this task). GitHub Actions deploy for this repo is FEAT-027 (Planned).
