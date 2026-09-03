# TASK-004 — Honor in-flight cancel between stages

**Feature:** FEAT-033 Honor cancel  
**Status:** Done  
**Priority:** P1  
**Date:** 3 Sep 2026

## Goal

When the user hits Cancel on a job that is already claimed or running, keep Mongo at `cancelled`, finish the current major stage, then stop. Do not persist a delivery or write `failed`.

## Acceptance criteria

- [x] `update_job` (Action + Ball-flight) only writes while status is `claimed|processing|analyzing`
- [x] `JobCancelled` / `raise_if_cancelled` at ingest and each Action / Ball-flight stage
- [x] Worker does not write `failed` on `JobCancelled`
- [x] In-flight cancel keeps the daily quota slot; queued cancel never took one
- [x] `_finish_slot` archives the original after an honored cancel
- [x] Ball-flight deletes any deliveries already written for that `job_id`
- [x] Tab close still does not cancel — only the Cancel button

## Notes

Do not kill MediaPipe/OpenCV mid-loop. Website API still writes `cancelled`; this process is what stops work.
