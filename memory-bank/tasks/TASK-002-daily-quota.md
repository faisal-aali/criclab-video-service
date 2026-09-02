# TASK-002 — Daily quota claim cap + global FIFO

**Feature:** FEAT-031 Daily video quota  
**Status:** Done  
**Priority:** P1

## Goal

Never start more than `DAILY_VIDEO_QUOTA` videos on one UTC day. Claim the oldest eligible job across Action and Ball-flight. Release the slot on fail and stale re-queue.

## Acceptance criteria

- [x] `DAILY_VIDEO_QUOTA` in worker Settings (same default as the website API)
- [x] Atomic lease on `quota_days.started` before claim
- [x] Global FIFO by `created_at` (not Action-before-Ball-flight)
- [x] Claim filter: `status=queued` and `available_at <= now` (missing `available_at` is eligible)
- [x] Fail and stale `claimed` → `queued` decrement today's `started` and set `quota_state.dirty_at`
- [x] Complete keeps the slot; dirty flag still set so later expected times move
- [x] Idle-stop when nothing is claimable now (quota full or tomorrow-only queue). Do not stop with in-flight jobs. The website API pokes at 00:00 UTC.

## Notes

SMTP and schedule math stay on `criclab-web-backend`. This repo only counts starts. The API (always-on) starts this EC2; this process stops itself.
