# TASK-003 — Glacier Flexible Retrieval after worker finish

**Feature:** FEAT-032 Glacier originals  
**Status:** Done  
**Priority:** P1  
**Date:** 3 Sep 2026

## Goal

When this worker writes `completed` or `failed`, move that job's `source_key` to Glacier Flexible Retrieval immediately. Keep Standard while the job is queued, claimed, or running.

## Acceptance criteria

- [x] `archive_original` / `head_original` in this repo's `s3_service`
- [x] `_finish_slot` archives on `completed` / `failed`
- [x] `_fail_stale_running_in` archives those failed jobs
- [x] Stale `claimed` re-queue does not archive
- [x] `InvalidObjectState` on download fails the job (no RestoreObject)
- [x] Skip if another live job shares `source_key`
- [x] Orphan sweep on the stale tick
- [x] Archive failures do not fail the job

## Notes

Website API owns queued-cancel archive. This process does not mint CloudFront URLs. Glacier Flexible Retrieval has a 90-day minimum bill.
