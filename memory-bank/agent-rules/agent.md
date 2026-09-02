# Agent Rules — Cric-Lab Video Service Memory Bank

How the agent should behave in the **criclab-video-service** repo.

## Session start (mandatory)

1. Read **all** Memory Bank files before non-trivial work:
   - `productBrief.md`
   - `techContext.md`
   - `systemPatterns.md` (highest priority for architecture)
   - `roadmap.md`
   - Relevant `tasks/`
   - These `agent-rules/`
2. Summarize goals, constraints, and gaps **before** coding when the task is large.
3. Do **not** invent Notera notes/PWA patterns, Hybrid CRM, or Next.js-as-frontend defaults — this product is **Cric-Lab**. UI is `criclab-web-frontend`. Website HTTP is `criclab-web-backend`.

## Context over prompts

- Prefer Memory Bank over assumptions
- If a prompt conflicts with `systemPatterns.md`, follow `systemPatterns.md` and say so
- UX inspiration may reference SpinLab AI; domain remains cricket **bowling** for v1

## Planning before coding

For non-trivial work:

1. Restate requirements against `productBrief.md` + task
2. Outline approach against `systemPatterns.md` (pipeline stages)
3. List files to touch under `app/` (pipeline / balltrack / coaching matching / agent / pdf / worker)
4. Implement

## Implementation rules

- This repo: claim `queued` jobs (global FIFO + daily `quota_days` lease); MediaPipe/OpenCV; overlay; PDF; Gemma **video** notes; drill **matching**
- Catalog HTTP (Train / admin) stays in `criclab-web-backend` — do not add JWT, CORS, SMTP, or drill CRUD routes here
- LLM: Ollama `gemma3:4b` locally or Bedrock in production; never measure frames
- Persist deliveries in the **shared** MongoDB; the website API reads them
- Label physical metrics as estimates unless calibrated + validated
- One job per worker process; parallelism = more processes
- Idle-stop the worker EC2 when nothing is claimable now; the website API starts it again (including 00:00 UTC)
- Keep the three-repo split — do not add Vite/React sources here

## Demo prompts

| Prompt | Expected behavior |
|--------|-------------------|
| `Read all memory bank files.` | Explain worker ownership, two film modes, matching vs catalog HTTP — no code yet |
| Add a bowling metric | Extend metrics stage + JSON; UI cards live in criclab-web-frontend |
| Change drill matching | Edit `app/coaching/recommend.py` here; do not edit website `load_catalog` unless Train/admin should change |

## Updates

When asked to **update the memory bank**:

- Sync `roadmap.md` status and `tasks/` with reality
- Record architecture decisions in `systemPatterns.md`
- Keep files concise — this is the agent’s persistent project memory
