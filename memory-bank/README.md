# Memory Bank — Cric-Lab (Video service)

Structured source of truth the AI reads first for the **Cric-Lab video worker** (OpenCV / MediaPipe / overlay / PDF / Gemma video notes / drill matching).

Sibling website API: `../criclab-web-backend`. Sibling UI: `../criclab-web-frontend`.

## Layout

```text
memory-bank/
├── productBrief.md      # What Cric-Lab is — users, features, success
├── techContext.md       # Worker stack, venv, env, layout
├── systemPatterns.md    # Architecture rules — MOST IMPORTANT FILE
├── roadmap.md           # Features this repo owns
├── tasks/               # Implementation / maintenance tasks
└── agent-rules/         # How the agent should behave
```

## Core files

| File | Role |
|------|------|
| `productBrief.md` | Cricket bowling lab: upload → analyze → metrics → PDF |
| `techContext.md` | Python worker, MediaPipe, Mongo (shared), Cloudinary, Bedrock/Ollama |
| `systemPatterns.md` | CV measures; Gemma coaches; two film modes; catalog matching vs HTTP |
| `roadmap.md` | Pipeline features that run in this process |

## Why this exists

Vague prompts force guessing. The Memory Bank encodes product intent and architecture so the agent behaves like a teammate who already knows Cric-Lab.

**Key lesson:** context changes output more than prompts.

## How to use

1. Ask: `Read all memory bank files.` — understanding before coding
2. Implement from a task under `tasks/` while obeying `systemPatterns.md`
3. To change architecture for the same feature, update `systemPatterns.md` first, then re-run the prompt
4. Website HTTP / auth / Train catalog CRUD live in `criclab-web-backend` — do not add them here

## Not Notera / SpinLab product code

This folder describes **Cric-Lab only**. UX inspiration comes from SpinLab AI; domain is cricket bowling, not baseball/throw sports generally (unless later expanded).
