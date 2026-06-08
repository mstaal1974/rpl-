# Scripts

## smoke_test.py — live end-to-end smoke test

Exercises the résumé-driven adaptive interview against a **running** deployment
(real Vertex calls): health → auth → create assessment → seed résumé →
profile experience → adaptive/start → adaptive/turn branching loop, asserting
the response shapes the UI relies on (including that AI-usage analysis ran on
each answer).

Zero dependencies (Python 3 stdlib only).

```bash
# Against a deployed URL, bootstrapping a throwaway org:
python3 scripts/smoke_test.py --base-url https://YOUR.run.app --bootstrap-key "$BOOTSTRAP_KEY"

# Or reuse an existing admin login:
python3 scripts/smoke_test.py --base-url https://YOUR.run.app \
    --admin-email you@rto.edu.au --admin-password 'secret'

# Offline check of the assertion logic only (no network, no cost):
python3 scripts/smoke_test.py --selfcheck
```

`BASE_URL`, `BOOTSTRAP_KEY`, `ADMIN_EMAIL`, `ADMIN_PASSWORD` env vars are honoured
as flag defaults. Exit code is non-zero if any check fails (CI-friendly).

Notes:
- Issues a handful of real LLM calls (~30–90s, a few cents).
- Uses an existing registry unit so it works on multi-instance Cloud Run; if the
  registry is empty it creates a temporary unit (single-instance only) and
  deletes it afterwards.
- Leaves the created assessment in place for inspection.
