#!/usr/bin/env python3
"""
Live end-to-end smoke test for the RPL adaptive interview.
==========================================================

Exercises the full résumé-driven adaptive / branching flow against a RUNNING
deployment (real Vertex / Anthropic calls):

    health → auth → create assessment → seed résumé → profile experience
           → adaptive/start → adaptive/turn (branching loop)

and asserts the response shapes that the UI depends on — including that
AI-usage analysis actually ran on each answer.

Zero dependencies (Python 3 stdlib only). Safe to run from Cloud Shell, a
laptop, or CI.

USAGE
-----
Against a deployed Cloud Run URL, bootstrapping a fresh throwaway org:

    python3 scripts/smoke_test.py \
        --base-url https://rpl-portal-xxxx.run.app \
        --bootstrap-key "$BOOTSTRAP_KEY"

Or reusing an existing admin login (no bootstrap key needed):

    python3 scripts/smoke_test.py \
        --base-url https://rpl-portal-xxxx.run.app \
        --admin-email you@rto.edu.au --admin-password 'secret'

Environment variables BASE_URL, BOOTSTRAP_KEY, ADMIN_EMAIL, ADMIN_PASSWORD are
honoured as defaults for the matching flags.

Offline check of the script's own assertion logic (no network, no cost):

    python3 scripts/smoke_test.py --selfcheck

NOTES
-----
* This issues a handful of real LLM calls (profile + plan + ~2-4 turns).
  Expect it to take 30-90s and cost a few cents.
* It uses an existing unit already in the registry (so it works on
  multi-instance deployments). If the registry is empty it falls back to
  creating a temporary unit and warns that this only works single-instance.
* It does not delete the assessment it creates (left for inspection); pass
  --keep-unit to also keep any temporary unit.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

BRANCHES = {"DEEPEN", "CHALLENGE", "PIVOT", "ADVANCE", "GAP"}

# ── tiny output helpers ────────────────────────────────────────────────────────
_GREEN, _RED, _YEL, _DIM, _RST = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
_results = []  # (name, ok, detail)

def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    mark = f"{_GREEN}PASS{_RST}" if ok else f"{_RED}FAIL{_RST}"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"  {_DIM}{detail}{_RST}"
    print(line)
    return ok

def info(msg):  print(f"{_DIM}    · {msg}{_RST}")
def step(msg):  print(f"\n{_YEL}▸ {msg}{_RST}")


# ── HTTP (stdlib only) ──────────────────────────────────────────────────────────
def http(method, url, token=None, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode() or "null"
            return r.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            payload = None
        return e.code, payload
    except Exception as e:
        return 0, {"_transport_error": str(e)}


# ── response validators (shared by live run and --selfcheck) ───────────────────
def validate_profile(resp):
    out = []
    ep = (resp or {}).get("experience_profile")
    out.append(("profile: experience_profile present", isinstance(ep, dict), ""))
    cov = (resp or {}).get("pc_coverage")
    out.append(("profile: pc_coverage is a non-empty list",
                isinstance(cov, list) and len(cov) > 0,
                f"{len(cov) if isinstance(cov,list) else 0} PCs mapped"))
    if isinstance(cov, list) and cov:
        pr = cov[0].get("questioning_priority")
        out.append(("profile: PC carries a questioning_priority",
                    pr in ("CONFIRM", "PROBE", "EXPLORE", "GAP"), f"first PC = {pr}"))
    return out

def validate_start(resp):
    out = []
    plan = (resp or {}).get("plan")
    out.append(("start: plan is a non-empty list",
                isinstance(plan, list) and len(plan) > 0,
                f"{len(plan) if isinstance(plan,list) else 0} topics"))
    fs = (resp or {}).get("first_step") or (plan[0] if isinstance(plan, list) and plan else None)
    sc = (fs or {}).get("scenario") if isinstance(fs, dict) else None
    out.append(("start: first scenario has an opening_question",
                isinstance(sc, dict) and bool(sc.get("opening_question")),
                (sc.get("opening_question", "")[:60] + "…") if isinstance(sc, dict) and sc.get("opening_question") else ""))
    return out

def validate_turn(resp):
    out = []
    ta = (resp or {}).get("turn_analysis")
    out.append(("turn: turn_analysis present", isinstance(ta, dict),
                f"conf={ta.get('confidence') if isinstance(ta,dict) else '?'}"))
    branch = (resp or {}).get("branch") or {}
    dec = branch.get("decision")
    out.append(("turn: branch.decision is valid", dec in BRANCHES, f"decision={dec}"))
    au = (resp or {}).get("authenticity") or {}
    aiu = au.get("ai_usage") or {}
    out.append(("turn: AI-usage analysis ran on the answer",
                "probability" in aiu,
                f"AI={aiu.get('probability')} ({aiu.get('score')})"))
    na = (resp or {}).get("next_action")
    out.append(("turn: next_action is CONTINUE|CLOSE_PC",
                na in ("CONTINUE", "CLOSE_PC"), f"next_action={na}"))
    return out

def _emit(rows):
    ok_all = True
    for name, ok, detail in rows:
        ok_all = check(name, ok, detail) and ok_all
    return ok_all


# ── the live flow ───────────────────────────────────────────────────────────────
WEAK_ANSWER = ("It is important to follow best practice and ensure compliance at all "
               "times. One should always assess the situation and act in a timely manner.")
STRONG_ANSWER = ("At {emp} I do this most shifts. Last month a line reading drifted out of "
                 "spec, so I isolated the unit, re-calibrated against the reference standard, "
                 "logged the deviation in our QA sheet, and flagged it to my supervisor before "
                 "resuming. I double-checked with a second sample to confirm it was back in range.")

def run(args):
    base = args.base_url.rstrip("/")
    api = base + "/api"
    T = args.timeout

    # 1) HEALTH ----------------------------------------------------------------
    step("Health")
    s, h = http("GET", base + "/health", timeout=T)
    check("health: HTTP 200", s == 200, f"status={s}")
    if s != 200:
        return False
    check("health: status ok", (h or {}).get("status") == "ok", "")
    fs = (h or {}).get("firestore")
    if fs != "ok":
        info(f"WARNING firestore={fs!r} — multi-instance deployments need Firestore "
             f"for shared state; results may be flaky if not 'ok'.")
    info(f"units_loaded={h.get('units_loaded')} firestore={fs}")

    # 2) AUTH ------------------------------------------------------------------
    step("Authenticate")
    token = None
    if args.admin_email and args.admin_password:
        s, b = http("POST", api + "/auth/login",
                    body={"email": args.admin_email, "password": args.admin_password}, timeout=T)
        if check("auth: login", s == 200 and (b or {}).get("token"), f"status={s}"):
            token = b["token"]
            info(f"logged in as {args.admin_email}")
    if not token and args.bootstrap_key:
        ts = int(time.time())
        email = f"smoke+{ts}@example.com"
        pw = f"Smoke-{ts}-pw!"
        s, b = http("POST", api + "/auth/bootstrap", body={
            "bootstrap_key": args.bootstrap_key,
            "org_name": f"Smoke Test Org {ts}", "rto_code": "00000",
            "admin_email": email, "admin_password": pw, "admin_name": "Smoke Admin"}, timeout=T)
        if check("auth: bootstrap org+admin", s == 200, f"status={s} {(_err(b))}"):
            s2, b2 = http("POST", api + "/auth/login",
                          body={"email": email, "password": pw}, timeout=T)
            if check("auth: login (bootstrapped)", s2 == 200 and (b2 or {}).get("token"), f"status={s2}"):
                token = b2["token"]
                info(f"bootstrapped + logged in as {email}")
    if not token:
        check("auth: obtained a JWT", False,
              "provide --admin-email/--admin-password or --bootstrap-key")
        return False

    # 3) PICK A UNIT (existing → reliable across instances) --------------------
    step("Select a unit of competency")
    s, b = http("GET", api + "/units", token=token, timeout=T)
    units = (b or {}).get("units", []) if s == 200 else []
    unit_code, temp_unit = None, False
    if units:
        unit_code = units[0].get("code")
        info(f"using existing unit {unit_code} ({units[0].get('title','')[:50]})")
    else:
        # Fallback: create a temporary unit (single-instance only).
        unit_code = f"ZZSMOKE{int(time.time())}"
        temp_unit = True
        info("registry empty — creating a temporary unit (works on single-instance only)")
        s, b = http("POST", api + "/units/create", token=token, timeout=T, body={
            "code": unit_code, "title": "Smoke Test Unit",
            "training_package": "ZZ", "training_package_name": "Smoke",
            "application": "Smoke test unit.",
            "competent_person_statement": "Demonstrates the smoke test PCs.",
            "elements": [{"id": "E1", "title": "Perform the task", "analysis_focus": "", "pcs": [
                {"id": "1.1", "text": "carry out the procedure safely", "element_id": "E1",
                 "benchmark_statement": "The candidate demonstrates that they can carry out the procedure safely."},
                {"id": "1.2", "text": "record and report results", "element_id": "E1",
                 "benchmark_statement": "The candidate demonstrates that they can record and report results."}]}],
            "knowledge_requirements": [], "skill_requirements": [],
            "evidence_guide": [], "knowledge_questions": []})
        if not check("unit: created temporary unit", s == 200, f"status={s} {_err(b)}"):
            return _finish()
    if not check("unit: a unit code is available", bool(unit_code), unit_code or ""):
        return _finish()

    # 4) CREATE ASSESSMENT -----------------------------------------------------
    step("Create assessment")
    candidate = {"name": "Sam Smoke", "email": "sam.smoke@example.com",
                 "employer": "Acme Field Services", "role": "Field Technician",
                 "duration": "4 years", "industry": "Field operations"}
    s, b = http("POST", api + "/trainer/assessments/create", token=token, timeout=T, body={
        "trainer_name": "Smoke Trainer", "trainer_email": "trainer@example.com",
        "unit_codes": [unit_code], "candidate": candidate,
        "notes": "End-to-end smoke test."})
    if not check("assessment: created", s == 200 and (b or {}).get("assessment_id"), f"status={s} {_err(b)}"):
        return _finish(temp_unit, token, api, unit_code, T, args)
    assessment_id = b["assessment_id"]
    invite_token = b["invite_token"]
    info(f"assessment_id={assessment_id[:8]}… token={invite_token[:8]}…")

    # 5) STUDENT OPENS + SEEDS A RÉSUMÉ ---------------------------------------
    step("Seed candidate résumé (student side)")
    http("GET", api + f"/student/join/{invite_token}", timeout=T)  # mark opened
    resume = ("Field Technician at Acme Field Services for 4 years. Daily I calibrate "
              "instruments against reference standards, run sample checks, isolate faulty "
              "units, log deviations in the QA system and report non-conformances to my "
              "supervisor. Previously a trades assistant for 2 years.")
    s, b = http("POST", api + "/student/progress", timeout=T, body={
        "assessment_id": assessment_id, "token": invite_token,
        "progress": {"candidate_notes": {"resume": resume}}})
    check("résumé: saved to progress", s == 200 and (b or {}).get("saved"), f"status={s}")

    # 6) PROFILE EXPERIENCE (trainer; real Vertex) -----------------------------
    step("Profile experience from résumé (Vertex)")
    s, b = http("POST", api + "/assessment/profile-experience", token=token, timeout=T,
                body={"assessment_id": assessment_id})
    if not check("profile: HTTP 200", s == 200, f"status={s} {_err(b)}"):
        info("If 500: check the Cloud Run service account has roles/aiplatform.user.")
        return _finish(temp_unit, token, api, unit_code, T, args)
    _emit(validate_profile(b))

    # 7) ADAPTIVE START (student; real Vertex) ---------------------------------
    step("Start adaptive interview (Vertex)")
    s, b = http("POST", api + "/conversation/adaptive/start", timeout=T,
                body={"token": invite_token, "assessment_id": assessment_id, "unit_code": unit_code})
    if not check("start: HTTP 200", s == 200, f"status={s} {_err(b)}"):
        return _finish(temp_unit, token, api, unit_code, T, args)
    _emit(validate_start(b))
    plan = b.get("plan") or []
    first = b.get("first_step") or (plan[0] if plan else None)
    if not first:
        check("start: have a first topic to drive", False, "")
        return _finish(temp_unit, token, api, unit_code, T, args)
    pc_id = first.get("pc_id")
    scenario = first.get("scenario") or {}

    # 8) ADAPTIVE TURN LOOP (branching) ----------------------------------------
    step(f"Branching turns on PC {pc_id} (Vertex)")
    dialogue = []
    branch_seen, ai_seen, closed = [], False, False
    max_turns = 4
    for turn in range(1, max_turns + 1):
        answer = WEAK_ANSWER if turn == 1 else STRONG_ANSWER.format(emp=candidate["employer"])
        dialogue.append({"role": "candidate", "content": answer, "turn": turn})
        s, b = http("POST", api + "/conversation/adaptive/turn", timeout=T, body={
            "token": invite_token, "assessment_id": assessment_id, "unit_code": unit_code,
            "pc_id": pc_id, "scenario": scenario, "dialogue_history": dialogue,
            "latest_answer": answer, "turn_number": turn, "max_turns": max_turns})
        if not check(f"turn {turn}: HTTP 200", s == 200, f"status={s} {_err(b)}"):
            break
        ok = _emit(validate_turn(b))
        branch = (b or {}).get("branch") or {}
        dec = branch.get("decision")
        if dec:
            branch_seen.append(dec)
        if ((b or {}).get("authenticity") or {}).get("ai_usage", {}).get("probability"):
            ai_seen = True
        info(f"turn {turn}: branch={dec} "
             f"AI={((b.get('authenticity') or {}).get('ai_usage') or {}).get('probability')} "
             f"next={b.get('next_action')}")
        if (b or {}).get("next_action") == "CLOSE_PC":
            closed = True
            break
        # Continue: record the AI guide's follow-up as the next prompt.
        nq = branch.get("next_question")
        if nq:
            dialogue.append({"role": "assessor", "content": nq, "turn": turn})

    step("Branching assertions")
    check("branching: at least one branch decision returned", len(branch_seen) >= 1,
          " → ".join(branch_seen))
    check("branching: AI-usage analysis observed on a turn", ai_seen, "")
    check("branching: topic reached a terminal state (ADVANCE/GAP) within max turns",
          closed or any(d in ("ADVANCE", "GAP") for d in branch_seen),
          "closed" if closed else "max turns reached")

    return _finish(temp_unit, token, api, unit_code, T, args, assessment_id)


def _err(b):
    if isinstance(b, dict):
        return str(b.get("detail") or b.get("_transport_error") or "")[:160]
    return ""

def _finish(temp_unit=False, token=None, api=None, unit_code=None, T=120, args=None, assessment_id=None):
    if temp_unit and token and api and unit_code and not (args and args.keep_unit):
        s, _ = http("DELETE", api + f"/units/{unit_code}", token=token, timeout=T)
        info(f"cleaned up temporary unit {unit_code} (status {s})")
    if assessment_id:
        info(f"assessment {assessment_id} left in place for inspection")
    return all(ok for _, ok, _ in _results)


# ── offline self-check of the validators ───────────────────────────────────────
def selfcheck():
    print("Running offline self-check of assertion logic (no network)…\n")
    good_profile = {"experience_profile": {"summary": "x"},
                    "pc_coverage": [{"pc_id": "1.1", "questioning_priority": "PROBE"}]}
    good_start = {"plan": [{"pc_id": "1.1", "scenario": {"opening_question": "Walk me through…"}}],
                  "first_step": {"pc_id": "1.1", "scenario": {"opening_question": "Walk me through…"}}}
    good_turn = {"turn_analysis": {"confidence": 0.7, "judgement": "Satisfactory"},
                 "branch": {"decision": "ADVANCE"},
                 "authenticity": {"verdict": "AUTHENTIC", "ai_usage": {"probability": "LOW", "score": 10}},
                 "next_action": "CLOSE_PC"}
    bad_turn = {"branch": {"decision": "NONSENSE"}, "authenticity": {}, "next_action": "??"}

    print("Good payloads (all should PASS):")
    g = _emit(validate_profile(good_profile)) and _emit(validate_start(good_start)) and _emit(validate_turn(good_turn))
    print("\nMalformed turn (all should FAIL):")
    before = len(_results)
    _emit(validate_turn(bad_turn))
    bad_rows = _results[before:]
    bad_all_failed = all(not ok for _, ok, _ in bad_rows)

    print()
    if g and bad_all_failed:
        print(f"{_GREEN}SELF-CHECK OK{_RST} — validators pass on good data and fail on malformed data.")
        return True
    print(f"{_RED}SELF-CHECK FAILED{_RST} — validator logic is wrong.")
    return False


def main():
    ap = argparse.ArgumentParser(description="Live end-to-end smoke test for the RPL adaptive interview.")
    ap.add_argument("--base-url", default=os.getenv("BASE_URL"))
    ap.add_argument("--bootstrap-key", default=os.getenv("BOOTSTRAP_KEY"))
    ap.add_argument("--admin-email", default=os.getenv("ADMIN_EMAIL"))
    ap.add_argument("--admin-password", default=os.getenv("ADMIN_PASSWORD"))
    ap.add_argument("--timeout", type=int, default=120, help="per-request timeout (s)")
    ap.add_argument("--keep-unit", action="store_true", help="don't delete a temporary unit")
    ap.add_argument("--selfcheck", action="store_true", help="offline validator check; no network")
    args = ap.parse_args()

    if args.selfcheck:
        sys.exit(0 if selfcheck() else 1)

    if not args.base_url:
        ap.error("--base-url (or BASE_URL) is required (or use --selfcheck)")

    print(f"RPL adaptive interview — live smoke test\nTarget: {args.base_url}")
    try:
        ok = run(args)
    except KeyboardInterrupt:
        print("\ninterrupted"); sys.exit(130)

    passed = sum(1 for _, o, _ in _results if o)
    failed = sum(1 for _, o, _ in _results if not o)
    print(f"\n{'='*52}\nRESULT: {passed} passed, {failed} failed")
    if failed:
        print(f"{_RED}SMOKE TEST FAILED{_RST}")
        for n, o, d in _results:
            if not o:
                print(f"  - {n} {('· '+d) if d else ''}")
    else:
        print(f"{_GREEN}SMOKE TEST PASSED{_RST} — adaptive interview works end-to-end.")
    sys.exit(0 if ok and not failed else 1)


if __name__ == "__main__":
    main()
