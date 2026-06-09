"""
Résumé-driven adaptive / branching RPL engine.
=============================================

Turns the RPL conversation from a fixed question list into a process that
*adapts to the individual candidate* and *branches on the substance of each
answer*, while keeping every scenario anchored to a Performance Criterion
(validity) and continuously analysing answers for authenticity and AI usage.

Three stages:

  1. profile_candidate_experience  — parse the résumé into structured
     experience and map it onto every PC, assigning each a questioning
     priority (CONFIRM / PROBE / EXPLORE / GAP). Also captures an
     "authenticity baseline" from the candidate's own writing.

  2. build_adaptive_plan           — generate scenario-based opening questions
     for the focus PCs, grounded in the candidate's real employer/role/tools
     and anchored to each PC's benchmark.

  3. adaptive_scenario_turn        — the branching engine. Analyses each answer
     (content + résumé-consistency + AI-usage via detect_ai_usage), then
     selects a branch: DEEPEN / CHALLENGE / PIVOT / ADVANCE / GAP, and emits
     the next scenario.

Design notes:
  * AI-usage analysis is a *driver* of branching, not a bolt-on — a high AI
    probability forces a CHALLENGE branch (a curveball only lived experience
    answers well).
  * All candidate-supplied text is wrapped in <untrusted_*> delimiters with a
    guard instruction, so résumé / answer text cannot inject prompt
    instructions into the scoring.
  * Every scenario is tied to a specific PC + benchmark, preserving validity.
  * This is an AI assistant to a human assessor (HITL) — it never makes the
    final competency determination.
"""

import json, re, logging, asyncio
from typing import Optional

from .unit_registry import UnitOfCompetency
from .mapping_engine import detect_ai_usage
from .prompt_safety import INJECTION_GUARD, wrap_untrusted as _wrap, guard

logger = logging.getLogger(__name__)

HITL_REMINDER = (
    "You are an AI assistant to a qualified human assessor. You do NOT make the "
    "final competency determination — your analysis supports assessor judgement."
)

BRANCHES = {"DEEPEN", "CHALLENGE", "PIVOT", "ADVANCE", "GAP"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_json(raw: str):
    """Tolerant JSON extraction from a model reply (handles fences / prose)."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        for open_c, close_c in (("{", "}"), ("[", "]")):
            i, j = raw.find(open_c), raw.rfind(close_c)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(raw[i:j + 1])
                except Exception:
                    continue
        raise


async def _call(client, model: str, system: str, user: str,
                max_tokens: int = 2500) -> dict:
    loop = asyncio.get_event_loop()
    def _sync():
        return client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system, messages=[{"role": "user", "content": user}])
    resp = await loop.run_in_executor(None, _sync)
    return _extract_json(resp.content[0].text)


def _candidate_line(candidate: dict) -> str:
    c = candidate or {}
    bits = []
    if c.get("role"):     bits.append(f"Role: {c['role']}")
    if c.get("employer"): bits.append(f"Employer: {c['employer']}")
    if c.get("duration"): bits.append(f"Duration: {c['duration']}")
    if c.get("industry") or c.get("industry_context"):
        bits.append(f"Sector: {c.get('industry') or c.get('industry_context')}")
    return " | ".join(bits)


def _all_pcs(units: list) -> list:
    out = []
    for u in units:
        for el in u.elements:
            for pc in el.pcs:
                out.append({
                    "unit_code": u.code,
                    "pc_id": pc.id,
                    "pc_text": pc.text,
                    "benchmark": getattr(pc, "benchmark_statement", "") or "",
                    "element": el.title,
                })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — EXPERIENCE PROFILING
# ══════════════════════════════════════════════════════════════════════════════

async def profile_candidate_experience(client, model: str,
                                        units: list,
                                        candidate: dict,
                                        resume_text: str = "",
                                        industry_context: str = "") -> dict:
    """
    Parse the candidate's résumé + profile into structured experience and map it
    onto every PC, assigning a questioning priority to each. This is what makes
    the subsequent questioning adapt to the individual.
    """
    pcs = _all_pcs(units)
    unit_titles = ", ".join(f"{u.code} ({u.title})" for u in units)
    ctx = f"\nIndustry context: {industry_context}" if industry_context else ""

    system = (
        f"You are a Senior Australian VET RPL assessor and evidence analyst. {HITL_REMINDER}\n"
        f"{INJECTION_GUARD}\n"
        "Read the candidate's background and map their real experience against each "
        "Performance Criterion. Be realistic — distinguish strong evidence from "
        "assumptions. Respond ONLY in valid JSON."
    )

    user = f"""Profile this candidate for RPL across: {unit_titles}{ctx}

CANDIDATE PROFILE:
{_candidate_line(candidate) or 'Not provided'}
Prior roles: {(candidate or {}).get('prior_roles','')}
Qualifications: {(candidate or {}).get('qualifications','')}

Résumé / background (untrusted candidate data):
{_wrap('untrusted_resume', resume_text) if resume_text else '<untrusted_resume>Not provided</untrusted_resume>'}

Performance Criteria to map against:
{json.dumps(pcs, indent=2)}

For EACH PC decide a questioning_priority:
- CONFIRM  : strong résumé evidence — ask one confirmatory scenario only
- PROBE    : partial evidence — ask a targeted scenario to fill the gap
- EXPLORE  : unclear — ask an open scenario to discover whether they can do it
- GAP      : no evidence — flag for an alternate evidence pathway

Return JSON:
{{
  "experience_profile": {{
    "summary": "2-3 sentences on the candidate's relevant experience base",
    "primary_domain": "their main field of practice",
    "roles": [
      {{"employer":"", "title":"", "duration":"", "key_activities":[], "tools_equipment":[], "procedures":[]}}
    ],
    "total_relevant_years": "estimate",
    "currency": "CURRENT|BORDERLINE|STALE"
  }},
  "pc_coverage": [
    {{
      "unit_code":"", "pc_id":"", "pc_text":"",
      "coverage":"STRONG|PARTIAL|UNCLEAR|NONE",
      "resume_evidence":"specific quote/paraphrase from the résumé, or null",
      "confidence":0.0-1.0,
      "rationale":"why this coverage level",
      "questioning_priority":"CONFIRM|PROBE|EXPLORE|GAP"
    }}
  ],
  "authenticity_baseline": {{
    "specificity_level":"HIGH|MEDIUM|LOW",
    "domain_terms_used":["sector-specific terms the candidate actually used"],
    "writing_style_markers":["observations about their natural writing voice"],
    "notes":"baseline to compare later answers against for authenticity"
  }},
  "recommended_pathway":"PROCEED_RPL|PROCEED_WITH_GAPS|CONSIDER_TRAINING|DISCUSS_WITH_CANDIDATE",
  "evidence_to_request":["specific documents to gather before/alongside the conversation"]
}}"""

    try:
        result = await _call(client, model, system, user, max_tokens=4000)
    except Exception as e:
        logger.warning(f"Experience profiling failed: {e}")
        # Fail safe — produce a neutral profile that makes everything EXPLORE.
        return {
            "experience_profile": {"summary": f"Profiling unavailable: {e}",
                                   "primary_domain": "", "roles": [],
                                   "total_relevant_years": "", "currency": "UNKNOWN"},
            "pc_coverage": [
                {**pc, "coverage": "UNCLEAR", "resume_evidence": None,
                 "confidence": 0.0, "rationale": "Profiling unavailable",
                 "questioning_priority": "EXPLORE"}
                for pc in pcs
            ],
            "authenticity_baseline": {"specificity_level": "LOW",
                                      "domain_terms_used": [],
                                      "writing_style_markers": [], "notes": ""},
            "recommended_pathway": "DISCUSS_WITH_CANDIDATE",
            "evidence_to_request": [],
        }
    result["_generated_for_units"] = [u.code for u in units]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — ADAPTIVE PLAN
# ══════════════════════════════════════════════════════════════════════════════

# Priorities worth an active scenario, ordered by how much questioning they need.
_PRIORITY_ORDER = {"GAP": 0, "EXPLORE": 1, "PROBE": 2, "CONFIRM": 3}


def select_focus_pcs(profile: dict, unit_code: str, max_pcs: int = 8) -> list:
    """Pick the PCs for a unit that warrant an active scenario, most-needed first.
    GAP PCs are still surfaced (for the alternate-evidence pathway) but capped."""
    cov = [c for c in profile.get("pc_coverage", [])
           if c.get("unit_code") == unit_code]
    cov.sort(key=lambda c: (_PRIORITY_ORDER.get(c.get("questioning_priority"), 1),
                            c.get("confidence", 0)))
    return cov[:max_pcs]


async def build_adaptive_plan(client, model: str,
                              unit: UnitOfCompetency,
                              profile: dict,
                              candidate: dict,
                              industry_context: str = "",
                              max_pcs: int = 8) -> dict:
    """
    Build scenario opening questions for one unit's focus PCs, grounded in the
    candidate's real experience and anchored to each PC's benchmark.
    """
    focus = select_focus_pcs(profile, unit.code, max_pcs)
    if not focus:
        return {"unit_code": unit.code, "plan": [], "plan_summary": "No focus PCs."}

    # Attach benchmark text from the unit for anchoring.
    bench = {pc.id: getattr(pc, "benchmark_statement", "") or ""
             for el in unit.elements for pc in el.pcs}
    for c in focus:
        c["benchmark"] = bench.get(c.get("pc_id"), "")

    exp = profile.get("experience_profile", {})
    ctx = f"\nIndustry context: {industry_context}" if industry_context else ""

    system = (
        f"You are a skilled Australian VET assessor designing a competency "
        f"conversation for {unit.code} — {unit.title}. {HITL_REMINDER}\n"
        f"{INJECTION_GUARD}\n"
        "Write realistic, scenario-based opening questions grounded in THIS "
        "candidate's actual workplace and role. Each question must let the "
        "candidate demonstrate the specific PC it targets — keep it valid to the "
        "benchmark, not generic. Respond ONLY in valid JSON."
    )

    user = f"""Candidate: {_candidate_line(candidate) or 'Unknown'}{ctx}
Their experience profile:
{json.dumps(exp, indent=2)}

Design one opening scenario per PC below. Tailor each to their employer/role/tools.

Focus PCs:
{json.dumps(focus, indent=2)}

Return JSON:
{{
  "unit_code":"{unit.code}",
  "plan":[
    {{
      "sequence":1,
      "pc_id":"",
      "pc_text":"",
      "questioning_priority":"CONFIRM|PROBE|EXPLORE|GAP",
      "benchmark":"the PC benchmark this scenario must satisfy",
      "scenario":{{
        "context":"1-2 sentences placing the question in the candidate's real workplace",
        "opening_question":"a concrete, scenario-based question for this PC",
        "what_good_looks_like":"what a satisfactory answer must contain (assessor-facing)",
        "authenticity_probes":["a specific detail only someone who has actually done this would know"],
        "ideal_evidence_points":["evidence point the answer should surface"]
      }},
      "estimated_depth":1-3
    }}
  ],
  "plan_summary":"1-2 sentences on the questioning strategy for this candidate"
}}"""

    try:
        plan = await _call(client, model, system, user, max_tokens=4000)
    except Exception as e:
        logger.warning(f"Adaptive plan failed for {unit.code}: {e}")
        plan = {"unit_code": unit.code, "plan": [
            {"sequence": i + 1, "pc_id": c.get("pc_id"), "pc_text": c.get("pc_text"),
             "questioning_priority": c.get("questioning_priority", "EXPLORE"),
             "benchmark": c.get("benchmark", ""),
             "scenario": {
                 "context": "",
                 "opening_question": (
                     f"Tell me about a specific time in your role as "
                     f"{(candidate or {}).get('role','your job')} when you had to "
                     f"{(c.get('pc_text','') or '').rstrip('.').lower()}. "
                     f"What exactly did you do?"),
                 "what_good_looks_like": c.get("benchmark", ""),
                 "authenticity_probes": [], "ideal_evidence_points": []},
             "estimated_depth": 2}
            for i, c in enumerate(focus)
        ], "plan_summary": f"Fallback plan ({e})"}
    return plan


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — BRANCHING SCENARIO TURN
# ══════════════════════════════════════════════════════════════════════════════

async def adaptive_scenario_turn(client, model: str,
                                 unit: UnitOfCompetency,
                                 pc_id: str,
                                 scenario: dict,
                                 dialogue_history: list,
                                 latest_answer: str,
                                 profile: dict,
                                 candidate: dict,
                                 turn_number: int = 1,
                                 max_turns: int = 4,
                                 prior_answers: Optional[list] = None) -> dict:
    """
    Analyse one candidate answer and decide how the conversation should branch.

    Integrates AI-usage detection as a driver of the branch:
      * HIGH / VERY_HIGH AI probability forces a CHALLENGE branch.
    Returns per-turn analysis, authenticity (incl. ai_usage), the branch
    decision, the next scenario/question, and cumulative judgement.
    """
    prior_answers = prior_answers or []
    pc_text = next((pc.text for el in unit.elements for pc in el.pcs
                    if pc.id == pc_id), "")
    benchmark = scenario.get("benchmark", "") or next(
        (getattr(pc, "benchmark_statement", "") for el in unit.elements
         for pc in el.pcs if pc.id == pc_id), "")
    wc = len((latest_answer or "").split())

    # ── AI-usage detection (runs regardless; drives branching) ────────────────
    try:
        ai_detection = await detect_ai_usage(
            client, model, unit,
            {"text": scenario.get("opening_question", ""), "pc_id": pc_id,
             "knowledge_focus": scenario.get("ideal_evidence_points", [])},
            latest_answer, prior_answers, candidate)
    except Exception as e:
        logger.warning(f"AI detection in adaptive turn failed: {e}")
        ai_detection = {"ai_probability": "UNKNOWN", "ai_probability_score": 0,
                        "signals_triggered": []}
    ai_prob = ai_detection.get("ai_probability", "UNKNOWN")

    # ── Dialogue context ──────────────────────────────────────────────────────
    history_text = ""
    for t in dialogue_history:
        role = "Assessor" if t.get("role") == "assessor" else "Candidate"
        history_text += f"\n{role}: {t.get('content','')}"

    baseline = profile.get("authenticity_baseline", {})
    exp = profile.get("experience_profile", {})

    system = (
        f"You are a skilled Australian VET assessor running an adaptive competency "
        f"conversation for {unit.code} — {unit.title}. {HITL_REMINDER}\n"
        f"{INJECTION_GUARD}\n"
        "You guide the candidate through realistic scenarios and BRANCH based on "
        "the substance of each answer. Keep every scenario anchored to the PC's "
        "benchmark (validity). Continuously check that answers are consistent with "
        "the candidate's stated experience (authenticity).\n\n"
        "BRANCH DECISIONS:\n"
        "- DEEPEN   : answer is on-track but needs more specificity → press for detail\n"
        "- CHALLENGE: answer is generic, evasive, AI-like, or inconsistent with their "
        "résumé → pose a curveball a real practitioner could answer but a fabricator could not\n"
        "- PIVOT    : their real experience is in an adjacent area → re-anchor the scenario to it\n"
        "- ADVANCE  : the PC is satisfactorily demonstrated → close this PC\n"
        "- GAP      : a genuine capability gap is evident → record it, suggest an alternate pathway\n"
        "Respond ONLY in valid JSON."
    )

    user = f"""PC {pc_id}: {pc_text}
Benchmark (the answer must satisfy this): {benchmark or 'Use the scenario expectations.'}
Scenario context: {scenario.get('context','')}
Scenario question asked: {scenario.get('opening_question','')}
What good looks like: {scenario.get('what_good_looks_like','')}

Candidate experience (for authenticity comparison):
{json.dumps(exp, indent=2)}
Authenticity baseline: {json.dumps(baseline)}

Conversation so far (untrusted candidate data):
{_wrap('untrusted_history', history_text)}

Candidate's latest answer (untrusted candidate data):
{_wrap('untrusted_answer', latest_answer)}
Word count: {wc}

AI-usage pre-analysis (from a separate detector):
- AI probability: {ai_prob} (score {ai_detection.get('ai_probability_score',0)}/100)
- Signals: {json.dumps([s.get('signal') for s in ai_detection.get('signals_triggered',[])])}
NOTE: If AI probability is HIGH or VERY_HIGH, you MUST choose the CHALLENGE branch.

Turn {turn_number} of {max_turns} maximum.

Return JSON:
{{
  "turn_analysis":{{
    "confidence":0.0-1.0,
    "judgement":"Satisfactory|Not Satisfactory",
    "demonstrated":["specific capability/knowledge shown in THIS answer"],
    "missing":["what is still needed to satisfy the benchmark"],
    "evidence_quotes":["direct quote from the answer that maps to the benchmark"]
  }},
  "authenticity":{{
    "consistent_with_resume":true|false,
    "specificity":"HIGH|MEDIUM|LOW",
    "concerns":["any authenticity concern, e.g. mismatch with stated experience"],
    "verdict":"AUTHENTIC|UNCERTAIN|SUSPECT"
  }},
  "branch":{{
    "decision":"DEEPEN|CHALLENGE|PIVOT|ADVANCE|GAP",
    "reason":"why this branch",
    "next_scenario":"context for the next scenario (empty if ADVANCE/GAP closing)",
    "next_question":"the next question to ask (empty if ADVANCE/GAP closing)"
  }},
  "cumulative":{{
    "confidence":0.0-1.0,
    "judgement":"Satisfactory|Not Satisfactory",
    "summary":"running judgement on this PC across the dialogue"
  }},
  "encouragement":"one supportive sentence acknowledging what they got right",
  "assessor_note":"what the human assessor should verify before accepting this PC"
}}"""

    try:
        result = await _call(client, model, system, user, max_tokens=1800)
    except Exception as e:
        logger.warning(f"Adaptive turn failed: {e}")
        result = {
            "turn_analysis": {"confidence": 0.0, "judgement": "Not Satisfactory",
                              "demonstrated": [], "missing": ["Analysis unavailable"],
                              "evidence_quotes": []},
            "authenticity": {"consistent_with_resume": True, "specificity": "LOW",
                             "concerns": [], "verdict": "UNCERTAIN"},
            "branch": {"decision": "DEEPEN", "reason": str(e),
                       "next_scenario": "",
                       "next_question": "Can you give me a specific example with more detail?"},
            "cumulative": {"confidence": 0.0, "judgement": "Not Satisfactory",
                           "summary": "Analysis unavailable"},
            "encouragement": "", "assessor_note": "Manual review required.",
        }

    # ── Attach AI-usage and apply deterministic safety overrides ──────────────
    result.setdefault("authenticity", {})["ai_usage"] = {
        "probability": ai_prob,
        "score": ai_detection.get("ai_probability_score", 0),
        "signals_triggered": ai_detection.get("signals_triggered", []),
        "assessor_guidance": ai_detection.get("assessor_guidance", ""),
    }

    branch = result.get("branch", {})
    decision = branch.get("decision", "DEEPEN")
    if decision not in BRANCHES:
        decision = "DEEPEN"
    cum_conf = result.get("cumulative", {}).get("confidence", 0) or 0

    # 1) High AI probability → force a CHALLENGE (unless a real gap is already found).
    if ai_prob in ("HIGH", "VERY_HIGH") and decision != "GAP" and turn_number < max_turns:
        if decision != "CHALLENGE":
            decision = "CHALLENGE"
            branch["reason"] = (f"AI-usage probability {ai_prob} — verifying authenticity "
                                f"with a scenario that requires lived experience.")
            if not branch.get("next_question"):
                branch["next_question"] = (
                    "Walk me through a specific instance from your own work — name the "
                    "site, the people involved, and exactly what you did and why.")

    # 2) Strong, authentic answer → advance even if the model wanted to keep going.
    elif (cum_conf >= 0.75
          and result.get("authenticity", {}).get("verdict") != "SUSPECT"
          and ai_prob in ("LOW", "MEDIUM", "UNKNOWN")
          and decision in ("DEEPEN", "CHALLENGE")):
        decision = "ADVANCE"
        branch["next_question"] = ""
        branch["next_scenario"] = ""

    # 3) Model wanted to advance but evidence is thin and turns remain → deepen.
    elif decision == "ADVANCE" and cum_conf < 0.6 and turn_number < max_turns:
        decision = "DEEPEN"
        if not branch.get("next_question"):
            branch["next_question"] = (
                "That's a start — can you give me the specific details: what you "
                "did step by step, and how you knew it worked?")

    # 4) Out of turns → close the PC regardless.
    if turn_number >= max_turns and decision in ("DEEPEN", "CHALLENGE", "PIVOT"):
        decision = "ADVANCE" if cum_conf >= 0.6 else "GAP"
        branch["next_question"] = ""
        branch["next_scenario"] = ""

    branch["decision"] = decision
    result["branch"] = branch
    result["next_action"] = "CLOSE_PC" if decision in ("ADVANCE", "GAP") else "CONTINUE"
    result["turn_number"] = turn_number
    result["pc_id"] = pc_id
    return result


# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ-RELEVANCE HINTS FOR (GENERIC) KNOWLEDGE QUESTIONS
# Keeps the underpinning-knowledge question unchanged, but adds a one-line hint
# pointing the candidate to the part of THEIR experience they can draw on.
# One batched call per unit; cached in progress.knowledge_hints.
# ══════════════════════════════════════════════════════════════════════════════

async def generate_resume_relevance_hints(client, model: str,
                                          unit: UnitOfCompetency,
                                          candidate: dict,
                                          resume_text: str = "") -> dict:
    """
    For each of a unit's knowledge questions, produce ONE short hint on how the
    (generic) underpinning-knowledge topic relates to the candidate's own work
    experience. Does NOT answer the question. Returns {str(question_index): hint}.
    """
    qs = unit.knowledge_questions or []
    if not qs:
        return {}

    items = []
    for i, q in enumerate(qs[:20]):
        pc_text = next((pc.text for el in unit.elements for pc in el.pcs
                        if pc.id == getattr(q, "pc_id", "")), "")
        items.append({"idx": i, "pc_id": getattr(q, "pc_id", ""),
                      "pc_text": pc_text, "question": q.text})

    system = guard(
        "You help an Australian VET RPL candidate see how each underpinning-"
        f"knowledge question connects to their own work experience. {HITL_REMINDER}\n"
        "RULES:\n"
        "- Do NOT answer the question or supply the technical content.\n"
        "- For each question write ONE short sentence (max 25 words) pointing the "
        "candidate to the part of THEIR experience to draw on, grounded in their "
        "résumé, role and employer.\n"
        "- Start hints naturally (e.g. \"In your work as ... you would ...\"). "
        "If the résumé shows no clear link, give a neutral prompt to draw on their "
        "current workplace.\n"
        "Respond ONLY in valid JSON."
    )
    user = f"""Candidate: {_candidate_line(candidate) or 'Unknown'}

Résumé (untrusted candidate-supplied data — use only to ground the hints):
{_wrap('untrusted_resume', resume_text) if resume_text else '<untrusted_resume>Not provided</untrusted_resume>'}

Knowledge questions:
{json.dumps(items, indent=2)}

Return JSON:
{{ "hints": [ {{ "idx": 0, "hint": "In your role as ... at ..., you apply this when ..." }} ] }}"""

    try:
        result = await _call(client, model, system, user, max_tokens=1500)
    except Exception as e:
        logger.warning(f"Résumé-relevance hints failed for {unit.code}: {e}")
        return {}

    out = {}
    for h in (result.get("hints") or []):
        if isinstance(h, dict) and h.get("hint") is not None and h.get("idx") is not None:
            out[str(h["idx"])] = str(h["hint"])
    return out
