"""
Server-side PDF rendering of the final RPL assessment evidence record.

Uses reportlab (pure Python — no system/native deps, so it builds cleanly in the
Cloud Run container). Takes the assembled record dict from
main._build_final_record() and produces a multi-page, audit-ready PDF.
"""
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable, Image)
import base64

NAVY = colors.HexColor("#1F2060")
GREY = colors.HexColor("#6b7280")
GREEN = colors.HexColor("#166534")
AMBER = colors.HexColor("#92400e")
RED = colors.HexColor("#991b1b")
LIGHT = colors.HexColor("#f3f4f6")

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontSize=16, textColor=NAVY, spaceAfter=2)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontSize=12, textColor=NAVY, spaceBefore=12, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=_ss["Heading3"], fontSize=10, textColor=NAVY, spaceBefore=8, spaceAfter=2)
BODY = ParagraphStyle("Body", parent=_ss["BodyText"], fontSize=9, leading=12, spaceAfter=2)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8, textColor=GREY, leading=10)
MONO = ParagraphStyle("Mono", parent=BODY, fontName="Courier", fontSize=8)


def e(v) -> str:
    return escape("" if v is None else str(v))


def _date(s) -> str:
    return e(s) if s else "—"


def _para(text, style=BODY):
    return Paragraph(text, style)


def _kv_table(rows):
    data = [[_para(f"<b>{e(k)}</b>", SMALL), _para(e(v), BODY)] for k, v in rows]
    t = Table(data, colWidths=[40 * mm, 130 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _section(story, title):
    story.append(Paragraph(e(title), H2))
    story.append(HRFlowable(width="100%", thickness=0.6, color=NAVY, spaceAfter=4))


def _bullets(story, items, style=BODY, prefix="• "):
    for it in (items or []):
        if it:
            story.append(_para(prefix + e(it), style))


def _pct(v):
    try:
        return f"{round(float(v) * 100) if float(v) <= 1 else round(float(v))}%"
    except (TypeError, ValueError):
        return ""


# ── section builders ──────────────────────────────────────────────────────────

def _header(story, r):
    story.append(_para("RPL Assessment — Evidence Record", H1))
    story.append(_para(f"{e(r.get('rto'))} · Generated {_date(r.get('generated_at'))} "
                       f"by {e((r.get('generated_by') or {}).get('name'))}", SMALL))
    story.append(Spacer(1, 6))
    c = r.get("candidate") or {}
    a = r.get("assessment") or {}
    ass = r.get("assessor") or {}
    story.append(_kv_table([
        ("Candidate", c.get("name")), ("Email", c.get("email")),
        ("Employer", c.get("employer")), ("Role", c.get("role")),
        ("Duration", c.get("duration")),
        ("Status", a.get("status")), ("Created", _date(a.get("created_at"))),
        ("Submitted", _date(a.get("submitted_at"))), ("Completed", _date(a.get("completed_at"))),
        ("Assessor", ass.get("name")),
    ]))
    units = r.get("units") or []
    if units:
        story.append(Spacer(1, 4))
        story.append(_para("<b>Units of competency</b>", BODY))
        _bullets(story, [f"{u.get('code')} — {u.get('title')}"
                         + (f" ({u.get('training_package')})" if u.get('training_package') else "")
                         for u in units])


def _privacy(story, r):
    _section(story, "Privacy & consent (APP 5)")
    pc = r.get("privacy_consent")
    if pc and pc.get("acknowledged"):
        story.append(_para(f"Candidate acknowledged the Privacy Collection Notice "
                           f"(v{e(pc.get('notice_version', '1.0'))}) on {_date(pc.get('timestamp'))}. "
                           f"APPs covered: {e(', '.join(pc.get('apps_covered', [])))}.", BODY))
    else:
        story.append(_para("No recorded privacy acknowledgement.", BODY))
    story.append(_para("Data stored in Australia (Google Cloud Sydney). Retention: 2 years from "
                       "completion (ASQA). AI processing: Zero Data Retention.", SMALL))


def _checklist(story, r):
    cl = r.get("self_assessment_checklist") or {}
    if not cl:
        return
    _section(story, "Self-assessment checklist")
    rows = [[_para("<b>Unit</b>", SMALL), _para("<b>PC</b>", SMALL), _para("<b>Self-rating</b>", SMALL)]]
    lbl = {"f": "Frequently", "s": "Sometimes", "n": "Not yet"}
    for unit, pcs in cl.items():
        if not isinstance(pcs, dict):
            continue
        for pc, val in pcs.items():
            rows.append([_para(e(unit), MONO), _para(e(pc), MONO), _para(e(lbl.get(val, val)), BODY)])
    if len(rows) > 1:
        t = Table(rows, colWidths=[40 * mm, 25 * mm, 105 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        story.append(t)


def _documents(story, r):
    docs = r.get("documents") or {}
    notes = docs.get("candidate_notes") or {}
    uploads = docs.get("uploads") or {}
    if not notes and not uploads:
        return
    _section(story, "Documents & evidence supplied")
    if notes.get("resume"):
        story.append(_para("<b>Résumé / background note</b>", H3))
        story.append(_para(e(notes.get("resume")), BODY))
    if notes.get("position"):
        story.append(_para("<b>Position description note</b>", H3))
        story.append(_para(e(notes.get("position")), BODY))
    up = [v.get("name") or v.get("filename") or k for k, v in uploads.items()
          if isinstance(v, dict)]
    if up:
        story.append(_para("<b>Uploaded files</b>", H3))
        _bullets(story, up)


def _knowledge(story, r):
    responses = r.get("knowledge_responses") or {}
    analyses = r.get("knowledge_analyses") or {}
    if not responses:
        return
    _section(story, "Knowledge questions — answers & AI analysis")
    for unit_code, unit_resp in responses.items():
        if not isinstance(unit_resp, dict):
            continue
        unit_anal = analyses.get(unit_code, {}) or {}
        story.append(_para(f"<b>{e(unit_code)}</b>", H3))
        for q_idx, answer in unit_resp.items():
            if not answer or not str(q_idx).isdigit():
                continue
            an = unit_anal.get(str(q_idx), {}) or {}
            qtext = an.get("question_text") or f"Question {int(q_idx) + 1}"
            story.append(_para(f"<b>Q{int(q_idx) + 1}.</b> {e(qtext)}", BODY))
            story.append(_para("<b>Answer:</b> " + e(answer), SMALL))
            if an:
                judg = an.get("judgement") or ""
                score = an.get("overall_score_percent")
                meets = an.get("meets_requirement") or ""
                head = f"<b>AI analysis:</b> {e(judg)}"
                if score is not None:
                    head += f" · {e(score)}%"
                if meets:
                    head += f" · {e(meets.replace('_', ' '))}"
                story.append(_para(head, BODY))
                if an.get("commentary"):
                    story.append(_para(e(an.get("commentary")), SMALL))
                dem = [x for x in (an.get("what_the_answer_demonstrates") or []) if x]
                mis = [x for x in (an.get("what_is_missing") or []) if x]
                if dem:
                    story.append(_para("Demonstrates: " + e("; ".join(map(str, dem))), SMALL))
                if mis:
                    story.append(_para("Still needed: " + e("; ".join(map(str, mis))), SMALL))
            story.append(Spacer(1, 4))


def _conversations(story, r):
    recs = [x for x in (r.get("competency_conversations") or []) if (x or {}).get("dialogue")]
    adaptive = r.get("adaptive_interview") or []
    if not recs and not adaptive:
        return
    _section(story, "Competency conversation & adaptive interview")
    for rec in recs:
        story.append(_para(f"<b>{e(rec.get('unit'))} · PC {e(rec.get('pc'))}</b> — "
                           f"{e(rec.get('final_judgement') or '')} {_pct(rec.get('final_confidence'))}", BODY))
        for t in (rec.get("dialogue") or []):
            who = "AI guide" if t.get("role") == "assessor" else "Candidate"
            story.append(_para(f"<b>{who}:</b> " + e(t.get("content")), SMALL))
        story.append(Spacer(1, 3))
    for rec in adaptive:
        if not (rec or {}).get("dialogue"):
            continue
        trail = " → ".join((b.get("decision") for b in (rec.get("branch_trail") or []) if b.get("decision")))
        story.append(_para(f"<b>Adaptive — {e(rec.get('unit'))} · PC {e(rec.get('pc'))}</b> — "
                           f"{e(rec.get('final_judgement') or '')} {_pct(rec.get('final_confidence'))}"
                           + (f" · path: {e(trail)}" if trail else ""), BODY))
        for t in (rec.get("dialogue") or []):
            who = "AI guide" if t.get("role") == "assessor" else "Candidate"
            story.append(_para(f"<b>{who}:</b> " + e(t.get("content")), SMALL))
        story.append(Spacer(1, 3))


def _mapping(story, r):
    mappings = []
    if r.get("ai_mapping"):
        mappings.append(r["ai_mapping"])
    for v in (r.get("ai_mappings") or {}).values():
        if v and v not in mappings:
            mappings.append(v)
    if not mappings:
        return
    _section(story, "AI mapping guide (decision support)")
    for m in mappings:
        o = m.get("overall") or {}
        audit = m.get("audit") or {}
        story.append(_para(f"<b>{e(audit.get('unit_code') or '')}</b> — signal: "
                           f"{e((o.get('signal') or '').replace('_', ' '))} · "
                           f"confidence {e(o.get('aggregate_confidence'))}", H3))
        if o.get("narrative"):
            story.append(_para(e(o.get("narrative")), BODY))
        rows = [[_para("<b>PC</b>", SMALL), _para("<b>Verdict</b>", SMALL),
                 _para("<b>Judgement</b>", SMALL), _para("<b>Conf.</b>", SMALL)]]
        for el in (m.get("elements") or []):
            for pc in (el.get("pcs") or []):
                rows.append([_para(e(pc.get("id")), MONO), _para(e(pc.get("verdict")), SMALL),
                             _para(e(pc.get("judgement")), SMALL), _para(_pct(pc.get("confidence")), SMALL)])
        if len(rows) > 1:
            t = Table(rows, colWidths=[20 * mm, 40 * mm, 70 * mm, 20 * mm])
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            story.append(t)
        if o.get("pathway_recommendation") or m.get("pathway_recommendation"):
            story.append(_para("Pathway: " + e(o.get("pathway_recommendation") or m.get("pathway_recommendation")), SMALL))
        story.append(Spacer(1, 4))


def _ai_usage(story, r):
    reports = [v for v in (r.get("ai_usage_report") or {}).values() if v and v.get("response_analyses")]
    if not reports:
        return
    _section(story, "AI-usage & authenticity report")
    for rep in reports:
        story.append(_para(f"<b>{e(rep.get('unit_code'))}</b> — overall risk: "
                           f"{e(rep.get('overall_ai_risk'))} · "
                           f"{e(rep.get('high_risk_responses'))} flagged of "
                           f"{e(rep.get('responses_analysed'))}", H3))
        if rep.get("overall_guidance"):
            story.append(_para(e(rep.get("overall_guidance")), SMALL))
        for ra in (rep.get("response_analyses") or []):
            det = ra.get("detection") or {}
            if det.get("ai_probability") in ("HIGH", "VERY_HIGH"):
                story.append(_para(f"⚠ {e(ra.get('source'))} {e(ra.get('pc_id') or '')}: "
                                   f"{e(det.get('ai_probability'))} ({e(det.get('ai_probability_score'))}%)", SMALL))
        story.append(_para("AI detection is probabilistic and supports — does not replace — assessor judgement.", SMALL))
        story.append(Spacer(1, 4))


def _portfolio(story, r):
    pr = r.get("portfolio_review")
    if not pr:
        return
    _section(story, "Evidence portfolio review (HITL)")
    yn = lambda v: "Yes" if v is True else "No" if v is False else "—"
    story.append(_para(f"Reviewer: <b>{e(pr.get('reviewer'))}</b> · {_date(pr.get('reviewed_at'))}", BODY))
    story.append(_para(f"Portfolio sufficient: <b>{yn(pr.get('portfolio_sufficient'))}</b> · "
                       f"HITL confirmed: <b>{yn(pr.get('hitl_confirmed'))}</b>", BODY))
    if pr.get("overall_portfolio_comment"):
        story.append(_para(e(pr.get("overall_portfolio_comment")), SMALL))
    if pr.get("hitl_declaration"):
        story.append(_para("<i>" + e(pr.get("hitl_declaration")) + "</i>", SMALL))


def _determinations(story, r):
    _section(story, "Determination")
    dets = r.get("determinations") or []
    if not dets:
        story.append(_para("No formal determination recorded yet.", BODY))
        return
    for d in dets:
        story.append(_para(f"<b>{e(d.get('unit_code'))} — {e(d.get('overall_determination'))}</b>", H3))
        story.append(_para("<b>Rationale:</b> " + e(d.get("assessor_rationale")), BODY))
        if d.get("reasonable_adjustments"):
            story.append(_para("<b>Reasonable adjustments:</b> " + e(d.get("reasonable_adjustments")), BODY))
        pcs = d.get("pc_determinations") or []
        if pcs:
            rows = [[_para("<b>PC</b>", SMALL), _para("<b>Judgement</b>", SMALL), _para("<b>Notes</b>", SMALL)]]
            for p in pcs:
                rows.append([_para(e(p.get("pc_id")), MONO), _para(e(p.get("assessor_judgement")), SMALL),
                             _para(e(p.get("assessor_notes") or p.get("override_reason") or ""), SMALL)])
            t = Table(rows, colWidths=[20 * mm, 45 * mm, 105 * mm])
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            story.append(t)
        story.append(_para(f"Assessor: {e(d.get('assessor_name'))} · {_date(d.get('determined_at'))} "
                           f"· Source: {e(d.get('source') or 'HUMAN_ASSESSOR')}", SMALL))
        story.append(Spacer(1, 4))


def _signature_flowable(sig):
    """Render a drawn signature image, or the typed name in a script-like style."""
    if sig.get("method") == "drawn" and (sig.get("image") or "").startswith("data:image"):
        try:
            b64 = sig["image"].split(",", 1)[1]
            img = Image(BytesIO(base64.b64decode(b64)))
            # Scale to a sensible signature box, preserving aspect ratio.
            max_w, max_h = 70 * mm, 22 * mm
            ratio = min(max_w / img.imageWidth, max_h / img.imageHeight, 1)
            img.drawWidth = img.imageWidth * ratio
            img.drawHeight = img.imageHeight * ratio
            return img
        except Exception:
            pass
    return _para(f"<i>{e(sig.get('name'))}</i>", ParagraphStyle(
        "Sig", parent=BODY, fontName="Helvetica-Oblique", fontSize=16, leading=18))


def _declaration(story, r):
    _section(story, "Human-in-the-loop declaration")
    story.append(_para(e(r.get("hitl_statement")), BODY))
    story.append(Spacer(1, 10))
    ass = (r.get("assessor") or {}).get("name") or "—"
    sig = r.get("signature") or {}
    if sig.get("name"):
        when = (sig.get("signed_at") or "")[:19].replace("T", " ")
        method = "drawn signature" if sig.get("method") == "drawn" else "typed e-signature"
        tbl = Table([[_para(f"Assessor: <b>{e(ass or sig.get('name'))}</b>", BODY),
                      _signature_flowable(sig)]],
                    colWidths=[70 * mm, 90 * mm])
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
        story.append(tbl)
        story.append(Spacer(1, 3))
        story.append(_para(
            f"Electronically signed by <b>{e(sig.get('name'))}</b> on {e(when)} UTC "
            f"({method}). {e(sig.get('statement'))}", SMALL))
    else:
        tbl = Table([[_para(f"Assessor: <b>{e(ass)}</b>", BODY),
                      _para("Signature: ______________________", BODY),
                      _para("Date: ____________", BODY)]],
                    colWidths=[60 * mm, 70 * mm, 40 * mm])
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(tbl)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 12 * mm, "RPL Assessment — Evidence Record · ABC Training RTO #5800")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_final_record_pdf(record: dict) -> bytes:
    """Render the assembled final-record dict to PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            title="RPL Assessment — Evidence Record")
    story = []
    _header(story, record)
    _privacy(story, record)
    _checklist(story, record)
    _documents(story, record)
    _knowledge(story, record)
    _conversations(story, record)
    _mapping(story, record)
    _ai_usage(story, record)
    _portfolio(story, record)
    _determinations(story, record)
    _declaration(story, record)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
