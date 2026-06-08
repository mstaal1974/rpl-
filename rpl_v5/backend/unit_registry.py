"""
Unit of Competency Registry
Loads from /units/ JSON files on startup.
Imports any unit live from training.gov.au on demand.
"""
import json, logging, os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
UNITS_DIR = Path(__file__).parent.parent / "units"


class EvidenceRecord(BaseModel):
    evidence_items: list[str] = []        # quotes from candidate answer mapped to this PC
    assessor_comments: str = ""
    judgement: Optional[str] = None       # "Satisfactory" | "Not Satisfactory" | None
    gap_notes: str = ""                   # what is missing; used to generate targeted follow-ups


class PerformanceCriteria(BaseModel):
    id: str
    text: str
    element_id: str
    analysis_prompt: str = ""    # "What valid, sufficient, authentic and current evidence shows..."
    benchmark_statement: str = ""  # "The candidate demonstrates that they can..."
    evidence_record: EvidenceRecord = Field(default_factory=EvidenceRecord)


class Element(BaseModel):
    id: str
    title: str
    analysis_focus: str = ""     # "Analyse evidence against Element N: <title>"
    pcs: list[PerformanceCriteria]


class KnowledgeRequirement(BaseModel):
    id: str
    category: str
    text: str


class SkillRequirement(BaseModel):
    id: str
    category: str
    text: str


class EvidenceGuideItem(BaseModel):
    title: str
    priority: str       # "priority" | "recommended"
    pc_refs: list[str]
    icon: str
    description: str
    acceptable_forms: list[str]


class ModelAnswerGuide(BaseModel):
    expected_knowledge_points: list[str] = []
    acceptable_answer_examples: list[str] = []
    strong_answer_indicators: list[str] = []
    weak_answer_indicators: list[str] = []
    common_gaps_or_errors: list[str] = []

class AssessorEvaluationFramework(BaseModel):
    what_to_look_for: list[str] = []
    minimum_expected_knowledge: list[str] = []
    indicators_of_partial_understanding: list[str] = []
    indicators_of_strong_understanding: list[str] = []

class KnowledgeQuestion(BaseModel):
    num: int
    element_ref: str
    pc_refs: list[str]
    pc_id: str = ""
    text: str
    hint: str
    difficulty_level: str = "Applied"    # Basic | Applied | Advanced
    question_purpose: str = ""
    why_task_specific: str = ""
    benchmark_statement: str = ""
    analysis_prompt: str = ""
    practical_task_interpretation: str = ""
    knowledge_focus: list[str] = []
    workplace_context_examples: list[str] = []
    model_answer_guide: ModelAnswerGuide = Field(default_factory=ModelAnswerGuide)
    assessor_framework: AssessorEvaluationFramework = Field(default_factory=AssessorEvaluationFramework)


class UnitOfCompetency(BaseModel):
    code: str
    title: str
    training_package: str
    training_package_name: str
    application: str
    competent_person_statement: str
    elements: list[Element]
    knowledge_requirements: list[KnowledgeRequirement]
    skill_requirements: list[SkillRequirement]
    evidence_guide: list[EvidenceGuideItem] = []
    knowledge_questions: list[KnowledgeQuestion] = []
    currency_years: int = 5
    source: str = "manual"
    version: str = "1.0"
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UnitRegistry:
    def __init__(self):
        self._units: dict[str, UnitOfCompetency] = {}
        self._load_local()

    def _load_local(self):
        if not UNITS_DIR.exists():
            logger.warning(f"Units directory not found: {UNITS_DIR}")
            return
        for f in UNITS_DIR.rglob("*.json"):
            if "_cache" in str(f):
                continue
            try:
                unit = UnitOfCompetency(**json.loads(f.read_text()))
                self._units[unit.code.upper()] = unit
                logger.info(f"Loaded unit: {unit.code} — {unit.title}")
            except Exception as e:
                logger.error(f"Failed to load {f}: {e}")
        logger.info(f"Registry: {len(self._units)} units loaded")

    def get(self, code: str) -> Optional[UnitOfCompetency]:
        return self._units.get(code.upper())

    def list_all(self) -> list[dict]:
        return [
            {"code": u.code, "title": u.title, "training_package": u.training_package,
             "training_package_name": u.training_package_name,
             "element_count": len(u.elements),
             "pc_count": sum(len(e.pcs) for e in u.elements),
             "source": u.source, "last_updated": u.last_updated}
            for u in self._units.values()
        ]

    def add(self, unit: UnitOfCompetency) -> bool:
        self._units[unit.code.upper()] = unit
        self._persist(unit)
        return True

    def _persist(self, unit: UnitOfCompetency):
        d = UNITS_DIR / unit.training_package.lower()
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{unit.code}.json"
        p.write_text(json.dumps(unit.model_dump(), indent=2))
        logger.info(f"Saved unit to {p}")

    def delete(self, code: str) -> bool:
        code = code.upper()
        if code not in self._units:
            return False
        unit = self._units.pop(code)
        p = UNITS_DIR / unit.training_package.lower() / f"{unit.code}.json"
        if p.exists():
            p.unlink()
        return True

    @property
    def count(self) -> int:
        return len(self._units)


registry = UnitRegistry()


async def sync_registry_from_firestore():
    """
    On startup, pull any units previously uploaded via the admin portal
    from Firestore. This means uploaded units survive every redeployment —
    no need to re-upload after each deploy.
    """
    import os, json, asyncio
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        logger.info("No GCP project set — skipping Firestore registry sync")
        return
    try:
        from google.cloud import firestore
        db = firestore.Client(project=project)
        col = db.collection("rpl_unit_registry")
        count = 0
        # Stream all docs — each doc is one unit, keyed by unit code
        for doc in col.stream():
            data = doc.to_dict()
            if not data or not data.get("code"):
                continue
            # Only load if not already present from local JSON files
            # (local JSON files take precedence — they are the pre-baked packages)
            if not registry.get(data["code"]):
                try:
                    unit = UnitOfCompetency(**data)
                    registry.add(unit)
                    count += 1
                except Exception as e:
                    logger.warning(f"Skipping Firestore unit {data.get('code')}: {e}")
        if count:
            logger.info(f"Restored {count} units from Firestore (total registry: {registry.count})")
        else:
            logger.info(f"Firestore registry sync complete — no additional units (registry: {registry.count})")
    except Exception as e:
        logger.warning(f"Firestore registry sync failed (non-fatal): {e}")


async def import_from_tgau(unit_code: str) -> UnitOfCompetency:
    """
    Import unit from TGA staging SOAP service using urllib (not httpx).
    Exact format verified working: no xmlns on body elements, minimal request.
    """
    import urllib.request as _req
    import urllib.error   as _err
    import asyncio        as _asyncio

    code  = unit_code.upper().strip()
    cache = UNITS_DIR / "_cache" / f"{code}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        age = (datetime.now(timezone.utc) - datetime.fromtimestamp(
            cache.stat().st_mtime, tz=timezone.utc)).days
        if age < 30:
            logger.info(f"Returning cached {code}")
            return UnitOfCompetency(**json.loads(cache.read_text()))

    user = os.environ.get("TGA_USER", "WebService.Read")
    pw   = os.environ.get("TGA_PASS",  "Asdf098")
    WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
    PT   = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText"
    URL  = "https://ws.staging.training.gov.au/Deewr.Tga.Webservices/TrainingComponentServiceV12.svc/Training"

    # Build body using string concat — NO f-string to avoid any xmlns injection
    body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope'
        ' xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"'
        ' xmlns:wsse="' + WSSE + '">'
        '<s:Header>'
        '<wsse:Security>'
        '<wsse:UsernameToken>'
        '<wsse:Username>' + user + '</wsse:Username>'
        '<wsse:Password Type="' + PT + '">' + pw + '</wsse:Password>'
        '</wsse:UsernameToken>'
        '</wsse:Security>'
        '</s:Header>'
        '<s:Body>'
        '<GetDetails>'           # NO xmlns here — this was the bug
        '<request>'
        '<Code>' + code + '</Code>'
        '<ShowReleases>true</ShowReleases>'
        '</request>'
        '</GetDetails>'
        '</s:Body>'
        '</s:Envelope>'
    ).encode("utf-8")

    def _call_soap():
        r = _req.Request(URL, data=body, headers={
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction":   "http://training.gov.au/ITrainingComponentService/GetDetails",
        })
        try:
            with _req.urlopen(r, timeout=25) as resp:
                return resp.read().decode("utf-8"), None
        except _err.HTTPError as e:
            return e.read().decode("utf-8"), f"HTTP {e.code}"
        except Exception as e:
            return None, str(e)

    data       = None
    soap_error = None

    try:
        loop = _asyncio.get_event_loop()
        xml_text, err = await loop.run_in_executor(None, _call_soap)

        logger.info(f"SOAP response length: {len(xml_text or '')} err={err}")
        logger.info(f"SOAP preview: {(xml_text or '')[:400]}")

        if err and xml_text is None:
            raise ValueError(f"Network error: {err}")

        if "<Fault>" in (xml_text or "") or "<s:Fault>" in (xml_text or ""):
            fault = _extract_soap_fault(xml_text)
            raise ValueError(f"SOAP fault: {fault}")

        data = _parse_soap_response(xml_text, code)
        logger.info(f"SOAP parsed: title={data.get('title')!r} elements={len(data.get('elements',[]))}")

        if not data.get("title") or data["title"] == code:
            raise ValueError("SOAP returned no usable title — falling back to scraper")

        logger.info(f"SOAP import OK: {code} — {data['title']}")

    except Exception as e:
        soap_error = str(e)
        logger.warning(f"SOAP failed for {code}: {soap_error} — trying HTML scraper")

    # ── HTML scraper fallback ──────────────────────────────────────────────────
    if data is None:
        try:
            from bs4 import BeautifulSoup
            scrape_urls = [
                f"https://training.gov.au/training/details/{code}/unitdetails",
                f"https://training.gov.au/Training/Details/{code}",
            ]
            scrape_resp = None
            scrape_url_used = None
            for url in scrape_urls:
                try:
                    async with httpx.AsyncClient(timeout=30, follow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0"}) as client:
                        r = await client.get(url)
                        logger.info(f"Scraper {url} -> HTTP {r.status_code}, {len(r.text)} chars")
                        if r.is_success and len(r.text) > 1000:
                            scrape_resp = r
                            scrape_url_used = url
                            break
                except Exception as ue:
                    logger.warning(f"Scraper URL {url} failed: {ue}")

            if scrape_resp is None:
                raise ValueError("All scraper URLs failed. SOAP error was: " + str(soap_error))

            soup = BeautifulSoup(scrape_resp.text, "html.parser")
            data = _scrape_tgau_page(soup, code)
            logger.info(f"Scraper import OK: {code} — {data.get('title')} via {scrape_url_used}")

        except Exception as scrape_error:
            raise ValueError(
                f"Could not import {code}. "
                f"SOAP: {soap_error} | Scraper: {scrape_error}"
            )

    unit = _convert_tgau(code, data)
    cache.write_text(json.dumps(unit.model_dump(), indent=2))
    logger.info(f"Imported {code}: {unit.title} ({len(unit.elements)} elements, "
                f"{sum(len(e.pcs) for e in unit.elements)} PCs)")
    return unit


def _extract_soap_fault(xml_text: str) -> str:
    """Extract fault string from SOAP error response."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
        ns = {"s": "http://schemas.xmlsoap.org/soap/envelope/"}
        fault = root.find(".//faultstring")
        if fault is not None:
            return fault.text or "Unknown SOAP fault"
        # Try without namespace
        for el in root.iter():
            if el.tag.endswith("faultstring") or el.tag.endswith("Text"):
                return el.text or "Unknown error"
    except Exception:
        pass
    return xml_text[:200]


def _parse_soap_response(xml_text: str, code: str) -> dict:
    """
    Parse the SOAP GetDetails response from training.gov.au.
    Returns a normalised dict matching what _convert_tgau expects.
    """
    import xml.etree.ElementTree as ET
    import re

    root = ET.fromstring(xml_text)

    def find_text(parent, *tags):
        """Search for any tag by local name (ignores namespace)."""
        for tag in tags:
            for el in parent.iter():
                local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
                if local == tag and el.text:
                    return el.text.strip()
        return ""

    def find_all(parent, tag):
        """Find all elements by local name."""
        return [el for el in parent.iter()
                if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) == tag]

    # ── Title ──────────────────────────────────────────────────────────────────
    title = find_text(root, "Title", "UnitTitle", "ComponentTitle")

    # ── Application ───────────────────────────────────────────────────────────
    application = find_text(root, "ApplicationOfUnit", "Application", "UnitDescriptor")

    # ── Training package ───────────────────────────────────────────────────────
    pkg_title = find_text(root, "TrainingPackageTitle", "PackageTitle")
    pkg_code_match = re.match(r"^([A-Za-z]+)", code)
    pkg_code = pkg_code_match.group(1).upper() if pkg_code_match else code[:3].upper()

    # ── Elements and PCs ───────────────────────────────────────────────────────
    elements = []
    for el_node in find_all(root, "Element"):
        el_num_el = None
        el_title_el = None
        for child in el_node:
            local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local in ("ElementNumber", "Num", "Number"):
                el_num_el = child.text
            if local in ("ElementTitle", "Title", "Description"):
                el_title_el = child.text

        el_title = (el_title_el or "").strip()
        el_num = (el_num_el or str(len(elements) + 1)).strip()

        pcs = []
        for pc_node in find_all(el_node, "PerformanceCriteria"):
            pc_num = find_text(pc_node, "PCNumber", "Number", "Num")
            pc_text = find_text(pc_node, "PCText", "Text", "Description", "Criteria")
            if not pc_text:
                # Try getting text content directly
                pc_text = pc_node.text or ""
            pc_text = pc_text.strip()
            if pc_text:
                pcs.append({
                    "id": pc_num or f"{el_num}.{len(pcs)+1}",
                    "text": pc_text,
                    "element_id": f"E{len(elements)+1}"
                })

        if el_title:
            elements.append({"title": el_title, "pcs": pcs})

    # ── Knowledge evidence ─────────────────────────────────────────────────────
    knowledge = []
    for ke_node in find_all(root, "KnowledgeEvidence"):
        text = ke_node.text or find_text(ke_node, "Text", "Description")
        if text and text.strip():
            knowledge.append(text.strip())

    # Also try RequiredKnowledge / KnowledgeRequirement
    if not knowledge:
        for k_node in find_all(root, "RequiredKnowledge"):
            text = k_node.text or ""
            if text.strip():
                knowledge.append(text.strip())

    # ── Performance / skill evidence ───────────────────────────────────────────
    skills = []
    for pe_node in find_all(root, "PerformanceEvidence"):
        text = pe_node.text or find_text(pe_node, "Text", "Description")
        if text and text.strip():
            skills.append(text.strip())
    if not skills:
        for s_node in find_all(root, "RequiredSkills"):
            text = s_node.text or ""
            if text.strip():
                skills.append(text.strip())

    # Fallback if nothing parsed — log raw XML snippet for debugging
    if not elements:
        logger.warning(f"No elements parsed for {code}. XML snippet: {xml_text[1000:1500]}")
        elements = [{"title": f"Apply {title or code}", "pcs": [
            {"id": "1.1", "text": "Perform duties consistent with this unit in a workplace context", "element_id": "E1"},
        ]}]

    return {
        "title": title or code,
        "application": application,
        "elements": elements,
        "knowledgeEvidence": knowledge,
        "performanceEvidence": skills,
        "trainingPackage": {"title": pkg_title or pkg_code},
    }




def _scrape_tgau_page(soup, code: str) -> dict:
    """
    Fallback HTML scraper for training.gov.au unit detail page.
    Used when the SOAP API is unavailable or returns no data.
    URL format: https://training.gov.au/training/details/{code}/unitdetails
    """
    import re

    def txt(el):
        return el.get_text(separator=" ", strip=True) if el else ""

    def clean(s):
        return re.sub(r"\s+", " ", s).strip()

    # ── Title ──────────────────────────────────────────────────────────────────
    title = ""
    for sel in ["h1", ".unit-title", ".page-title", "h2"]:
        el = soup.select_one(sel)
        if el:
            raw = txt(el)
            # Strip unit code prefix e.g. "BSBMED301 Interpret and apply..."
            raw = re.sub(r"^[A-Z]{2,10}\d{3,6}\s*[-–]?\s*", "", raw).strip()
            if raw and len(raw) > 5:
                title = clean(raw)
                break

    if not title:
        page_title = soup.find("title")
        if page_title:
            raw = txt(page_title)
            raw = re.sub(r"^[A-Z]{2,10}\d{3,6}\s*[-–]?\s*", "", raw)
            raw = re.sub(r"\s*[-–|].*$", "", raw)
            title = clean(raw)

    # ── Application ────────────────────────────────────────────────────────────
    application = ""
    for heading in soup.find_all(["h2","h3","h4","h5"]):
        if re.search(r"application", txt(heading), re.I):
            parts = []
            for sib in heading.find_next_siblings():
                if sib.name in ["h2","h3","h4","h5"]: break
                t = txt(sib)
                if t: parts.append(t)
                if len(parts) >= 3: break
            application = " ".join(parts)
            break

    # ── Elements and PCs ───────────────────────────────────────────────────────
    elements = []

    # Strategy 1: numbered table rows
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        current_element = None
        current_pcs = []
        el_num = 0
        for row in rows:
            cells = row.find_all(["td","th"])
            if not cells: continue
            cell_text = [clean(txt(c)) for c in cells]
            m = re.match(r"^(\d+)\.?\s+(.+)", cell_text[0])
            if m and len(cell_text[0]) > 3:
                if current_element and current_pcs:
                    elements.append({"title": current_element, "pcs": current_pcs})
                el_num = int(m.group(1))
                current_element = clean(m.group(2))
                current_pcs = []
            elif current_element:
                for i, ct in enumerate(cell_text):
                    pm = re.match(r"^(\d+\.\d+)\s+(.+)", ct)
                    if pm:
                        current_pcs.append({
                            "id": pm.group(1), "text": clean(pm.group(2)),
                            "element_id": f"E{el_num}"})
                    elif len(ct) > 15 and i > 0:
                        current_pcs.append({
                            "id": f"{el_num}.{len(current_pcs)+1}", "text": clean(ct),
                            "element_id": f"E{el_num}"})
        if current_element and current_pcs:
            elements.append({"title": current_element, "pcs": current_pcs})

    # Strategy 2: numbered headings with list items
    if not elements:
        el_num = 0
        for el in soup.find_all(["h3","h4","h5","strong","b","p"]):
            t = clean(txt(el))
            m = re.match(r"^(\d+)\.?\s+(.{5,120})$", t)
            if m and int(m.group(1)) == el_num + 1:
                el_num += 1
                el_title = m.group(2)
                pcs = []
                lst = el.find_next(["ul","ol"])
                if lst:
                    for j, item in enumerate(lst.find_all("li"), 1):
                        pc_txt = clean(txt(item))
                        pm = re.match(r"^(\d+\.\d+)\s+(.+)", pc_txt)
                        if pm:
                            pcs.append({"id": pm.group(1), "text": pm.group(2), "element_id": f"E{el_num}"})
                        elif pc_txt:
                            pcs.append({"id": f"{el_num}.{j}", "text": pc_txt, "element_id": f"E{el_num}"})
                if el_title:
                    elements.append({"title": el_title, "pcs": pcs})

    if not elements:
        elements = [{"title": f"Apply {title or code}", "pcs": [
            {"id": "1.1", "text": "Perform tasks consistent with this unit in a workplace context", "element_id": "E1"},
            {"id": "1.2", "text": "Apply knowledge and skills as required by workplace procedures", "element_id": "E1"},
        ]}]

    # ── Knowledge evidence ─────────────────────────────────────────────────────
    knowledge = []
    for heading in soup.find_all(["h2","h3","h4","h5"]):
        if re.search(r"knowledge\s+evidence", txt(heading), re.I):
            lst = heading.find_next(["ul","ol"])
            if lst:
                knowledge = [clean(txt(li)) for li in lst.find_all("li") if txt(li).strip()]
            break

    # ── Skills / performance evidence ─────────────────────────────────────────
    skills = []
    for heading in soup.find_all(["h2","h3","h4","h5"]):
        if re.search(r"performance\s+evidence", txt(heading), re.I):
            lst = heading.find_next(["ul","ol"])
            if lst:
                skills = [clean(txt(li)) for li in lst.find_all("li") if txt(li).strip()]
            break

    pkg_match = re.match(r"^([A-Za-z]+)", code)
    pkg_code = pkg_match.group(1).upper() if pkg_match else code[:3].upper()

    return {
        "title": title or code,
        "application": application,
        "elements": elements,
        "knowledgeEvidence": knowledge,
        "performanceEvidence": skills,
        "trainingPackage": {"title": pkg_code},
    }


def _convert_tgau(code: str, data: dict) -> UnitOfCompetency:
    elements = []
    for i, el in enumerate(data.get("elements", []), 1):
        pcs = [PerformanceCriteria(id=f"{i}.{j}", text=pc, element_id=f"E{i}")
               for j, pc in enumerate(el.get("performanceCriteria", []), 1)]
        elements.append(Element(id=f"E{i}", title=el.get("title", f"Element {i}"), pcs=pcs))

    knowledge = [KnowledgeRequirement(id=f"K{i}", category=_infer_cat(k), text=k)
                 for i, k in enumerate(data.get("knowledgeEvidence", []), 1)]
    skills = [SkillRequirement(id=f"S{i}", category=_infer_cat(s), text=s)
              for i, s in enumerate(data.get("performanceEvidence", []), 1)]

    pkg = ''.join(filter(str.isalpha, code))[:3].upper()
    application = data.get("applicationOfUnit", "")
    return UnitOfCompetency(
        code=code, title=data.get("title", code),
        training_package=pkg,
        training_package_name=data.get("trainingPackage", {}).get("title", pkg),
        application=application,
        competent_person_statement=application[:600] if application else
            f"A competent person in {data.get('title', code)} demonstrates consistent "
            f"performance across all elements and performance criteria.",
        elements=elements, knowledge_requirements=knowledge, skill_requirements=skills,
        knowledge_questions=_auto_questions(elements),
        evidence_guide=_auto_guide(elements),
        source="tgau")


def _infer_cat(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ['whs', 'safety', 'hazard', 'ppe', 'risk']): return "WHS and safety"
    if any(w in t for w in ['sop', 'procedure', 'quality', 'instruction']): return "Quality systems"
    if any(w in t for w in ['environment', 'waste', 'disposal', 'spill']): return "Environmental"
    if any(w in t for w in ['communicat', 'report', 'document', 'record']): return "Communication"
    if any(w in t for w in ['train', 'develop', 'learn']): return "Training and development"
    if any(w in t for w in ['equipment', 'instrument', 'calibrat', 'maintain']): return "Equipment"
    if any(w in t for w in ['legislation', 'regulatory', 'legal', 'act', 'standard']): return "Legislation"
    return "General"


def _auto_questions(elements: list[Element]) -> list[KnowledgeQuestion]:
    return [KnowledgeQuestion(
        num=i + 1, element_ref=f"Element {i + 1}",
        pc_refs=[pc.id for pc in el.pcs[:3]],
        text=f"Describe your experience with {el.title.lower()}. Use specific examples "
             f"from your current or most recent workplace — what you do, how you do it, and the outcome.",
        hint="Think about: " + "; ".join(pc.text[:80] for pc in el.pcs[:2]))
        for i, el in enumerate(elements)]


def _auto_guide(elements: list[Element]) -> list[EvidenceGuideItem]:
    items = [EvidenceGuideItem(
        title=f"Evidence for {el.title}", priority="recommended",
        pc_refs=[pc.id for pc in el.pcs], icon="📁",
        description=f"Evidence demonstrating competency across {el.title}.",
        acceptable_forms=["Position description referencing these duties",
                          "Work samples or documented outputs",
                          "Supervisor or third party confirmation",
                          "Workplace records or logs"])
        for el in elements]
    items.append(EvidenceGuideItem(
        title="Third Party Report — Workplace Supervisor (Required)",
        priority="priority",
        pc_refs=[pc.id for el in elements for pc in el.pcs][:5],
        icon="📝",
        description="Completed by your direct supervisor — the strongest evidence item for RPL.",
        acceptable_forms=["Download the Third Party Report template",
                          "Supervisor completes and signs",
                          "Upload the completed signed form"]))
    return items
