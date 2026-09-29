"""ResumeIQ core logic: PDF parsing, semantic matching (FAISS), ATS checks, Groq LLM analysis."""
import json
import re

import faiss
import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------- parsing
def load_embedder():
    return TextEmbedding(EMBED_MODEL)


def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_units(text: str, min_len: int = 25, max_len: int = 350):
    """Split text into line/sentence-sized units (resume bullets, JD requirements)."""
    units = []
    for part in re.split(r"\n+|(?<=[.;])\s+", text):
        part = part.strip(" \t•●▪◦-–—*")
        if len(part) >= min_len:
            units.append(part[:max_len])
    return units


# ---------------------------------------------------------------- semantic matching
def embed(model, texts):
    vecs = np.array(list(model.embed(texts)), dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


def build_resume_index(model, resume_text: str):
    units = split_units(resume_text)
    if not units:
        return units, None
    vecs = embed(model, units)
    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product on normalized vectors = cosine
    index.add(vecs)
    return units, index


def semantic_match(model, units, index, jd_text: str):
    """For every job-requirement line, find the closest resume line. Returns (score 0-100, rows)."""
    jd_units = split_units(jd_text)
    if index is None or not jd_units:
        return 0.0, []
    sims, ids = index.search(embed(model, jd_units), 1)
    best = sims[:, 0]
    # MiniLM cosine for related text is ~0.15 (unrelated) to ~0.65 (strong) -> rescale to 0-100
    score = float(np.clip((best.mean() - 0.15) / 0.50, 0, 1) * 100)
    rows = [
        {
            "Job requirement": jd_units[i],
            "Closest resume line": units[ids[i, 0]],
            "Similarity": round(float(best[i]), 2),
        }
        for i in range(len(jd_units))
    ]
    rows.sort(key=lambda r: r["Similarity"])  # weakest coverage first
    return round(score, 1), rows


# ---------------------------------------------------------------- ATS checks
SECTIONS = {
    "Education section": ["education"],
    "Experience / Internship section": ["experience", "internship", "employment"],
    "Skills section": ["skills"],
    "Projects section": ["projects"],
}


def ats_check(text: str):
    low = text.lower()
    words = len(text.split())
    checks = {
        "Email address": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)),
        "Phone number": bool(re.search(r"\+?\d[\d\s\-()]{8,}\d", text)),
        "LinkedIn / GitHub link": "linkedin" in low or "github" in low,
    }
    for name, keys in SECTIONS.items():
        checks[name] = any(k in low for k in keys)
    checks["Length 250-900 words"] = 250 <= words <= 900
    checks["Quantified impact (3+ numbers / %)"] = (
        len(re.findall(r"\d+\s?%|\d+\+|\$\s?\d+|\b\d{2,}\b", text)) >= 3
    )
    score = round(100 * sum(checks.values()) / len(checks))
    return score, checks


# ---------------------------------------------------------------- LLM analysis
SYSTEM_PROMPT = (
    "You are an expert technical recruiter and ATS specialist. "
    "You respond with valid JSON only, no markdown, no extra text."
)

USER_PROMPT = """Compare the RESUME against the JOB DESCRIPTION and return a JSON object with EXACTLY these keys:
{{
  "job_title": "short job title inferred from the job description",
  "required_skills": ["key skills/tools/qualifications the job requires, max 15"],
  "matched_skills": ["required skills clearly evidenced in the resume"],
  "missing_skills": ["required skills NOT evidenced in the resume"],
  "strengths": ["3-5 specific strengths of this resume for this job"],
  "weaknesses": ["3-5 specific weaknesses or gaps for this job"],
  "suggestions": ["4-6 specific, actionable improvements to make the resume fit this job"],
  "improved_summary": "a 3-sentence professional summary tailored to this job, using ONLY facts present in the resume",
  "bullet_rewrites": [{{"original": "a weak bullet from the resume", "improved": "stronger version with action verb and impact, without inventing facts"}}]
}}
Rules: never invent experience, employers, or numbers. bullet_rewrites has at most 3 items.
matched_skills and missing_skills must be subsets of required_skills.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
"""


def _list(x):
    return [str(i).strip() for i in x if str(i).strip()] if isinstance(x, list) else []


def analyze_with_llm(client, model_name: str, resume: str, jd: str) -> dict:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(resume=resume[:12000], jd=jd[:6000])},
        ],
        temperature=0.2,
        max_completion_tokens=4096,
        response_format={"type": "json_object"},
        **({"reasoning_effort": "low"} if model_name.startswith("openai/gpt-oss") else {}),
    )
    data = json.loads(resp.choices[0].message.content)
    rewrites = [
        {"original": str(r.get("original", "")), "improved": str(r.get("improved", ""))}
        for r in data.get("bullet_rewrites", [])
        if isinstance(r, dict)
    ][:3]
    return {
        "job_title": str(data.get("job_title") or "Target role"),
        "required_skills": _list(data.get("required_skills")),
        "matched_skills": _list(data.get("matched_skills")),
        "missing_skills": _list(data.get("missing_skills")),
        "strengths": _list(data.get("strengths")),
        "weaknesses": _list(data.get("weaknesses")),
        "suggestions": _list(data.get("suggestions")),
        "improved_summary": str(data.get("improved_summary") or ""),
        "bullet_rewrites": rewrites,
    }


# ---------------------------------------------------------------- scoring + report
def combine(llm: dict, semantic: float, gaps: list, ats_score: int, ats_checks: dict) -> dict:
    required = llm["required_skills"]
    if required:
        skill_pct = 100 * min(len(llm["matched_skills"]), len(required)) / len(required)
    else:
        skill_pct = semantic
    overall = round(0.4 * semantic + 0.4 * skill_pct + 0.2 * ats_score)
    return {
        "job_title": llm["job_title"],
        "overall": overall,
        "semantic": round(semantic),
        "skill_pct": round(skill_pct),
        "ats": ats_score,
        "ats_checks": ats_checks,
        "gaps": gaps,
        "llm": llm,
    }


def verdict(score: int) -> str:
    if score >= 75:
        return "Strong match - apply with confidence"
    if score >= 55:
        return "Good match - tailor your resume before applying"
    if score >= 35:
        return "Partial match - close the skill gaps first"
    return "Weak match - consider a different role or major changes"


def build_report(res: dict) -> str:
    l = res["llm"]
    bullets = lambda items: "\n".join(f"- {i}" for i in items) or "- None"
    rewrites = "\n".join(f"- **Before:** {r['original']}\n  **After:** {r['improved']}" for r in l["bullet_rewrites"]) or "- None"
    ats = "\n".join(f"- [{'x' if ok else ' '}] {name}" for name, ok in res["ats_checks"].items())
    return f"""# ResumeIQ Report - {res['job_title']}

**Overall match: {res['overall']}/100** - {verdict(res['overall'])}

| Semantic match | Skill match | ATS readiness |
|---|---|---|
| {res['semantic']}% | {res['skill_pct']}% | {res['ats']}% |

## Matched skills
{bullets(l['matched_skills'])}

## Missing skills
{bullets(l['missing_skills'])}

## Strengths
{bullets(l['strengths'])}

## Weaknesses
{bullets(l['weaknesses'])}

## Suggestions
{bullets(l['suggestions'])}

## Tailored summary
{l['improved_summary']}

## Bullet rewrites
{rewrites}

## ATS checklist
{ats}
"""