import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

import core

load_dotenv()
st.set_page_config(page_title="ResumeIQ - AI Resume Analyzer", page_icon="📄", layout="wide")

PREFERRED_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


@st.cache_data(ttl=3600, show_spinner=False)
def available_models(api_key):
    """Ask Groq which models this account can use, so retired/restricted models never break the app."""
    try:
        ids = [m.id for m in Groq(api_key=api_key).models.list().data]
    except Exception:
        return PREFERRED_MODELS
    skip = ("whisper", "guard", "tts", "orpheus", "compound", "safeguard", "allam")
    chat = [i for i in ids if not any(s in i for s in skip)]
    ordered = [m for m in PREFERRED_MODELS if m in chat] + [m for m in chat if m not in PREFERRED_MODELS]
    return ordered or PREFERRED_MODELS


# ------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedder():
    return core.load_embedder()


@st.cache_resource(show_spinner=False)
def get_resume_index(_embedder, text):
    return core.build_resume_index(_embedder, text)


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GROQ_API_KEY")


def gauge(value, title):
    color = "#16a34a" if value >= 70 else "#f59e0b" if value >= 45 else "#dc2626"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def chips(items, empty="None"):
    return " ".join(f"`{i}`" for i in items) if items else empty


def render(res):
    l = res["llm"]
    st.divider()
    st.subheader(f"Result: {res['job_title']}")
    st.markdown(f"**Verdict:** {core.verdict(res['overall'])}")

    c1, c2, c3, c4 = st.columns(4)
    c1.plotly_chart(gauge(res["overall"], "Overall match"))
    c2.plotly_chart(gauge(res["semantic"], "Semantic match"))
    c3.plotly_chart(gauge(res["skill_pct"], "Skill match"))
    c4.plotly_chart(gauge(res["ats"], "ATS readiness"))

    a, b = st.columns(2)
    a.markdown("#### ✅ Matched skills")
    a.markdown(chips(l["matched_skills"]))
    b.markdown("#### ❌ Missing skills")
    b.markdown(chips(l["missing_skills"]))

    a, b = st.columns(2)
    a.markdown("#### 💪 Strengths")
    a.markdown("\n".join(f"- {s}" for s in l["strengths"]) or "-")
    b.markdown("#### ⚠️ Weaknesses")
    b.markdown("\n".join(f"- {s}" for s in l["weaknesses"]) or "-")

    st.markdown("#### 🛠️ Suggestions")
    st.markdown("\n".join(f"{n}. {s}" for n, s in enumerate(l["suggestions"], 1)) or "-")

    if l["improved_summary"]:
        st.markdown("#### ✍️ Tailored summary (copy into your resume)")
        st.info(l["improved_summary"])

    if l["bullet_rewrites"]:
        with st.expander("Bullet rewrites"):
            for r in l["bullet_rewrites"]:
                st.markdown(f"**Before:** {r['original']}")
                st.markdown(f"**After:** {r['improved']}")
                st.markdown("---")

    with st.expander("Least-covered job requirements (semantic gaps)"):
        if res["gaps"]:
            st.dataframe(pd.DataFrame(res["gaps"][:8]), hide_index=True)

    with st.expander("ATS checklist"):
        for name, ok in res["ats_checks"].items():
            st.markdown(f"{'✅' if ok else '❌'} {name}")

    st.download_button(
        "⬇️ Download report (.md)",
        core.build_report(res),
        file_name="resumeiq_report.md",
        mime="text/markdown",
    )


# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📄 ResumeIQ")
    api_key = get_api_key() or st.text_input("Groq API key", type="password")
    model_name = st.selectbox("LLM model", available_models(api_key) if api_key else PREFERRED_MODELS)
    st.divider()
    uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    pasted = st.text_area("...or paste resume text", height=120)

resume_text = ""
if uploaded:
    try:
        resume_text = core.extract_pdf_text(uploaded)
    except Exception as e:
        st.sidebar.error(f"Could not read PDF: {e}")
elif pasted.strip():
    resume_text = pasted.strip()

# ------------------------------------------------------------ main
st.title("AI Resume Analyzer & Job Matcher")
st.caption("Semantic matching (FastEmbed + FAISS) and structured feedback (Groq LLM)")

if not resume_text:
    st.info("👈 Upload your resume (PDF) or paste its text in the sidebar to begin.")
    st.stop()
if len(resume_text.split()) < 50:
    st.warning("Very little text was extracted. If your PDF is a scanned image, paste the text instead.")
    st.stop()

embedder = get_embedder()
units, index = get_resume_index(embedder, resume_text)
ats_score, ats_checks = core.ats_check(resume_text)

tab1, tab2, tab3 = st.tabs(["🎯 Deep Match", "🏆 Rank Jobs", "ℹ️ How it works"])

with tab1:
    jd = st.text_area("Paste the job description", height=220, key="jd_single")
    if st.button("Analyze match", type="primary"):
        if len(jd.split()) < 30:
            st.warning("Please paste a fuller job description (30+ words).")
        elif not api_key:
            st.error("Add your Groq API key in the sidebar or in .env / Streamlit secrets.")
        else:
            try:
                with st.spinner("Analyzing resume against job description..."):
                    llm = core.analyze_with_llm(Groq(api_key=api_key), model_name, resume_text, jd)
                    sem, gaps = core.semantic_match(embedder, units, index, jd)
                st.session_state["result"] = core.combine(llm, sem, gaps, ats_score, ats_checks)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    if st.session_state.get("result"):
        render(st.session_state["result"])

with tab2:
    st.caption("Paste several job descriptions, separated by a line containing only `---`. "
               "Ranking uses embeddings only, so it is instant and uses no API calls.")
    multi = st.text_area("Job descriptions", height=300, key="jd_multi")
    if st.button("Rank jobs"):
        jobs = [j.strip() for j in re.split(r"^\s*---\s*$", multi, flags=re.M) if j.strip()]
        if len(jobs) < 2:
            st.warning("Add at least two job descriptions separated by ---")
        else:
            rows = []
            for n, j in enumerate(jobs, 1):
                score, _ = core.semantic_match(embedder, units, index, j)
                title = j.splitlines()[0][:60]
                rows.append({"Job": f"{n}. {title}", "Match %": score})
            df = pd.DataFrame(rows).sort_values("Match %", ascending=False)
            fig = go.Figure(go.Bar(x=df["Match %"], y=df["Job"], orientation="h"))
            fig.update_layout(yaxis=dict(autorange="reversed"), height=100 + 60 * len(df),
                              xaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
            st.dataframe(df, hide_index=True)

with tab3:
    st.markdown(
        """
1. **Parse** - text is extracted from your PDF with `pypdf`.
2. **Embed** - resume lines and job requirements become vectors (MiniLM via FastEmbed).
3. **Match** - a FAISS index finds the closest resume line for each job requirement (cosine similarity).
4. **Analyze** - A Groq-hosted LLM (GPT-OSS 120B by default) returns structured JSON: skills, gaps, strengths, rewrites.
5. **Score** - `Overall = 40% semantic + 40% skill match + 20% ATS readiness`.

Your resume is processed in memory and never stored.
"""
    )