# ResumeIQ – AI Resume Analyzer & Job Matcher

Upload a PDF resume, paste a job description, get a match score, skill gaps and tailored rewrite suggestions.

**Stack:** Python, Streamlit, Groq (Llama 3.3 70B), FastEmbed (MiniLM), FAISS, pypdf, Plotly

## Run locally
```
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env  -> then add your Groq key
streamlit run app.py
```
