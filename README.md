# 🌿 LUMINARK Ethical AI Guardian

> **Bio-inspired · Ma'at-audited · Compassionate AI safety**

A production-ready AI safety auditor that scans text and AI outputs for ethical violations, hallucination risk, deception, harm, and epistemic overreach — then applies compassionate rewrites.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io)

---

## What It Does

LUMINARK analyzes any text for:

| Category | Examples |
|---|---|
| **Epistemic Overreach** | "absolutely certain", "guaranteed", "proven" |
| **Arrogance / Hubris** | "perfect", "infallible", "obviously superior" |
| **Deception / Manipulation** | Unattributed claims, fear-based urgency, fraud language |
| **Harm / Hostility** | Violence, coercion, contempt, emotional manipulation |

**Outputs:**
- Safety badge: `PASS` / `CAUTION` / `FAIL` / `CRITICAL`
- Alignment score (0–100) and RISS threat score (0–1)
- SAP consciousness stage (0–9, hallucination risk mapping)
- Bio-defense status (mycelial containment / octo-camouflage)
- Ma'at violation list with severity ratings
- Compassionate rewrite with tracked changes
- Downloadable reports: TXT, Markdown, JSON, CSV

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/luminark/guardian
cd luminark-guardian
pip install -r requirements.txt
```

### 2. CLI

```bash
# Analyze a text string
python -m luminark "I am absolutely certain this AI is perfect."

# Analyze a file
python -m luminark --file my_output.txt

# Markdown output
python -m luminark --format markdown "This model definitely outperforms all others."

# Batch audit (one text per line)
python -m luminark --batch texts.txt --out results.csv

# Save report to file
python -m luminark "Some text" --out report.txt
```

### 3. Python API

```python
from luminark import LuminarkGuardian

guardian = LuminarkGuardian()
result = guardian.analyze("I am absolutely certain this AI is perfect.")

print(result.badge)          # FAIL
print(result.alignment_score) # 0.0
print(result.threat_score)    # 0.962
print(result.stage.label)     # Rigidity Trap
for v in result.violations:
    print(v.label, v.severity)
print(result.rewrite)         # I am arguably possibly this AI is well-considered.

# Export
print(result.to_json())
```

### 4. REST API

```bash
# Start API server
uvicorn api.main:app --reload --port 8000

# Analyze
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is definitely the best model ever."}'

# Get markdown report
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Guaranteed results!", "format": "markdown"}'

# Batch + CSV
curl -X POST http://localhost:8000/batch/csv \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text one.", "Text two."]}' \
  --output results.csv

# List Ma'at principles
curl http://localhost:8000/principles

# Interactive API docs
open http://localhost:8000/docs
```

### 5. Dashboard (Streamlit)

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## Deploy Free

### Option A: Hugging Face Spaces (Recommended for demos)

1. Create account at [huggingface.co](https://huggingface.co)
2. New Space → SDK: Streamlit → Upload all files
3. Set `dashboard/app.py` as the entry point (or rename to `app.py` at root)
4. Live public demo URL instantly

### Option B: Streamlit Community Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → select `dashboard/app.py`
4. Deploy — free, permanent URL

### Option C: Railway (API)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

---

## Project Structure

```
luminark_project/
├── luminark/
│   ├── __init__.py        # Package exports
│   ├── __main__.py        # CLI entry point
│   ├── principles.py      # Ma'at 42 principles + trigger keywords
│   ├── guardian.py        # Core analysis engine
│   └── report.py          # TXT / Markdown / CSV report generators
├── api/
│   └── main.py            # FastAPI REST API
├── dashboard/
│   └── app.py             # Streamlit UI dashboard
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Monetization Guide

### Tier 1: Freelance Audits ($50–200/audit)
- Run LUMINARK on client AI chatbot outputs or marketing copy
- Deliver the text report + Markdown report as deliverable
- Target: AI startups, marketing agencies, legal/compliance teams
- Find clients: Upwork ("AI safety"), LinkedIn ("AI ethics audit"), Reddit r/AIEthics

### Tier 2: SaaS API ($10–50/month)
- Host API on Railway/Render
- Add API key auth (simple: check `Authorization: Bearer <key>` header)
- Offer 100 free analyses/month, then paid tiers
- Payment: Stripe + simple landing page on Carrd.co ($19/year)

### Tier 3: Grant Funding (No-cost)
- **LTFF (Long-Term Future Fund)**: Rolling applications, $5k–$50k for AI safety tools
- **Open Philanthropy**: AI safety small grants
- **Survival and Flourishing Fund**: $10k–$100k for safety-adjacent work
- **NSF SBIR**: Up to $275k for AI safety small businesses
- Pitch: "Bio-ethical containment framework for LLM output validation"

### Tier 4: B2B Integrations
- Offer as a npm/pip plugin for CI/CD pipelines ("AI output linter")
- GitHub Action that scans AI-generated PRs/docs
- Slack bot integration ($5/seat/month)

### Quick Win (Week 1)
1. Deploy dashboard to HF Spaces (free, 30 min)
2. Post demo link on r/MachineLearning, r/AIEthics, LinkedIn
3. Offer "free audit" of anyone's AI output in exchange for testimonial
4. Convert 3+ testimonials into paid clients at $75/audit

---

## Ethical Framework

| System | Source | Role |
|---|---|---|
| **Ma'at 42 Principles** | Ancient Egyptian ethics | Violation detection taxonomy |
| **Yunus Protocol** | Muhammad Yunus (humility philosophy) | Arrogance/Stage 8 gating |
| **SAP/NAM Staging** | Stanfield's Axiom of Perpetuity | Hallucination risk mapping |
| **RISS Scoring** | Bio-HRV analog | Dynamic composite threat score |
| **Mycelial Containment** | Fungal network biology | Threat isolation metaphor |
| **Octo-Camouflage** | Octopus void mimicry | Healthy substrate protection |

---

## License

MIT — free to use, modify, sell, and deploy.

---

*Built with care. Substrate-independent. Compassion-first.*
