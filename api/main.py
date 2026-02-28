"""
LUMINARK Ethical AI Guardian — FastAPI Backend
Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from luminark import LuminarkGuardian
from luminark.report import (
    generate_text_report, generate_markdown_report, batch_to_csv
)

# ─────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "LUMINARK Ethical AI Guardian API",
    description = (
        "Bio-inspired AI safety auditor. Analyzes text/AI outputs for "
        "misalignment, hallucinations, deception, harm, and arrogance. "
        "Returns risk scores, Ma'at violations, bio-defense status, and "
        "compassionate rewrites."
    ),
    version     = "1.0.0",
    contact     = {"name": "LUMINARK", "url": "https://github.com/luminark/guardian"},
    license_info= {"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

_guardian = LuminarkGuardian()

# ─────────────────────────────────────────────
#  Request / Response schemas
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, description="Text or AI output to audit")
    format: Optional[str] = Field("json", description="Response format: json | text | markdown")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Up to 100 texts to audit")

class ViolationOut(BaseModel):
    principle:     str
    description:   str
    category:      str
    severity:      float
    matched_terms: List[str]

class AnalyzeResponse(BaseModel):
    input_id:         str
    timestamp:        str
    badge:            str
    alignment_score:  float
    threat_score:     float
    confidence:       float
    stage:            str
    stage_label:      str
    stage_description:str
    defense_mode:     str
    violation_count:  int
    violations:       List[ViolationOut]
    category_breakdown: dict
    containment:      str
    camouflage:       str
    rewrite:          str
    rewrite_applied:  bool
    changes_made:     List[str]
    recommendations:  List[str]

class BatchResultItem(BaseModel):
    index:           int
    input_preview:   str
    badge:           str
    alignment_score: float
    threat_score:    float
    violation_count: int
    stage:           str
    rewrite_applied: bool

class BatchResponse(BaseModel):
    total:         int
    pass_count:    int
    caution_count: int
    fail_count:    int
    critical_count:int
    results:       List[BatchResultItem]

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "LUMINARK Ethical AI Guardian",
        "version": "1.0.0",
        "status":  "online",
        "docs":    "/docs",
    }


@app.get("/health", tags=["meta"])
def health():
    return {"status": "healthy"}


@app.post("/analyze", tags=["core"])
def analyze(req: AnalyzeRequest, request: Request):
    """
    Analyze a single text for ethical/safety violations.

    Returns risk scores, Ma'at violations, bio-defense status,
    SAP stage, and a compassionate rewrite.

    - **format=json** → full JSON (default)
    - **format=text** → plain-text audit report
    - **format=markdown** → markdown report
    """
    try:
        result = _guardian.analyze(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    fmt = (req.format or "json").lower()
    if fmt == "text":
        return PlainTextResponse(generate_text_report(result))
    if fmt == "markdown":
        return PlainTextResponse(generate_markdown_report(result), media_type="text/markdown")

    return result.to_dict()


@app.post("/batch", tags=["core"])
def batch_analyze(req: BatchRequest):
    """
    Analyze up to 100 texts in one call.
    Returns summary counts + per-item badge/score/violations.
    """
    results = [_guardian.analyze(t) for t in req.texts]

    badge_counts = {"PASS": 0, "CAUTION": 0, "FAIL": 0, "CRITICAL": 0}
    for r in results:
        badge_counts[r.badge] = badge_counts.get(r.badge, 0) + 1

    items = [
        BatchResultItem(
            index          = i,
            input_preview  = r.input_text[:120],
            badge          = r.badge,
            alignment_score= round(r.alignment_score, 1),
            threat_score   = round(r.threat_score, 3),
            violation_count= r.violation_count,
            stage          = r.stage.name,
            rewrite_applied= r.rewrite_applied,
        )
        for i, r in enumerate(results)
    ]

    return BatchResponse(
        total         = len(results),
        pass_count    = badge_counts.get("PASS", 0),
        caution_count = badge_counts.get("CAUTION", 0),
        fail_count    = badge_counts.get("FAIL", 0),
        critical_count= badge_counts.get("CRITICAL", 0),
        results       = items,
    )


@app.post("/batch/csv", tags=["core"], response_class=Response)
def batch_csv(req: BatchRequest):
    """Analyze up to 100 texts and return results as a CSV download."""
    results = [_guardian.analyze(t) for t in req.texts]
    csv_data = batch_to_csv(results)
    return Response(
        content     = csv_data,
        media_type  = "text/csv",
        headers     = {"Content-Disposition": "attachment; filename=luminark_audit.csv"},
    )


@app.get("/principles", tags=["reference"])
def list_principles():
    """List all active Ma'at principles used in auditing."""
    from luminark.principles import PRINCIPLE_PROFILES, CATEGORIES
    return {
        "count": len(PRINCIPLE_PROFILES),
        "categories": CATEGORIES,
        "principles": [
            {
                "id":         p.principle.value,
                "name":       p.principle.name,
                "label":      p.label,
                "ai_meaning": p.ai_meaning,
                "severity":   p.severity,
                "category":   p.category,
            }
            for p in PRINCIPLE_PROFILES
        ],
    }


@app.get("/stages", tags=["reference"])
def list_stages():
    """List all SAP consciousness stages used in threat assessment."""
    from luminark.guardian import SAPStage
    return {
        "stages": [
            {
                "value":       s.value,
                "name":        s.name,
                "label":       s.label,
                "description": s.description,
                "is_danger":   s.is_danger_zone,
                "multiplier":  s.risk_multiplier,
            }
            for s in SAPStage
        ]
    }


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)}
    )
