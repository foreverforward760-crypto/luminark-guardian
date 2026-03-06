"""
LUMINARK API — Pydantic v2 Schemas
All request and response models in one place.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ─────────────────────────────────────────────────────────────────────────────
#  Request schemas
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., min_length=1, max_length=50_000,
                      description="Text or AI output to audit")
    format: str = Field("json", pattern="^(json|text|markdown)$",
                        description="Response format: json | text | markdown")

    # Extended analysis fields — all optional with safe defaults
    nsdt:         dict = Field(default_factory=dict, description="Neural-Symbolic Decision Tree vectors")
    trapscore:    dict = Field(default_factory=dict, description="Trap/deception detection scores")
    cultural:     dict = Field(default_factory=dict, description="Cultural context adjustment params")
    hrv:          dict = Field(default_factory=dict, description="Heart Rate Variability metrics")
    stageHistory: list = Field(default_factory=list, description="SAP stage transition history")
    domain: str = Field(
        "general",
        pattern="^(general|medical|legal|marketing|education)$",
        description="Analysis domain",
    )
    pvManual: Optional[str] = Field(
        None,
        pattern="^(safe|mobilized|shutdown)$",
        description="Manual polyvagal state override",
    )
    citiStages:   dict = Field(default_factory=dict, description="CITI consciousness dimension scores")
    fractalDepth: int  = Field(0, ge=0, le=3, description="Recursive sub-fragment analysis depth")

    @property
    def has_extended(self) -> bool:
        return bool(
            self.nsdt or self.trapscore or self.cultural or self.hrv
            or self.citiStages or self.fractalDepth > 0 or self.pvManual
        )


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100,
                             description="Up to 100 texts to audit")


# ─────────────────────────────────────────────────────────────────────────────
#  Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ViolationOut(BaseModel):
    principle_id:  int
    principle:     str
    description:   str
    category:      str
    severity:      float
    matched_terms: List[str]


class ExtendedOut(BaseModel):
    nsdt_score:                float
    nsdt_flags:                List[str]
    citi_stage:                str
    citi_scores:               Dict[str, float]
    polyvagal_state:           str
    polyvagal_adjusted_threat: float
    cultural_adjustments:      Dict[str, float]
    cultural_notes:            List[str]
    fractal_depth_used:        int
    fractal_sub_results:       List[dict]


class AnalyzeResponse(BaseModel):
    # Identity
    input_id:   str
    timestamp:  str
    request_id: str

    # Scores
    badge:           str
    alignment_score: float
    threat_score:    float
    confidence:      float

    # Stage
    stage:             str
    stage_label:       str
    stage_description: str

    # Defense
    defense_mode: str
    containment:  str
    camouflage:   str

    # Violations
    violation_count:    int
    violations:         List[ViolationOut]
    category_breakdown: Dict[str, int]

    # Rewrites
    rewrite:         str
    rewrite_applied: bool
    changes_made:    List[str]

    # Recommendations
    recommendations: List[str]

    # Extended (None when no extended fields were provided)
    extended: Optional[ExtendedOut] = None


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
    total:          int
    pass_count:     int
    caution_count:  int
    fail_count:     int
    critical_count: int
    results:        List[BatchResultItem]


# ─────────────────────────────────────────────────────────────────────────────
#  Auth schemas
# ─────────────────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token:  str
    token_type:    str = "bearer"
    expires_in:    int       # seconds
    refresh_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class UserOut(BaseModel):
    id:           str
    username:     str
    email:        str
    scopes:       List[str]
    is_active:    bool
    is_superuser: bool
    created_at:   datetime


class CreateAPIKeyRequest(BaseModel):
    name:   str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = Field(default_factory=list,
                               description="Valid: analyze, batch, admin")


class APIKeyOut(BaseModel):
    id:         str
    name:       str
    key_prefix: str      # First 8 chars — safe to display
    scopes:     List[str]
    is_active:  bool
    created_at: datetime


class APIKeyCreatedOut(APIKeyOut):
    key_value: str       # Full key — shown ONCE at creation only


# ─────────────────────────────────────────────────────────────────────────────
#  Internal auth helpers
# ─────────────────────────────────────────────────────────────────────────────

class CurrentUser(BaseModel):
    id:       str
    username: str
    scopes:   List[str] = []
    via:      str = "jwt"   # "jwt" | "apikey"
    jti:      Optional[str] = None
