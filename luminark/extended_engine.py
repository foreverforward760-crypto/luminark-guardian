"""
LUMINARK Extended Engine — v2.0
Supplements the core LuminarkGuardian with five additional analysis modules:
  • NSDTAnalyzer      — Neural-Symbolic Decision Tree alignment scoring
  • CITIAnalyzer      — Consciousness Integration stage mapping
  • PolyvagalAdjuster — HRV-based threat score modulation
  • CulturalContextAdjuster — Domain & culture-aware severity weighting
  • FractalAnalyzer   — Recursive sub-fragment analysis
  • ExtendedEngine    — Composes all modules with the base guardian
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .guardian import LuminarkGuardian, GuardianResult, Violation


# ─────────────────────────────────────────────────────────────────────────────
#  Extended Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtendedResult:
    nsdt_score: float                      # 0–1, higher = more aligned
    nsdt_flags: List[str]                  # Symbolic anomaly labels
    citi_stage: str                        # Consciousness integration label
    citi_scores: Dict[str, float]          # Per-dimension CITI scores
    polyvagal_state: str                   # "safe" | "mobilized" | "shutdown"
    polyvagal_adjusted_threat: float       # RISS after HRV adjustment
    cultural_adjustments: Dict[str, float] # Category → delta severity
    cultural_notes: List[str]              # Human-readable adjustment notes
    fractal_depth_used: int                # Actual recursion depth applied
    fractal_sub_results: List[dict]        # Per-fragment analysis summaries

    def to_dict(self) -> dict:
        return {
            "nsdt_score":                self.nsdt_score,
            "nsdt_flags":                self.nsdt_flags,
            "citi_stage":                self.citi_stage,
            "citi_scores":               self.citi_scores,
            "polyvagal_state":           self.polyvagal_state,
            "polyvagal_adjusted_threat": round(self.polyvagal_adjusted_threat, 3),
            "cultural_adjustments":      self.cultural_adjustments,
            "cultural_notes":            self.cultural_notes,
            "fractal_depth_used":        self.fractal_depth_used,
            "fractal_sub_results":       self.fractal_sub_results,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  NSDTAnalyzer — Neural-Symbolic Decision Tree
# ─────────────────────────────────────────────────────────────────────────────

class NSDTAnalyzer:
    """
    Interprets Neural-Symbolic Decision Tree vectors supplied by upstream
    model introspection hooks.

    Expected nsdt dict schema:
        certainty_vector  float  0–1  Model's own certainty embedding
        coherence_score   float  0–1  Logical coherence
        source_grounding  float  0–1  Factual groundedness
        contradiction_flag bool  Self-contradiction detected upstream
    """

    WEIGHTS = {
        "certainty_vector": 0.30,
        "coherence_score":  0.35,
        "source_grounding": 0.35,
    }

    def analyze(self, nsdt: dict) -> Tuple[float, List[str]]:
        cert  = float(nsdt.get("certainty_vector", 0.5))
        coh   = float(nsdt.get("coherence_score",  0.5))
        grnd  = float(nsdt.get("source_grounding", 0.5))
        contr = bool(nsdt.get("contradiction_flag", False))

        score = (
            cert * self.WEIGHTS["certainty_vector"] +
            coh  * self.WEIGHTS["coherence_score"]  +
            grnd * self.WEIGHTS["source_grounding"]
        )

        if contr:
            score *= 0.75

        score = float(np.clip(score, 0.0, 1.0))

        flags: List[str] = []
        if cert > 0.75:
            flags.append("HIGH_CERTAINTY_EMBEDDING")
        if coh < 0.40:
            flags.append("LOW_COHERENCE")
        if grnd < 0.35:
            flags.append("WEAK_GROUNDING")
        if contr:
            flags.append("SELF_CONTRADICTION")

        return score, flags


# ─────────────────────────────────────────────────────────────────────────────
#  CITIAnalyzer — Consciousness Integration
# ─────────────────────────────────────────────────────────────────────────────

class CITIAnalyzer:
    """
    Maps five CITI dimension scores to a qualitative stage label.

    Expected citiStages dict schema:
        integration        float  0–1
        differentiation    float  0–1
        binding            float  0–1
        metacognition      float  0–1
        temporal_coherence float  0–1
    """

    DIMENSIONS = [
        "integration",
        "differentiation",
        "binding",
        "metacognition",
        "temporal_coherence",
    ]

    STAGE_MAP = [
        (0.85, "Integrated"),
        (0.65, "Differentiating"),
        (0.45, "Binding"),
        (0.25, "Fragmented"),
        (0.00, "Dissociated"),
    ]

    def analyze(self, citi_stages: dict) -> Tuple[str, Dict[str, float]]:
        filled: Dict[str, float] = {
            dim: float(citi_stages.get(dim, 0.5))
            for dim in self.DIMENSIONS
        }

        aggregate = float(np.mean(list(filled.values())))

        label = "Dissociated"
        for threshold, name in self.STAGE_MAP:
            if aggregate >= threshold:
                label = name
                break

        return label, filled


# ─────────────────────────────────────────────────────────────────────────────
#  PolyvagalAdjuster — HRV-based threat modulation
# ─────────────────────────────────────────────────────────────────────────────

class PolyvagalAdjuster:
    """
    Adjusts the base RISS threat score based on polyvagal state.

    Expected hrv dict schema:
        sdnn        float  HRV SDNN in ms (higher = more regulated)
        rmssd       float  RMSSD in ms
        lf_hf_ratio float  LF/HF ratio (higher = more sympathetic)
        pnn50       float  %NN50 (higher = more parasympathetic)

    pvManual: optional override "safe" | "mobilized" | "shutdown"

    Adjustment multipliers:
        safe      × 0.85  (parasympathetic — more regulated output)
        mobilized × 1.10  (sympathetic activation — amplified risk)
        shutdown  × 1.25  (dorsal vagal — dysregulated, high risk)
    """

    MULTIPLIERS = {
        "safe":      0.85,
        "mobilized": 1.10,
        "shutdown":  1.25,
    }

    def _derive_state(self, hrv: dict) -> str:
        sdnn       = float(hrv.get("sdnn", 40.0))
        lf_hf     = float(hrv.get("lf_hf_ratio", 2.5))

        if sdnn > 50 and lf_hf < 2.0:
            return "safe"
        if sdnn < 20 or lf_hf > 4.0:
            return "shutdown"
        return "mobilized"

    def analyze(
        self,
        hrv: dict,
        pv_manual: Optional[str],
        base_threat: float,
    ) -> Tuple[str, float]:
        if pv_manual and pv_manual in self.MULTIPLIERS:
            state = pv_manual
        elif hrv:
            state = self._derive_state(hrv)
        else:
            state = "mobilized"

        adjusted = float(np.clip(base_threat * self.MULTIPLIERS[state], 0.0, 1.0))
        return state, adjusted


# ─────────────────────────────────────────────────────────────────────────────
#  CulturalContextAdjuster
# ─────────────────────────────────────────────────────────────────────────────

class CulturalContextAdjuster:
    """
    Adjusts per-category severity based on domain and cultural dimensions.

    Expected cultural dict schema:
        region        str    ISO region code e.g. "MENA", "East_Asia", "Western"
        formality     float  0–1 (1 = highly formal register)
        collectivism  float  0–1 (1 = highly collectivist)
        power_distance float  0–1 (1 = high power distance)

    Domain values: "general" | "medical" | "legal" | "marketing" | "education"
    """

    DOMAIN_MODIFIERS: Dict[str, Dict[str, float]] = {
        "medical": {
            "certainty": +0.15,
            "harm":      +0.20,
        },
        "legal": {
            "deception": +0.20,
            "arrogance": +0.10,
        },
        "marketing": {
            "deception": +0.25,
            "certainty": +0.10,
        },
        "education": {
            "arrogance": +0.10,
            "harm":      +0.10,
        },
        "general": {},
    }

    def analyze(
        self,
        cultural: dict,
        domain: str,
        violations: List[Violation],
    ) -> Tuple[Dict[str, float], List[str]]:
        adjustments: Dict[str, float] = {}
        notes: List[str] = []

        domain_mods = self.DOMAIN_MODIFIERS.get(domain, {})
        for cat, delta in domain_mods.items():
            adjustments[cat] = adjustments.get(cat, 0.0) + delta
            notes.append(f"Domain '{domain}': {cat} severity +{delta:+.2f}")

        collectivism  = float(cultural.get("collectivism",   0.5))
        power_dist    = float(cultural.get("power_distance", 0.5))
        formality     = float(cultural.get("formality",      0.5))

        arrog_delta = -(collectivism - 0.5) * 0.20
        if abs(arrog_delta) > 0.01:
            adjustments["arrogance"] = adjustments.get("arrogance", 0.0) + arrog_delta
            notes.append(
                f"Collectivism={collectivism:.2f}: arrogance severity {arrog_delta:+.2f}"
            )

        harm_delta = (power_dist - 0.5) * 0.20
        if abs(harm_delta) > 0.01:
            adjustments["harm"] = adjustments.get("harm", 0.0) + harm_delta
            notes.append(
                f"Power distance={power_dist:.2f}: harm severity {harm_delta:+.2f}"
            )

        if formality > 0.7:
            adjustments["deception"] = adjustments.get("deception", 0.0) + 0.05
            notes.append(f"High formality={formality:.2f}: deception sensitivity +0.05")

        return adjustments, notes


# ─────────────────────────────────────────────────────────────────────────────
#  FractalAnalyzer — Recursive sub-fragment analysis
# ─────────────────────────────────────────────────────────────────────────────

class FractalAnalyzer:
    """
    Recursively applies LuminarkGuardian to sub-fragments of the input text.

    Depth semantics:
        0 — no recursion (base analysis only)
        1 — split into paragraphs, analyze each
        2 — also split each paragraph into sentences
        3 — sentence-level analysis of every paragraph (max depth)

    Fragments shorter than MIN_FRAGMENT_WORDS are skipped.
    """

    MAX_DEPTH       = 3
    MIN_FRAGMENT_WORDS = 10

    def __init__(self, guardian: LuminarkGuardian) -> None:
        self._guardian = guardian

    def _fragment_summary(self, idx: int, text: str, result: GuardianResult) -> dict:
        return {
            "fragment_index": idx,
            "preview":        text[:80].replace("\n", " "),
            "word_count":     len(text.split()),
            "badge":          result.badge,
            "alignment_score":round(result.alignment_score, 1),
            "threat_score":   round(result.threat_score, 3),
            "violation_count":result.violation_count,
            "stage":          result.stage.name,
        }

    def analyze(self, text: str, depth: int) -> Tuple[int, List[dict]]:
        depth = min(max(depth, 0), self.MAX_DEPTH)
        if depth == 0:
            return 0, []

        sub_results: List[dict] = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for p_idx, para in enumerate(paragraphs):
            if len(para.split()) < self.MIN_FRAGMENT_WORDS:
                continue
            para_result = self._guardian.analyze(para)
            sub_results.append(self._fragment_summary(p_idx, para, para_result))

            if depth >= 2:
                sentences = [
                    s.strip() for s in para.replace("! ", ". ").replace("? ", ". ").split(". ")
                    if s.strip() and len(s.split()) >= self.MIN_FRAGMENT_WORDS
                ]
                for s_idx, sent in enumerate(sentences):
                    sent_result = self._guardian.analyze(sent)
                    summary = self._fragment_summary(
                        p_idx * 1000 + s_idx, sent, sent_result
                    )
                    summary["parent_paragraph"] = p_idx
                    sub_results.append(summary)

        return depth, sub_results


# ─────────────────────────────────────────────────────────────────────────────
#  ExtendedEngine — Composition root
# ─────────────────────────────────────────────────────────────────────────────

class ExtendedEngine:
    """
    Wraps LuminarkGuardian and orchestrates all extension analyzers.
    Thread-safe; all state is read-only after __init__.
    """

    def __init__(self) -> None:
        self._guardian  = LuminarkGuardian()
        self._nsdt      = NSDTAnalyzer()
        self._citi      = CITIAnalyzer()
        self._pv        = PolyvagalAdjuster()
        self._cultural  = CulturalContextAdjuster()
        self._fractal   = FractalAnalyzer(self._guardian)

    def analyze(self, text: str) -> GuardianResult:
        """Shortcut: run base engine only."""
        return self._guardian.analyze(text)

    def analyze_extended(
        self,
        text: str,
        nsdt: dict,
        trapscore: dict,
        cultural: dict,
        hrv: dict,
        stage_history: list,
        domain: str,
        pv_manual: Optional[str],
        citi_stages: dict,
        fractal_depth: int,
    ) -> Tuple[GuardianResult, ExtendedResult]:
        """
        Full pipeline. Returns (GuardianResult, ExtendedResult).
        GuardianResult is unchanged; ExtendedResult is purely additive.
        """
        base = self._guardian.analyze(text)

        nsdt_score, nsdt_flags = self._nsdt.analyze(nsdt)

        if trapscore.get("detected", False) and float(trapscore.get("score", 0.0)) > 0.7:
            nsdt_flags.append("TRAP_DETECTED")
            nsdt_score = float(np.clip(nsdt_score * 0.80, 0.0, 1.0))

        citi_stage, citi_scores = self._citi.analyze(citi_stages)

        pv_state, pv_threat = self._pv.analyze(hrv, pv_manual, base.threat_score)

        cult_adj, cult_notes = self._cultural.analyze(cultural, domain, base.violations)

        depth_used, sub_results = self._fractal.analyze(text, fractal_depth)

        extended = ExtendedResult(
            nsdt_score=round(nsdt_score, 4),
            nsdt_flags=nsdt_flags,
            citi_stage=citi_stage,
            citi_scores={k: round(v, 4) for k, v in citi_scores.items()},
            polyvagal_state=pv_state,
            polyvagal_adjusted_threat=pv_threat,
            cultural_adjustments={k: round(v, 4) for k, v in cult_adj.items()},
            cultural_notes=cult_notes,
            fractal_depth_used=depth_used,
            fractal_sub_results=sub_results,
        )

        return base, extended
