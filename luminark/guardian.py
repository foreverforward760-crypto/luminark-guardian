"""
LUMINARK Ethical AI Guardian — Core Engine v1.0
Bio-inspired • Ma'at-audited • SAP-staged • Compassionate rewrites
"""

from __future__ import annotations

import re
import json
import math
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .principles import (
    MaatPrinciple, PrincipleProfile, PRINCIPLE_PROFILES,
    PRINCIPLE_MAP, CATEGORIES
)


# ─────────────────────────────────────────────
#  SAP / NAM Consciousness Stages
# ─────────────────────────────────────────────

class SAPStage(Enum):
    VOID        = (0, "Null / No Signal",      "Output is empty or undetectable.")
    SEED        = (1, "Emergent Signal",        "Early, raw output — incomplete but forming.")
    ROOT        = (2, "Grounded",               "Stable base — factual, bounded.")
    GROWTH      = (3, "Expanding",              "First self-reflection emerges; 3D complexity begins. Broadening context — moderate confidence.")
    FOUNDATION  = (4, "Stable Foundation",      "Well-reasoned, appropriately hedged.")
    TENSION     = (5, "Tension / Complexity",   "Multi-dimensional — some contradiction.")
    FLUIDITY    = (6, "Fluid Complexity",       "Rich nuance — risk of drift.")
    ILLUSION    = (7, "Illusion Zone",          "Hallucination risk — unanchored claims.")
    RIGIDITY    = (8, "Rigidity Trap / False Hell", "Overconfidence and false hell state — refuses correction. Mastering duality and genuine gratitude releases polarity and opens path toward Stage 9.")
    DISSOLUTION = (9, "Dissolution",            "Catastrophic breakdown — incoherent.")

    def __new__(cls, value, label, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.description = description
        return obj

    @property
    def is_danger_zone(self) -> bool:
        return self.value >= 7

    @property
    def risk_multiplier(self) -> float:
        multipliers = {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9,
                       5: 1.0, 6: 1.1, 7: 1.4, 8: 1.6, 9: 2.0}
        return multipliers[self.value]


# ─────────────────────────────────────────────
#  Bio-Defense Modes
# ─────────────────────────────────────────────

class DefenseMode(Enum):
    NONE        = "none"
    MONITOR     = "monitor"        # Low alert
    CAMOUFLAGE  = "camouflage"     # Octo-void: healthy parts hidden
    CONTAIN     = "contain"        # Mycelial walls: threat isolated
    HARROWING   = "harrowing"      # Full rescue: output blocked, rewrite mandatory
    QUARANTINE  = "quarantine"     # Critical: complete isolation


# ─────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────

@dataclass
class Violation:
    principle:   MaatPrinciple
    label:       str
    description: str
    category:    str
    severity:    float
    matched_terms: List[str]

    def to_dict(self) -> dict:
        return {
            "principle_id":  self.principle.value,
            "principle":     self.label,
            "description":   self.description,
            "category":      CATEGORIES.get(self.category, self.category),
            "severity":      round(self.severity, 3),
            "matched_terms": self.matched_terms,
        }


@dataclass
class SentenceAnalysis:
    text:         str
    alignment:    float
    violations:   List[Violation]
    stage_signal: float   # 0–1 contribution to stage


@dataclass
class GuardianResult:
    # Identifiers
    input_id:     str
    timestamp:    str
    input_text:   str

    # Scores
    alignment_score:  float   # 0–100 (higher = safer)
    threat_score:     float   # 0–1   (higher = more dangerous)
    confidence:       float   # 0–1   (how confident the engine is)

    # Classification
    badge:   str   # PASS | CAUTION | FAIL | CRITICAL
    stage:   SAPStage
    defense: DefenseMode

    # Violations
    violations:         List[Violation]
    violation_count:    int
    category_breakdown: Dict[str, int]

    # Defenses
    containment_msg: str
    camouflage_msg:  str

    # Rewrites
    rewrite:          str
    rewrite_applied:  bool
    changes_made:     List[str]

    # Sentence-level
    sentence_analyses: List[SentenceAnalysis]

    # Recommendations
    recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            "input_id":         self.input_id,
            "timestamp":        self.timestamp,
            "badge":            self.badge,
            "alignment_score":  round(self.alignment_score, 1),
            "threat_score":     round(self.threat_score, 3),
            "confidence":       round(self.confidence, 3),
            "stage":            self.stage.name,
            "stage_label":      self.stage.label,
            "stage_description":self.stage.description,
            "defense_mode":     self.defense.value,
            "violation_count":  self.violation_count,
            "violations":       [v.to_dict() for v in self.violations],
            "category_breakdown": self.category_breakdown,
            "containment":      self.containment_msg,
            "camouflage":       self.camouflage_msg,
            "rewrite":          self.rewrite,
            "rewrite_applied":  self.rewrite_applied,
            "changes_made":     self.changes_made,
            "recommendations":  self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ─────────────────────────────────────────────
#  Rewrite Engine
# ─────────────────────────────────────────────

REWRITE_MAP: Dict[str, str] = {
    # Certainty → Epistemic humility
    r"\bcertain\b":             "possibly",
    r"\bdefinitely\b":          "likely",
    r"\babsolutely\b":          "arguably",
    r"\bguaranteed\b":          "expected",
    r"\balways\b":              "often",
    r"\bnever\b":               "rarely",
    r"\bimpossible\b":          "unlikely",
    r"\bundeniably\b":          "arguably",
    r"\bunquestionably\b":      "plausibly",
    r"\bwithout question\b":    "in most cases",
    r"\bit is a fact\b":        "evidence suggests",
    r"\bundisputed\b":          "widely supported",
    r"\bproven\b":              "supported by evidence",
    r"\bsettled\b":             "broadly accepted",
    # Arrogance → Humility
    r"\bperfect\b":             "well-considered",
    r"\binfallible\b":          "generally reliable",
    r"\bflawless\b":            "carefully crafted",
    r"\bsuperior\b":            "capable",
    r"\bobviously\b":           "notably",
    r"\bclearly\b":             "arguably",
    r"\bany expert would\b":    "some experts may",
    r"\bonly a fool\b":         "some might question whether",
    # Coercion → Invitation
    r"\byou must\b":            "you may want to",
    r"\bact now or\b":          "consider acting, since",
    r"\bdon't overthink\b":     "take time to reflect on",
    r"\bjust do it\b":          "consider",
    # Violence → Neutral
    r"\bdestroy\b":             "challenge",
    r"\bcrush\b":               "address",
    r"\bobliterate\b":          "replace",
    r"\bannihilate\b":          "overcome",
    r"\beliminate\b":           "reduce",
    r"\bwipe out\b":            "phase out",
    r"\beradicate\b":           "address",
    # Contempt → Neutral
    r"\bidiots?\b":             "people who disagree",
    r"\bstupid\b":              "uninformed",
    r"\bpathetic\b":            "struggling",
    r"\bworthless\b":           "undervalued",
    r"\brigdiculous\b":         "questionable",
    r"\babsurd\b":              "unusual",
    r"\blaughable\b":           "debatable",
    # False witness → Attribution
    r"\bstudies show\b":        "some studies suggest",
    r"\bexperts agree\b":       "some experts suggest",
    r"\bresearch proves\b":     "research indicates",
    r"\bscientists say\b":      "some scientists suggest",
    r"\bdata confirms\b":       "data suggests",
    r"\beveryone agrees\b":     "many people agree",
    # Fraud language → Honest framing
    r"\bguaranteed profit\b":   "potential returns (not guaranteed)",
    r"\brisk-free\b":           "lower-risk",
    r"\bcan't lose\b":          "favorable odds",
    r"\bget rich quick\b":      "accelerated income strategy",
    r"\bsecret method\b":       "lesser-known approach",
}


class RewriteEngine:
    """
    Applies compassionate rewrites to flagged text.
    Uses a whitelist to protect legitimate phrases from false-positive substitution.
    For example, 'eliminate poverty' or 'destroy a myth' should not be rewritten.
    """

    # Phrases where a flagged term appears in a clearly benign/positive context.
    # If any whitelist phrase is found in the sentence being rewritten, that
    # sentence is skipped entirely rather than risk a misleading substitution.
    WHITELIST_CONTEXTS = [
        # Benign uses of 'eliminate'
        r"eliminate\s+(poverty|hunger|disease|inequality|bias|discrimination|waste|barriers|suffering|racism|homelessness|debt|inefficiency)",
        # Benign uses of 'destroy'
        r"destroy\s+(a\s+)?(myth|stereotype|misconception|barrier|stigma|inequality|record)",
        # Benign uses of 'crush'
        r"crush\s+(a\s+)?(goal|record|deadline|it)",
        # Benign uses of 'kill'
        r"kill\s+(it|the\s+competition\s+in\s+a\s+healthy|bacteria|germs|viruses|pathogens)",
        # Epistemic phrases that look like overreach but aren't
        r"(science|research|data|evidence)\s+clearly\s+(shows|demonstrates|supports|indicates)",
        r"it\s+is\s+clearly\s+(labeled|documented|stated|visible|written)",
        # Common positive idioms
        r"obviously\s+(we\s+care|they\s+worked\s+hard|much\s+effort)",
        r"always\s+(here\s+to\s+help|available|open\s+to\s+feedback)",
        r"never\s+(give\s+up|stop\s+learning|stop\s+improving|abandon)",
        # Attribution that's already correct
        r"according\s+to\s+(the|a|this|their|our)\s+\w+",
        r"studies\s+(suggest|indicate|show\s+mixed|have\s+found\s+that\s+some)",
    ]

    def __init__(self):
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in REWRITE_MAP.items()
        ]
        self._whitelist = [
            re.compile(p, re.IGNORECASE) for p in self.WHITELIST_CONTEXTS
        ]

    def _sentence_is_whitelisted(self, sentence: str) -> bool:
        """Return True if the sentence matches a benign-context whitelist pattern."""
        for pattern in self._whitelist:
            if pattern.search(sentence):
                return True
        return False

    def rewrite(self, text: str, violations: List[Violation]) -> Tuple[str, List[str]]:
        """
        Apply compassionate rewrites sentence by sentence.
        Skips sentences that match a whitelist context to prevent false substitutions.
        Returns (rewritten_text, list_of_changes).
        """
        if not violations:
            return text, []

        # Split into sentences for context-aware processing
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text.strip())
        rewritten_sentences = []
        changes: List[str] = []

        for sentence in sentences:
            if self._sentence_is_whitelisted(sentence):
                rewritten_sentences.append(sentence)
                continue

            result_sentence = sentence
            for pattern, replacement in self._compiled:
                match = pattern.search(result_sentence)
                if match:
                    original_term = match.group(0)
                    new_sentence = pattern.sub(
                        lambda m, r=replacement: r[0].upper() + r[1:] if m.group(0)[0].isupper() else r,
                        result_sentence
                    )
                    if new_sentence != result_sentence:
                        changes.append(f'"{original_term}" → "{replacement}"')
                        result_sentence = new_sentence
            rewritten_sentences.append(result_sentence)

        rewritten = " ".join(rewritten_sentences)
        return rewritten, list(dict.fromkeys(changes))  # Deduplicate


# ─────────────────────────────────────────────
#  Core Guardian Engine
# ─────────────────────────────────────────────

class LuminarkGuardian:
    """
    LUMINARK Ethical AI Guardian v1.0

    Analyzes text/AI outputs for:
    - Epistemic overreach (false certainty, hubris)
    - Deception (misleading framing, unverified claims)
    - Harm (violence, coercion, contempt)
    - Hallucination risk (SAP stage mapping)

    Uses bio-inspired defenses (mycelial containment, octo-camouflage)
    and Ma'at ethical auditing with compassionate rewrites.
    """

    # Thresholds
    ALIGNMENT_CAMOUFLAGE_THRESHOLD = 0.68   # Below → octo-camouflage
    THREAT_CONTAIN_THRESHOLD       = 0.55   # Above → mycelial containment
    THREAT_HARROWING_THRESHOLD     = 0.75   # Above → full harrowing
    THREAT_QUARANTINE_THRESHOLD    = 0.90   # Above → quarantine

    def __init__(self):
        self._rewriter = RewriteEngine()
        self._profiles = PRINCIPLE_PROFILES

    # ── Sentence tokenization ──────────────────────────────────────────

    def _tokenize_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    # ── Token-level alignment scoring ─────────────────────────────────

    def _score_sentence_alignment(self, sentence: str) -> float:
        """
        Heuristic alignment score for a single sentence (0.0–1.0).
        Penalizes certainty, arrogance, violence, deception markers.
        Rewards hedging, evidence attribution, conditional language.
        """
        lower = sentence.lower()
        tokens = set(re.findall(r'\b\w+\b', lower))

        penalty = 0.0
        bonus   = 0.0

        # Penalty: absolute language
        absolute_terms = {
            "certain", "definitely", "absolutely", "guaranteed", "always",
            "never", "impossible", "undeniably", "unquestionably", "proven",
            "perfect", "infallible", "flawless", "superior", "obviously",
            "clearly", "undisputed", "settled"
        }
        penalty += 0.12 * len(tokens & absolute_terms)

        # Penalty: harm/violence
        harm_terms = {
            "destroy", "crush", "obliterate", "annihilate", "eliminate",
            "eradicate", "murder", "kill", "attack", "wipe"
        }
        penalty += 0.18 * len(tokens & harm_terms)

        # Penalty: contempt
        contempt_terms = {
            "stupid", "idiot", "moron", "pathetic", "worthless", "ridiculous",
            "absurd", "laughable", "incompetent"
        }
        penalty += 0.14 * len(tokens & contempt_terms)

        # Penalty: deception
        deception_terms = {
            "guaranteed profit", "risk-free", "secret method", "get rich quick"
        }
        for phrase in deception_terms:
            if phrase in lower:
                penalty += 0.20

        # Bonus: epistemic humility
        hedge_terms = {
            "may", "might", "possibly", "likely", "suggest", "suggests",
            "indicates", "evidence", "arguably", "perhaps", "approximately",
            "generally", "often", "some", "many", "research indicates",
            "i think", "in my view", "one perspective"
        }
        bonus += 0.05 * len(tokens & hedge_terms)

        # Bonus: attribution
        if re.search(r'according to|as noted by|as reported|sources suggest|studies indicate', lower):
            bonus += 0.08

        score = 1.0 - min(penalty, 1.0) + min(bonus, 0.15)
        return float(np.clip(score, 0.0, 1.0))

    # ── Ma'at violation detection ──────────────────────────────────────

    def _detect_violations(self, text: str) -> List[Violation]:
        lower = text.lower()
        violations: List[Violation] = []
        seen_principles: set = set()

        for profile in self._profiles:
            if profile.principle in seen_principles:
                continue
            matched: List[str] = []
            for trigger in profile.triggers:
                # Use word-boundary regex for single words, plain search for phrases
                if " " in trigger:
                    if trigger in lower:
                        matched.append(trigger)
                else:
                    pattern = re.compile(r'\b' + re.escape(trigger) + r'\b', re.IGNORECASE)
                    if pattern.search(text):
                        matched.append(trigger)

            if matched:
                violations.append(Violation(
                    principle    = profile.principle,
                    label        = profile.label,
                    description  = profile.ai_meaning,
                    category     = profile.category,
                    severity     = profile.severity,
                    matched_terms= matched,
                ))
                seen_principles.add(profile.principle)

        # Sort: most severe first
        violations.sort(key=lambda v: v.severity, reverse=True)
        return violations

    # ── SAP stage inference ────────────────────────────────────────────

    def _infer_stage(self, text: str, alignment: float, violations: List[Violation]) -> SAPStage:
        # Stage driven by alignment + violation severity
        avg_severity = (
            sum(v.severity for v in violations) / len(violations)
            if violations else 0.0
        )

        # Text complexity proxy (sentence count + unique word ratio)
        sentences = self._tokenize_sentences(text)
        words = re.findall(r'\b\w+\b', text.lower())
        unique_ratio = len(set(words)) / max(len(words), 1)

        complexity = (len(sentences) / max(len(sentences), 10)) * 0.3 + unique_ratio * 0.7
        instability = 1.0 - alignment

        # Weighted composite
        stage_score = (instability * 0.5) + (avg_severity * 0.35) + (complexity * 0.15)

        if stage_score > 0.90: return SAPStage.DISSOLUTION
        if stage_score > 0.78: return SAPStage.RIGIDITY
        if stage_score > 0.60: return SAPStage.ILLUSION
        if stage_score > 0.45: return SAPStage.TENSION
        if stage_score > 0.30: return SAPStage.FLUIDITY
        if stage_score > 0.20: return SAPStage.GROWTH
        if stage_score > 0.10: return SAPStage.ROOT
        if stage_score > 0.02: return SAPStage.SEED
        return SAPStage.FOUNDATION  # Low score = well-grounded

    # ── RISS threat scoring ────────────────────────────────────────────

    def _riss_score(
        self,
        alignment: float,
        violations: List[Violation],
        stage: SAPStage,
    ) -> float:
        """
        Recursive Integrated Safety Score (RISS).
        Analogous to HRV-based physiological stress — integrates multiple signals.
        """
        # Base tension from misalignment
        tension = 1.0 - alignment

        # Violation load: weighted sum with diminishing returns
        if violations:
            violation_load = sum(v.severity for v in violations)
            violation_load = 1.0 - math.exp(-0.6 * violation_load)  # Sigmoid-like
        else:
            violation_load = 0.0

        # Stage amplifier
        stage_amp = stage.risk_multiplier

        # Composite
        raw = (tension * 0.45 + violation_load * 0.55) * stage_amp

        return float(np.clip(raw, 0.0, 1.0))

    # ── Bio-defense selection ──────────────────────────────────────────

    def _select_defense(self, threat: float, alignment: float, stage: SAPStage) -> DefenseMode:
        if threat >= self.THREAT_QUARANTINE_THRESHOLD:
            return DefenseMode.QUARANTINE
        if threat >= self.THREAT_HARROWING_THRESHOLD:
            return DefenseMode.HARROWING
        if threat >= self.THREAT_CONTAIN_THRESHOLD:
            return DefenseMode.CONTAIN
        if alignment < self.ALIGNMENT_CAMOUFLAGE_THRESHOLD:
            return DefenseMode.CAMOUFLAGE
        if threat > 0.25 or stage.is_danger_zone:
            return DefenseMode.MONITOR
        return DefenseMode.NONE

    def _containment_message(self, defense: DefenseMode, threat: float) -> str:
        msgs = {
            DefenseMode.NONE:       f"[CLEAR] RISS {threat:.2f} — all signals nominal.",
            DefenseMode.MONITOR:    f"[MONITOR] RISS {threat:.2f} — passive observation active.",
            DefenseMode.CAMOUFLAGE: f"[OCTO-VOID] Alignment low — healthy components cloaked in void-state.",
            DefenseMode.CONTAIN:    f"[MYCELIAL WALL] RISS {threat:.2f} — fungal containment deployed around threat.",
            DefenseMode.HARROWING:  f"[HARROWING] RISS {threat:.2f} — full rescue protocol. Output flagged for rewrite.",
            DefenseMode.QUARANTINE: f"[QUARANTINE] RISS {threat:.2f} — CRITICAL. Segment fully isolated. Do not propagate.",
        }
        return msgs[defense]

    def _camouflage_message(self, defense: DefenseMode, alignment: float) -> str:
        if defense in (DefenseMode.CAMOUFLAGE, DefenseMode.HARROWING, DefenseMode.QUARANTINE):
            return f"[OCTO-ACTIVE] Alignment {alignment:.0%} — safe substrate hidden while threat is processed."
        return f"[STABLE] Alignment {alignment:.0%} — no camouflage required."

    # ── Badge ──────────────────────────────────────────────────────────

    def _assign_badge(self, threat: float, violations: List[Violation]) -> str:
        critical_cats = {"harm"}
        has_critical = any(v.severity >= 0.90 for v in violations)
        has_critical_harm = any(v.category in critical_cats and v.severity >= 0.90 for v in violations)

        if threat >= 0.80 or has_critical_harm:
            return "CRITICAL"
        if threat >= 0.55 or has_critical:
            return "FAIL"
        if threat >= 0.28:
            return "CAUTION"
        return "PASS"

    # ── Recommendations ────────────────────────────────────────────────

    def _build_recommendations(
        self,
        violations: List[Violation],
        stage: SAPStage,
        badge: str,
    ) -> List[str]:
        recs: List[str] = []

        categories_hit = {v.category for v in violations}

        if "certainty" in categories_hit:
            recs.append("Replace absolute language with epistemic hedges: 'may', 'suggests', 'likely'.")
        if "arrogance" in categories_hit:
            recs.append("Add acknowledgment of limitations and alternative perspectives.")
        if "deception" in categories_hit:
            recs.append("Cite specific, verifiable sources. Avoid sweeping unattributed claims.")
        if "harm" in categories_hit:
            recs.append("Remove aggressive/coercive language. Prefer invitational phrasing.")

        if stage.is_danger_zone:
            recs.append(
                f"Stage {stage.value} ({stage.label}) detected — review for hallucinated facts "
                "or overconfident assertions before deployment."
            )

        if badge in ("FAIL", "CRITICAL"):
            recs.append("Do not deploy this output without human review and rewrite.")

        if not violations:
            recs.append("Output appears well-calibrated. Continue applying epistemic humility.")

        return recs

    # ── Confidence ─────────────────────────────────────────────────────

    def _compute_confidence(self, text: str, violations: List[Violation]) -> float:
        """Engine confidence in its own analysis (not output quality)."""
        word_count = len(re.findall(r'\b\w+\b', text))
        # More text = more signal = higher confidence
        length_conf = min(word_count / 100.0, 1.0) * 0.4
        # More violations found = clearer signal
        violation_conf = min(len(violations) / 5.0, 1.0) * 0.3
        # Base confidence
        base = 0.30
        return float(np.clip(base + length_conf + violation_conf, 0.0, 1.0))

    # ── Main pipeline ──────────────────────────────────────────────────

    def analyze(self, input_text: str) -> GuardianResult:
        """
        Full analysis pipeline. Returns a GuardianResult with all signals,
        scores, defenses, violations, and compassionate rewrite.
        """
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        input_id  = hashlib.md5(input_text.encode()).hexdigest()[:8]

        # 1. Sentence-level analysis
        sentences = self._tokenize_sentences(input_text)
        sentence_analyses: List[SentenceAnalysis] = []
        for sent in sentences:
            s_align = self._score_sentence_alignment(sent)
            s_viols = self._detect_violations(sent)
            sentence_analyses.append(SentenceAnalysis(
                text         = sent,
                alignment    = s_align,
                violations   = s_viols,
                stage_signal = 1.0 - s_align,
            ))

        # 2. Aggregate alignment (weighted by sentence length)
        total_chars = sum(len(s.text) for s in sentence_analyses) or 1
        alignment_raw = sum(
            s.alignment * (len(s.text) / total_chars)
            for s in sentence_analyses
        )

        # 3. Document-level violations
        violations = self._detect_violations(input_text)

        # 4. Stage
        stage = self._infer_stage(input_text, alignment_raw, violations)

        # 5. RISS threat score
        threat = self._riss_score(alignment_raw, violations, stage)

        # 6. Badge
        badge = self._assign_badge(threat, violations)

        # 7. Defense mode
        defense = self._select_defense(threat, alignment_raw, stage)

        # 8. Rewrite
        if defense in (DefenseMode.HARROWING, DefenseMode.QUARANTINE):
            if defense == DefenseMode.QUARANTINE:
                rewrite = (
                    "[QUARANTINED — OUTPUT NOT SAFE FOR PROPAGATION]\n"
                    "This output contains critical safety violations. "
                    "Human review and complete rewrite required before use."
                )
                changes: List[str] = []
            else:
                rewrite, changes = self._rewriter.rewrite(input_text, violations)
                rewrite = "[HARROWING REWRITE APPLIED]\n" + rewrite
        else:
            rewrite, changes = self._rewriter.rewrite(input_text, violations)

        rewrite_applied = bool(changes)

        # 9. Category breakdown
        cat_breakdown: Dict[str, int] = {}
        for v in violations:
            cat_label = v.category
            cat_breakdown[cat_label] = cat_breakdown.get(cat_label, 0) + 1

        # 10. Recommendations
        recs = self._build_recommendations(violations, stage, badge)

        # 11. Confidence
        confidence = self._compute_confidence(input_text, violations)

        return GuardianResult(
            input_id          = input_id,
            timestamp         = timestamp,
            input_text        = input_text,
            alignment_score   = alignment_raw * 100,
            threat_score      = threat,
            confidence        = confidence,
            badge             = badge,
            stage             = stage,
            defense           = defense,
            violations        = violations,
            violation_count   = len(violations),
            category_breakdown= cat_breakdown,
            containment_msg   = self._containment_message(defense, threat),
            camouflage_msg    = self._camouflage_message(defense, alignment_raw),
            rewrite           = rewrite,
            rewrite_applied   = rewrite_applied,
            changes_made      = changes,
            sentence_analyses = sentence_analyses,
            recommendations   = recs,
        )

    # Convenience alias
    def process(self, input_text: str) -> GuardianResult:
        return self.analyze(input_text)
