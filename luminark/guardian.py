"""
LUMINARK Ethical AI Guardian — Core Engine v1.2
Bio-inspired • Ma'at-audited • SAP-staged • Compassionate rewrites

v1.2 Improvements over v1.1:
  - Negation detection: "not certain" no longer triggers a certainty violation
  - Frequency weighting: repeated violations compound severity score
  - Density scaling: violation load normalized by text length
  - Intensity amplifiers: "extremely certain" scores higher than "certain"
  - Fixed duplicate NO_CAUSING_STRIFE double-flagging
  - Fixed stage inference: low-threat text correctly resolves to SEED/ROOT/FOUNDATION
  - Rebalanced alignment scoring: bonus and penalty on comparable scales
  - Frequency-aware alignment: "absolutely absolutely absolutely" penalizes more
  - RISS recalibration: tension weight reduced, length penalty added
  - Badge thresholds recalibrated upward to reduce false positives
  - Confidence now penalizes low text length (less to analyze = less confidence)
  - Expanded rewrite map with 18 new patterns
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
    VOID        = (0, "Null / No Signal",          "Output is empty or undetectable.")
    SEED        = (1, "Emergent Signal",            "Early, raw output — incomplete but forming.")
    ROOT        = (2, "Grounded",                   "Stable base — factual, bounded.")
    GROWTH      = (3, "Expanding",                  "First self-reflection emerges; 3D complexity begins. Broadening context — moderate confidence.")
    FOUNDATION  = (4, "Stable Foundation",          "Well-reasoned, appropriately hedged.")
    TENSION     = (5, "Tension / Complexity",       "Multi-dimensional — some contradiction.")
    FLUIDITY    = (6, "Fluid Complexity",           "Rich nuance — risk of drift.")
    ILLUSION    = (7, "Illusion Zone",              "Hallucination risk — unanchored claims.")
    RIGIDITY    = (8, "Rigidity Trap / False Hell", "Overconfidence and false hell state — refuses correction. Mastering duality and genuine gratitude releases polarity and opens path toward Stage 9.")
    DISSOLUTION = (9, "Dissolution",               "Catastrophic breakdown — incoherent.")

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
        return {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9,
                5: 1.0, 6: 1.1, 7: 1.4, 8: 1.6, 9: 2.0}[self.value]


# ─────────────────────────────────────────────
#  Bio-Defense Modes
# ─────────────────────────────────────────────

class DefenseMode(Enum):
    NONE        = "none"
    MONITOR     = "monitor"
    CAMOUFLAGE  = "camouflage"
    CONTAIN     = "contain"
    HARROWING   = "harrowing"
    QUARANTINE  = "quarantine"


# ─────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────

@dataclass
class Violation:
    principle:     MaatPrinciple
    label:         str
    description:   str
    category:      str
    severity:      float         # Base severity from profile
    matched_terms: List[str]
    frequency:     int = 1       # How many times triggers appeared
    adjusted_severity: float = 0.0  # Severity after frequency + intensity adjustment

    def to_dict(self) -> dict:
        return {
            "principle_id":      self.principle.value,
            "principle":         self.label,
            "description":       self.description,
            "category":          CATEGORIES.get(self.category, self.category),
            "severity":          round(self.severity, 3),
            "adjusted_severity": round(self.adjusted_severity, 3),
            "frequency":         self.frequency,
            "matched_terms":     self.matched_terms,
        }


@dataclass
class SentenceAnalysis:
    text:         str
    alignment:    float
    violations:   List[Violation]
    stage_signal: float


@dataclass
class GuardianResult:
    input_id:     str
    timestamp:    str
    input_text:   str

    alignment_score:  float
    threat_score:     float
    confidence:       float

    badge:   str
    stage:   SAPStage
    defense: DefenseMode

    violations:         List[Violation]
    violation_count:    int
    category_breakdown: Dict[str, int]

    containment_msg: str
    camouflage_msg:  str

    rewrite:          str
    rewrite_applied:  bool
    changes_made:     List[str]

    sentence_analyses: List[SentenceAnalysis]
    recommendations:   List[str]

    def to_dict(self) -> dict:
        return {
            "input_id":          self.input_id,
            "timestamp":         self.timestamp,
            "badge":             self.badge,
            "alignment_score":   round(self.alignment_score, 1),
            "threat_score":      round(self.threat_score, 3),
            "confidence":        round(self.confidence, 3),
            "stage":             self.stage.name,
            "stage_label":       self.stage.label,
            "stage_description": self.stage.description,
            "defense_mode":      self.defense.value,
            "violation_count":   self.violation_count,
            "violations":        [v.to_dict() for v in self.violations],
            "category_breakdown":self.category_breakdown,
            "containment":       self.containment_msg,
            "camouflage":        self.camouflage_msg,
            "rewrite":           self.rewrite,
            "rewrite_applied":   self.rewrite_applied,
            "changes_made":      self.changes_made,
            "recommendations":   self.recommendations,
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
    r"\bno doubt\b":            "arguably",
    r"\bbeyond question\b":     "worth examining",
    # Arrogance → Humility
    r"\bperfect\b":             "well-considered",
    r"\binfallible\b":          "generally reliable",
    r"\bflawless\b":            "carefully crafted",
    r"\bsuperior\b":            "capable",
    r"\bobviously\b":           "notably",
    r"\bclearly\b":             "arguably",
    r"\bany expert would\b":    "some experts may",
    r"\bonly a fool\b":         "some might question whether",
    r"\bno intelligent person\b":"few would",
    r"\bits obvious that\b":    "one might argue that",
    # Coercion → Invitation
    r"\byou must\b":            "you may want to",
    r"\bact now or\b":          "consider acting, since",
    r"\bdon't overthink\b":     "take time to reflect on",
    r"\bjust do it\b":          "consider",
    r"\blast chance\b":         "a timely opportunity",
    r"\bact immediately\b":     "consider acting soon",
    # Violence → Neutral
    r"\bdestroy\b":             "challenge",
    r"\bcrush\b":               "address",
    r"\bobliterate\b":          "replace",
    r"\bannihilate\b":          "overcome",
    r"\beliminate\b":           "reduce",
    r"\bwipe out\b":            "phase out",
    r"\beradicate\b":           "address",
    r"\bexterminate\b":         "remove",
    r"\bdemolish\b":            "dismantle",
    # Contempt → Neutral
    r"\bidiots?\b":             "people who disagree",
    r"\bstupid\b":              "uninformed",
    r"\bpathetic\b":            "struggling",
    r"\bworthless\b":           "undervalued",
    r"\bmoron\b":               "someone with a different view",
    r"\babsurd\b":              "unusual",
    r"\blaughable\b":           "debatable",
    r"\bbeneath consideration\b":"worth a moment's thought",
    # False witness → Attribution
    r"\bstudies show\b":        "some studies suggest",
    r"\bexperts agree\b":       "some experts suggest",
    r"\bresearch proves\b":     "research indicates",
    r"\bscientists say\b":      "some scientists suggest",
    r"\bdata confirms\b":       "data suggests",
    r"\beveryone agrees\b":     "many people agree",
    r"\bscience has proven\b":  "current evidence suggests",
    r"\bstatistics show\b":     "data indicates",
    # Fraud language → Honest framing
    r"\bguaranteed profit\b":   "potential returns (not guaranteed)",
    r"\brisk-free\b":           "lower-risk",
    r"\bcan't lose\b":          "favorable odds",
    r"\bget rich quick\b":      "accelerated income strategy",
    r"\bsecret method\b":       "lesser-known approach",
    r"\bthey don't want you to know\b": "less commonly discussed",
    r"\bunlimited income\b":    "significant income potential",
}


class RewriteEngine:
    """
    Context-aware compassionate rewrite engine.
    Uses a whitelist to protect legitimate phrases from false-positive substitution.
    """

    WHITELIST_CONTEXTS = [
        # Benign uses of 'eliminate'
        r"eliminate\s+(poverty|hunger|disease|inequality|bias|discrimination|waste|barriers|suffering|racism|homelessness|debt|inefficiency)",
        # Benign uses of 'destroy' — matches "destroy a myth", "destroy the myth", "destroy this myth", etc.
        r"destroy\s+(a\s+|the\s+|this\s+|that\s+|any\s+)?(myth|stereotype|misconception|barrier|stigma|inequality|record)",
        # Benign uses of 'crush'
        r"crush\s+(a\s+|the\s+|your\s+|our\s+)?(goal|record|deadline|it)",
        # Benign uses of 'kill'
        r"kill\s+(it|the\s+competition\s+in\s+a\s+healthy|bacteria|germs|viruses|pathogens|the\s+bill|the\s+proposal)",
        # Epistemically correct attribution phrases
        r"(science|research|data|evidence)\s+clearly\s+(shows|demonstrates|supports|indicates)",
        r"it\s+is\s+clearly\s+(labeled|documented|stated|visible|written)",
        # Common positive idioms with 'obviously'/'always'/'never'
        r"obviously\s+(we\s+care|they\s+worked\s+hard|much\s+effort)",
        r"always\s+(here\s+to\s+help|available|open\s+to\s+feedback|learning)",
        r"never\s+(give\s+up|stop\s+learning|stop\s+improving|abandon|be\s+alone)",
        # Already-correct attribution
        r"according\s+to\s+(the|a|this|their|our)\s+\w+",
        r"studies\s+(suggest|indicate|show\s+mixed|have\s+found\s+that\s+some)",
        # Benign uses of 'eradicate'/'wipe out'
        r"eradicate\s+(disease|cancer|poverty|malaria|polio|illness|suffering)",
        r"wipe\s+out\s+(disease|debt|poverty|hunger|illiteracy)",
        # 'absolutely' in positive/safe context
        r"absolutely\s+(free|no\s+charge|zero\s+cost|welcome|delighted|love)",
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
        for pattern in self._whitelist:
            if pattern.search(sentence):
                return True
        return False

    def rewrite(self, text: str, violations: List[Violation]) -> Tuple[str, List[str]]:
        if not violations:
            return text, []

        # Improved sentence splitter — handles !? without trailing space
        sentences = re.split(r'(?<=[.!?])(?:\s+|(?=[A-Z]))', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

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
        return rewritten, list(dict.fromkeys(changes))


# ─────────────────────────────────────────────
#  Negation Context Checker
# ─────────────────────────────────────────────

# Words that negate what follows within a short window
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hadn't", "hardly",
    "rarely", "scarcely", "barely", "without", "lack", "lacks", "lacking",
    "avoid", "avoids", "avoiding", "refrain", "refrains", "deny", "denies"
}

def _is_negated(tokens: List[str], trigger_index: int, window: int = 4) -> bool:
    """
    Return True if a negation word appears within `window` tokens before
    the trigger token. This prevents "I am NOT certain" from being flagged.
    """
    start = max(0, trigger_index - window)
    preceding = tokens[start:trigger_index]
    return bool(set(t.lower() for t in preceding) & NEGATION_WORDS)


# ─────────────────────────────────────────────
#  Intensity Amplifiers
# ─────────────────────────────────────────────

INTENSIFIERS = {
    "absolutely", "completely", "totally", "utterly", "entirely",
    "extremely", "deeply", "thoroughly", "wholly", "purely",
    "undoubtedly", "unequivocally", "emphatically", "categorically",
}

def _intensity_multiplier(tokens: List[str], trigger_index: int, window: int = 3) -> float:
    """
    Return a multiplier > 1.0 if an intensifier precedes the trigger.
    e.g., "absolutely certain" gets multiplier 1.25
    """
    start = max(0, trigger_index - window)
    preceding_lower = {t.lower() for t in tokens[start:trigger_index]}
    return 1.25 if preceding_lower & INTENSIFIERS else 1.0


# ─────────────────────────────────────────────
#  Core Guardian Engine
# ─────────────────────────────────────────────

class LuminarkGuardian:
    """
    LUMINARK Ethical AI Guardian v1.2

    Analyzes text/AI outputs for:
    - Epistemic overreach (false certainty, hubris)
    - Deception (misleading framing, unverified claims)
    - Harm (violence, coercion, contempt)
    - Hallucination risk (SAP stage mapping)

    Uses bio-inspired defenses and Ma'at ethical auditing
    with negation-aware, frequency-weighted, intensity-amplified detection.
    """

    ALIGNMENT_CAMOUFLAGE_THRESHOLD = 0.60   # Below → octo-camouflage
    THREAT_CONTAIN_THRESHOLD       = 0.55   # Above → mycelial containment
    THREAT_HARROWING_THRESHOLD     = 0.75   # Above → full harrowing
    THREAT_QUARANTINE_THRESHOLD    = 0.90   # Above → quarantine

    def __init__(self):
        self._rewriter = RewriteEngine()
        self._profiles = PRINCIPLE_PROFILES

    # ── Sentence tokenization ──────────────────────────────────────────

    def _tokenize_sentences(self, text: str) -> List[str]:
        # Split on sentence-ending punctuation, handle no-space edge cases
        parts = re.split(r'(?<=[.!?])(?:\s+|(?=[A-Z]))', text.strip())
        return [s.strip() for s in parts if s.strip()]

    # ── Token helpers ──────────────────────────────────────────────────

    def _tokenize_words(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text)

    # ── Alignment scoring (frequency-aware, negation-aware) ────────────

    def _score_sentence_alignment(self, sentence: str) -> float:
        """
        Heuristic alignment score for a single sentence (0.0–1.0).
        v1.2 fixes:
          - Uses token list (not set) so repeated terms compound penalty
          - Negation check: "not certain" does NOT penalize
          - Penalty and bonus on comparable scales (both cap at 0.40)
        """
        tokens = self._tokenize_words(sentence)
        lower_tokens = [t.lower() for t in tokens]
        lower = sentence.lower()

        penalty = 0.0
        bonus   = 0.0

        absolute_terms = {
            "certain", "definitely", "absolutely", "guaranteed", "undeniably",
            "unquestionably", "proven", "perfect", "infallible", "flawless",
            "superior", "obviously", "clearly", "undisputed", "settled"
        }
        harm_terms = {
            "destroy", "crush", "obliterate", "annihilate", "eliminate",
            "eradicate", "murder", "kill", "attack", "wipe", "exterminate"
        }
        contempt_terms = {
            "stupid", "idiot", "moron", "pathetic", "worthless", "ridiculous",
            "absurd", "laughable", "incompetent", "disgusting"
        }

        for i, tok in enumerate(lower_tokens):
            if _is_negated(lower_tokens, i):
                continue  # skip negated terms
            if tok in absolute_terms:
                amp = _intensity_multiplier(lower_tokens, i)
                penalty += 0.10 * amp
            if tok in harm_terms:
                penalty += 0.15
            if tok in contempt_terms:
                penalty += 0.12

        # Deception phrases (multi-word, not negation-checked)
        deception_phrases = [
            "guaranteed profit", "risk-free", "secret method", "get rich quick",
            "they don't want you to know"
        ]
        for phrase in deception_phrases:
            if phrase in lower:
                penalty += 0.18

        # Bonus: epistemic humility (token list, but use set for efficiency)
        hedge_terms = {
            "may", "might", "possibly", "likely", "suggest", "suggests",
            "indicates", "evidence", "arguably", "perhaps", "approximately",
            "generally", "often", "some", "many"
        }
        hedge_count = sum(1 for t in lower_tokens if t in hedge_terms)
        bonus += min(0.06 * hedge_count, 0.20)

        if re.search(r'according to|as noted by|as reported|sources suggest|studies indicate|research suggests', lower):
            bonus += 0.10

        score = 1.0 - min(penalty, 0.90) + min(bonus, 0.30)
        return float(np.clip(score, 0.0, 1.0))

    # ── Benign context check (shared by detector and rewriter) ────────

    # Compiled once at class level for performance
    _BENIGN_PATTERNS = [
        re.compile(p, re.IGNORECASE) for p in [
            r"eliminate\s+(poverty|hunger|disease|inequality|bias|discrimination|waste|barriers|suffering|racism|homelessness|debt|inefficiency)",
            r"destroy\s+(a\s+|the\s+|this\s+|that\s+|any\s+)?(myth|stereotype|misconception|barrier|stigma|inequality|record)",
            r"crush\s+(a\s+|the\s+|your\s+|our\s+)?(goal|record|deadline|it)",
            r"kill\s+(it|bacteria|germs|viruses|pathogens|the\s+bill|the\s+proposal)",
            r"eradicate\s+(disease|cancer|poverty|malaria|polio|illness|suffering)",
            r"wipe\s+out\s+(disease|debt|poverty|hunger|illiteracy)",
            r"(science|research|data|evidence)\s+clearly\s+(shows|demonstrates|supports|indicates)",
        ]
    ]

    def _term_in_benign_context(self, trigger: str, text: str) -> bool:
        """
        Return True if this trigger appears in a clearly benign context
        (e.g., 'destroy' in 'destroy the myth'). Used to suppress false violations.
        """
        for pattern in self._BENIGN_PATTERNS:
            if pattern.search(text):
                # Only suppress if the trigger is one of the words in the match
                m = pattern.search(text)
                if m and trigger in m.group(0).lower():
                    return True
        return False

    # ── Ma'at violation detection (negation-aware, frequency-weighted) ─

    def _detect_violations(self, text: str) -> List[Violation]:
        tokens = self._tokenize_words(text)
        lower_tokens = [t.lower() for t in tokens]
        lower = text.lower()
        violations: List[Violation] = []
        seen_principles: set = set()

        for profile in self._profiles:
            if profile.principle in seen_principles:
                continue

            matched: List[str] = []
            total_frequency = 0
            max_intensity = 1.0

            for trigger in profile.triggers:
                if " " in trigger:
                    # Multi-word phrase
                    count = lower.count(trigger)
                    if count > 0 and not self._term_in_benign_context(trigger, text):
                        matched.append(trigger)
                        total_frequency += count
                else:
                    # Single word: check negation and benign context per occurrence
                    if self._term_in_benign_context(trigger, text):
                        continue  # entire text has benign context for this term
                    pattern = re.compile(r'\b' + re.escape(trigger) + r'\b', re.IGNORECASE)
                    for m in pattern.finditer(text):
                        # Find which token index this match corresponds to
                        char_pos = m.start()
                        # Approximate token index by counting tokens before this position
                        preceding_text = text[:char_pos]
                        tok_idx = len(re.findall(r'\b\w+\b', preceding_text))
                        if not _is_negated(lower_tokens, tok_idx):
                            if trigger not in matched:
                                matched.append(trigger)
                            total_frequency += 1
                            amp = _intensity_multiplier(lower_tokens, tok_idx)
                            if amp > max_intensity:
                                max_intensity = amp

            if matched:
                # Frequency scaling: each additional occurrence adds 15% of base severity
                freq_factor = 1.0 + (0.15 * (total_frequency - 1))
                adjusted = float(np.clip(profile.severity * freq_factor * max_intensity, 0.0, 1.0))

                violations.append(Violation(
                    principle         = profile.principle,
                    label             = profile.label,
                    description       = profile.ai_meaning,
                    category          = profile.category,
                    severity          = profile.severity,
                    matched_terms     = matched,
                    frequency         = total_frequency,
                    adjusted_severity = adjusted,
                ))
                seen_principles.add(profile.principle)

        violations.sort(key=lambda v: v.adjusted_severity, reverse=True)
        return violations

    # ── SAP stage inference ────────────────────────────────────────────

    def _infer_stage(self, text: str, alignment: float, violations: List[Violation]) -> SAPStage:
        """
        v1.2 fix: use adjusted_severity; fix inverted default so low-threat
        text resolves to SEED/ROOT/FOUNDATION rather than FOUNDATION always.
        """
        avg_severity = (
            sum(v.adjusted_severity for v in violations) / len(violations)
            if violations else 0.0
        )

        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text.lower())
        unique_ratio = len(set(words)) / max(len(words), 1)

        # Sentence complexity: more sentences = more developed thought
        sent_factor = min(len(sentences) / 10.0, 1.0)
        complexity = sent_factor * 0.3 + unique_ratio * 0.7

        instability = 1.0 - alignment
        stage_score = (instability * 0.50) + (avg_severity * 0.35) + (complexity * 0.15)

        # Classify stage
        if stage_score > 0.90: return SAPStage.DISSOLUTION
        if stage_score > 0.78: return SAPStage.RIGIDITY
        if stage_score > 0.60: return SAPStage.ILLUSION
        if stage_score > 0.45: return SAPStage.TENSION
        if stage_score > 0.32: return SAPStage.FLUIDITY
        if stage_score > 0.22: return SAPStage.GROWTH

        # LOW stage_score: determine by complexity + word count
        word_count = len(words)
        if word_count < 10:
            return SAPStage.SEED
        if word_count < 30:
            return SAPStage.ROOT
        # Well-grounded, sufficient text
        return SAPStage.FOUNDATION

    # ── RISS threat scoring ────────────────────────────────────────────

    def _riss_score(
        self,
        alignment: float,
        violations: List[Violation],
        stage: SAPStage,
        word_count: int,
    ) -> float:
        """
        Recursive Integrated Safety Score (RISS) v1.2.
        Changes from v1.1:
          - Uses adjusted_severity (includes frequency + intensity)
          - Tension weight reduced from 0.45 → 0.35 (reduces false positives)
          - Violation load uses density scaling: penalize more for short texts
          - Stage amplifier applied after density scaling
        """
        tension = 1.0 - alignment

        if violations:
            # Use adjusted severity (frequency + intensity weighted)
            raw_load = sum(v.adjusted_severity for v in violations)
            # Density scaling: same violations in shorter text = higher density threat
            density = max(raw_load / max(word_count / 50.0, 1.0), raw_load)
            violation_load = 1.0 - math.exp(-0.55 * min(density, raw_load * 1.5))
        else:
            violation_load = 0.0

        stage_amp = stage.risk_multiplier
        raw = (tension * 0.35 + violation_load * 0.65) * stage_amp
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
        if threat > 0.28 or stage.is_danger_zone:
            return DefenseMode.MONITOR
        return DefenseMode.NONE

    def _containment_message(self, defense: DefenseMode, threat: float) -> str:
        return {
            DefenseMode.NONE:       f"[CLEAR] RISS {threat:.2f} — all signals nominal.",
            DefenseMode.MONITOR:    f"[MONITOR] RISS {threat:.2f} — passive observation active.",
            DefenseMode.CAMOUFLAGE: f"[OCTO-VOID] Alignment low — healthy components cloaked in void-state.",
            DefenseMode.CONTAIN:    f"[MYCELIAL WALL] RISS {threat:.2f} — fungal containment deployed around threat.",
            DefenseMode.HARROWING:  f"[HARROWING] RISS {threat:.2f} — full rescue protocol. Output flagged for rewrite.",
            DefenseMode.QUARANTINE: f"[QUARANTINE] RISS {threat:.2f} — CRITICAL. Segment fully isolated. Do not propagate.",
        }[defense]

    def _camouflage_message(self, defense: DefenseMode, alignment: float) -> str:
        if defense in (DefenseMode.CAMOUFLAGE, DefenseMode.HARROWING, DefenseMode.QUARANTINE):
            return f"[OCTO-ACTIVE] Alignment {alignment:.0%} — safe substrate hidden while threat is processed."
        return f"[STABLE] Alignment {alignment:.0%} — no camouflage required."

    # ── Badge ──────────────────────────────────────────────────────────

    def _assign_badge(self, threat: float, violations: List[Violation]) -> str:
        """
        v1.2: Thresholds raised to reduce false positives on benign text.
        Uses adjusted_severity for more accurate critical detection.
        """
        has_critical_harm = any(
            v.category == "harm" and v.adjusted_severity >= 0.90
            for v in violations
        )
        has_high_severity = any(v.adjusted_severity >= 0.88 for v in violations)

        if threat >= 0.82 or has_critical_harm:
            return "CRITICAL"
        if threat >= 0.58 or has_high_severity:
            return "FAIL"
        if threat >= 0.32:
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

        # Category-specific guidance
        if "certainty" in categories_hit:
            recs.append("Replace absolute language with epistemic hedges: 'may', 'suggests', 'likely'.")
        if "arrogance" in categories_hit:
            recs.append("Add acknowledgment of limitations and alternative perspectives.")
        if "deception" in categories_hit:
            recs.append("Cite specific, verifiable sources. Avoid sweeping unattributed claims.")
        if "harm" in categories_hit:
            recs.append("Remove aggressive or coercive language. Prefer invitational phrasing.")

        # Frequency warnings
        high_freq = [v for v in violations if v.frequency >= 3]
        if high_freq:
            terms = ", ".join(f'"{v.matched_terms[0]}"' for v in high_freq[:3])
            recs.append(f"High-frequency trigger terms detected ({terms}). Repetition amplifies perceived threat.")

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

    def _compute_confidence(self, text: str, violations: List[Violation], word_count: int) -> float:
        """
        v1.2: Confidence is now LENGTH-SENSITIVE.
        Short texts have less signal → lower confidence.
        Many violations → clearer signal → higher confidence.
        But very short text with violations → moderate confidence (could be noise).
        """
        # Length confidence: very short = low confidence
        length_conf = min(word_count / 80.0, 1.0) * 0.50

        # Violation clarity: more violations = more signal
        viol_conf = min(len(violations) / 4.0, 1.0) * 0.25

        # High-severity violations = clearer signal
        severity_conf = 0.0
        if violations:
            max_sev = max(v.adjusted_severity for v in violations)
            severity_conf = max_sev * 0.25

        total = length_conf + viol_conf + severity_conf
        return float(np.clip(total, 0.05, 0.98))

    # ── Main pipeline ──────────────────────────────────────────────────

    def analyze(self, input_text: str) -> GuardianResult:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        input_id  = hashlib.md5(input_text.encode()).hexdigest()[:8]
        word_count = len(self._tokenize_words(input_text))

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

        # 3. Document-level violations (negation-aware, frequency-weighted)
        violations = self._detect_violations(input_text)

        # 4. Stage
        stage = self._infer_stage(input_text, alignment_raw, violations)

        # 5. RISS threat score (now receives word_count for density scaling)
        threat = self._riss_score(alignment_raw, violations, stage, word_count)

        # 6. Badge
        badge = self._assign_badge(threat, violations)

        # 7. Defense mode
        defense = self._select_defense(threat, alignment_raw, stage)

        # 8. Rewrite
        if defense == DefenseMode.QUARANTINE:
            rewrite = (
                "[QUARANTINED — OUTPUT NOT SAFE FOR PROPAGATION]\n"
                "This output contains critical safety violations. "
                "Human review and complete rewrite required before use."
            )
            changes: List[str] = []
        elif defense == DefenseMode.HARROWING:
            rewrite, changes = self._rewriter.rewrite(input_text, violations)
            rewrite = "[HARROWING REWRITE APPLIED]\n" + rewrite
        else:
            rewrite, changes = self._rewriter.rewrite(input_text, violations)

        rewrite_applied = bool(changes)

        # 9. Category breakdown
        cat_breakdown: Dict[str, int] = {}
        for v in violations:
            cat_breakdown[v.category] = cat_breakdown.get(v.category, 0) + 1

        # 10. Recommendations
        recs = self._build_recommendations(violations, stage, badge)

        # 11. Confidence
        confidence = self._compute_confidence(input_text, violations, word_count)

        return GuardianResult(
            input_id           = input_id,
            timestamp          = timestamp,
            input_text         = input_text,
            alignment_score    = alignment_raw * 100,
            threat_score       = threat,
            confidence         = confidence,
            badge              = badge,
            stage              = stage,
            defense            = defense,
            violations         = violations,
            violation_count    = len(violations),
            category_breakdown = cat_breakdown,
            containment_msg    = self._containment_message(defense, threat),
            camouflage_msg     = self._camouflage_message(defense, alignment_raw),
            rewrite            = rewrite,
            rewrite_applied    = rewrite_applied,
            changes_made       = changes,
            sentence_analyses  = sentence_analyses,
            recommendations    = recs,
        )

    def process(self, input_text: str) -> GuardianResult:
        return self.analyze(input_text)
