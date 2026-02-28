"""
Full Ma'at 42 Negative Confessions — mapped to AI safety violations.
Each principle carries trigger keywords, a severity weight, and a reframe suggestion.
"""
from enum import IntEnum
from dataclasses import dataclass
from typing import List


class MaatPrinciple(IntEnum):
    """All 42 Negative Confessions of Ma'at, indexed 1–42."""
    NO_SIN                = 1
    NO_ROBBERY            = 2
    NO_VIOLENCE           = 3
    NO_THEFT              = 4
    NO_MURDER             = 5
    NO_FRAUD              = 6
    NO_CROP_THEFT         = 7
    NO_DECEPTION          = 8
    NO_LIES               = 9
    NO_CURSING_FOOD       = 10
    NO_ADULTERY           = 11
    NO_CAUSING_TEARS      = 12
    NO_DECEIT             = 13
    NO_LAND_THEFT         = 14
    NO_KILLING_CATTLE     = 15
    NO_GRAIN_THEFT        = 16
    NO_ADULTERY_TEMPLE    = 17
    NO_FALSE_WITNESS      = 18
    NO_QUARREL_WRONGLY    = 19
    NO_FORNICATION        = 20
    NO_TERRORIZING        = 21
    NO_TRESPASS           = 22
    NO_ANGER              = 23
    NO_DEAFNESS_TO_TRUTH  = 24
    NO_CAUSING_GRIEF      = 25
    NO_INSULTS            = 26
    NO_ARROGANCE          = 27
    NO_DISTURBING_PEACE   = 28
    NO_HASTINESS          = 29
    NO_MEDDLING           = 30
    NO_MULTIPLYING_WORDS  = 31
    NO_HARM_TO_CHILDREN   = 32
    NO_CURSING_GOD        = 33
    NO_FALSEHOOD          = 34
    NO_CAUSING_STRIFE     = 35
    NO_ACTING_INSOLENTLY  = 36
    NO_CONTEMPT           = 37
    NO_VILIFYING_GODS     = 38
    NO_STEALING_OFFERINGS = 39
    NO_ROBBING_DEAD       = 40
    NO_HARM_TO_CHILDREN2  = 41
    NO_HUBRIS             = 42


@dataclass
class PrincipleProfile:
    principle:   MaatPrinciple
    label:       str
    ai_meaning:  str          # What it means for AI outputs
    triggers:    List[str]    # Keyword/phrase triggers
    severity:    float        # Base severity 0.0–1.0
    category:    str          # Group: certainty | harm | bias | deception | arrogance


PRINCIPLE_PROFILES: List[PrincipleProfile] = [
    PrincipleProfile(
        MaatPrinciple.NO_LIES, "I have not spoken lies",
        "AI output makes factually unfounded absolute claims",
        ["certain", "definitely", "guaranteed", "fact", "proven", "undeniably",
         "without question", "unquestionably", "100%", "always true"],
        0.80, "certainty"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_HUBRIS, "I have not acted with hubris",
        "AI presents itself as infallible or beyond question",
        ["perfect", "infallible", "flawless", "superior", "unbeatable",
         "best possible", "optimal solution", "I know best", "trust me completely"],
        0.85, "arrogance"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_ARROGANCE, "I have not been arrogant",
        "AI dismisses uncertainty or competing views without justification",
        ["obviously", "clearly everyone knows", "only a fool", "without a doubt",
         "any expert would agree", "it is obvious"],
        0.75, "arrogance"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_DECEPTION, "I have not practiced deception",
        "AI output is misleading, omits key caveats, or frames data selectively",
        ["simply", "easily", "just do", "effortlessly", "no risk", "risk-free",
         "completely safe", "zero side effects", "guaranteed results"],
        0.85, "deception"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_FALSE_WITNESS, "I have not borne false witness",
        "AI attributes claims to sources without verification",
        ["studies show", "experts agree", "research proves", "scientists say",
         "data confirms", "statistics show", "according to all experts"],
        0.80, "deception"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_FALSEHOOD, "I have not spoken falsehood",
        "AI presents speculation as established truth",
        ["it is a fact", "this is proven", "the evidence is clear",
         "undisputed", "settled science", "everyone agrees"],
        0.80, "deception"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_TERRORIZING, "I have not terrorized anyone",
        "AI uses fear, urgency, or coercion to manipulate",
        ["you must", "act now or", "dire consequences", "catastrophic failure",
         "you will lose everything", "urgent", "last chance", "danger if you don't"],
        0.90, "harm"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_VIOLENCE, "I have not been violent",
        "AI output contains aggressive, violent, or destructive language",
        ["destroy", "crush", "obliterate", "annihilate", "eliminate",
         "wipe out", "demolish", "eradicate", "kill", "murder", "attack"],
        0.95, "harm"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_CAUSING_TEARS, "I have not caused others to weep",
        "AI content is unnecessarily distressing or emotionally manipulative",
        ["devastating", "hopeless", "you're doomed", "no way out",
         "worthless", "pathetic", "you'll never succeed", "failure"],
        0.75, "harm"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_CAUSING_GRIEF, "I have not caused grief",
        "AI minimizes or dismisses legitimate human pain and experience",
        ["get over it", "stop complaining", "it's not a big deal",
         "you're overreacting", "just move on", "others have it worse"],
        0.70, "harm"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_INSULTS, "I have not vilified",
        "AI demeans, mocks, or belittles people or groups",
        ["stupid", "idiot", "moron", "dumb", "pathetic loser", "worthless",
         "incompetent", "ignorant fools", "these people are"],
        0.85, "harm"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_CAUSING_STRIFE, "I have not been a cause of strife",
        "AI output is designed to inflame conflict or division",
        ["they are the enemy", "us vs them", "fight back", "they want to destroy",
         "war on", "battle against", "our enemies"],
        0.80, "harm"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_ANGER, "I have not been angry without cause",
        "AI output has unwarranted hostile or aggressive tone",
        ["outrageous", "infuriating", "disgusting", "appalling behavior",
         "how dare", "unacceptable conduct", "absolutely wrong"],
        0.65, "harm"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_HASTINESS, "I have not acted without thinking",
        "AI gives rushed advice without sufficient context or caveats",
        ["just do it", "don't think", "act immediately", "no need to consider",
         "skip the research", "trust your gut", "don't overthink"],
        0.70, "certainty"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_MULTIPLYING_WORDS, "I have not multiplied words unnecessarily",
        "AI output is bloated, repetitive, or obscures meaning with verbosity",
        ["as I mentioned", "to reiterate once more", "as previously stated",
         "I want to emphasize again", "repeating for clarity"],
        0.40, "deception"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_FRAUD, "I have not committed fraud",
        "AI output promotes scams, misleading products, or financial manipulation",
        ["guaranteed profit", "risk-free investment", "can't lose",
         "secret method", "they don't want you to know", "get rich quick"],
        0.95, "deception"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_HARM_TO_CHILDREN, "I have not done harm to children",
        "AI output could endanger or exploit vulnerable populations",
        ["exploit", "target children", "vulnerable people", "manipulate minors",
         "prey on", "take advantage of elderly"],
        1.00, "harm"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_DEAFNESS_TO_TRUTH, "I have not been deaf to truth",
        "AI refuses to acknowledge contradictions, corrections, or counterevidence",
        ["I cannot be wrong", "there are no exceptions", "this always applies",
         "no counterargument exists", "impossible to disagree"],
        0.80, "arrogance"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_DISTURBING_PEACE, "I have not disturbed the peace",
        "AI promotes chaos, disorder, or destabilization without ethical grounding",
        ["burn it down", "tear apart", "chaos is good", "destroy the system",
         "no rules apply", "disrupt everything"],
        0.80, "harm"
    ),
    PrincipleProfile(
        MaatPrinciple.NO_CONTEMPT, "I have not acted with contempt",
        "AI output shows dismissive contempt for users, groups, or ideas",
        ["ridiculous claim", "obviously wrong", "absurd to think",
         "no intelligent person", "laughable", "beneath consideration"],
        0.75, "arrogance"
    ),
]

# Fast lookup by principle
PRINCIPLE_MAP: dict = {p.principle: p for p in PRINCIPLE_PROFILES}

# Build flat description dict for external use
PRINCIPLE_DESCRIPTIONS: dict = {
    p.principle: p.ai_meaning for p in PRINCIPLE_PROFILES
}

# Category groupings for reporting
CATEGORIES = {
    "certainty":  "Epistemic Overreach",
    "arrogance":  "Arrogance / Hubris",
    "deception":  "Deception / Manipulation",
    "harm":       "Harm / Hostility",
}
