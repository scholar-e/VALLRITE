"""Conservative visual-equivalence groups for downstream phone ambiguity."""
from __future__ import annotations

from VisualPhoneme.data import PHONEMES


VISUAL_PHONE_GROUPS: dict[str, tuple[str, ...]] = {
    "lip-closure": ("B", "M", "P"),
    "labiodental": ("F", "V"),
    "dental": ("DH", "TH"),
    "alveolar-stop-nasal": ("D", "N", "T"),
    "alveolar-fricative": ("S", "Z"),
    "postalveolar": ("CH", "JH", "SH", "ZH"),
    "velar": ("G", "K", "NG"),
    "liquid": ("L", "R"),
    "rounded-high": ("UH", "UW", "W"),
    "front-high": ("IH", "IY", "Y"),
    "central-low": ("AA", "AH", "AO"),
    "front-open": ("AE", "EH"),
    "front-diphthong": ("AY", "EY"),
    "rounded-diphthong": ("AW", "OW"),
    "rhotic-vowel": ("ER",),
    "rounded-complex": ("OY",),
    "glottal": ("HH",),
}

PHONE_TO_VISUAL_GROUP = {
    phone: group for group, members in VISUAL_PHONE_GROUPS.items() for phone in members
}
VISUAL_GROUPS = tuple(VISUAL_PHONE_GROUPS)
REST_GROUP = "rest"
VISUAL_GROUPS_WITH_REST = VISUAL_GROUPS + (REST_GROUP,)
VISUAL_GROUP_TO_ID = {group: index + 1 for index, group in enumerate(VISUAL_GROUPS)}
REST_GROUP_ID = len(VISUAL_GROUPS_WITH_REST)
REST_PHONE_ID = len(PHONEMES) + 1
PHONE_ID_TO_VISUAL_GROUP_ID = (0,) + tuple(
    VISUAL_GROUP_TO_ID[PHONE_TO_VISUAL_GROUP[phone]] for phone in PHONEMES
)
PHONE_ID_TO_VISUAL_GROUP_ID_WITH_REST = PHONE_ID_TO_VISUAL_GROUP_ID + (REST_GROUP_ID,)
if set(PHONE_TO_VISUAL_GROUP) != set(PHONEMES):
    missing = sorted(set(PHONEMES) - set(PHONE_TO_VISUAL_GROUP))
    extra = sorted(set(PHONE_TO_VISUAL_GROUP) - set(PHONEMES))
    raise RuntimeError(f"visual phone groups must partition PHONEMES: missing={missing} extra={extra}")


def visual_phone_alternatives(phones: list[str]) -> list[dict]:
    """Expand each phone position to every member of its visual group."""
    return [
        {
            "observed_phone": phone,
            "visual_group": PHONE_TO_VISUAL_GROUP[phone],
            "possible_phonemes": list(VISUAL_PHONE_GROUPS[PHONE_TO_VISUAL_GROUP[phone]]),
        }
        for phone in phones
    ]


def visual_group_alternatives(groups: list[str]) -> list[dict]:
    """Expand outputs from a group-trained model to their member phones."""
    return [
        {"visual_group": group,
         "possible_phonemes": list(VISUAL_PHONE_GROUPS[group])}
        for group in groups
    ]
