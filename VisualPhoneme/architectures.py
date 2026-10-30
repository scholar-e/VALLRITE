"""Architecture-name families shared by training, prediction and evaluation.

Each family names checkpoints that share an input contract, so callers test
membership instead of repeating literal name sets.
"""

IMAGE_ONLY = frozenset({"image", "large-image"})
# Gated fusion: separate image gate, landmark bottleneck and per-path freezing.
GATED_FUSION = frozenset({
    "gated-fusion", "tongue-gated-fusion", "large-gated-fusion",
    "large-transformer-fusion", "autoavsr-fusion",
})
# Fusion models take (video, landmarks, landmark_mask).
FUSION = frozenset({"fusion"}) | GATED_FUSION
LANDMARK_BOTTLENECK = frozenset({"coordinates"}) | GATED_FUSION
USES_LANDMARKS = frozenset({"coordinates"}) | FUSION
USES_VIDEO = IMAGE_ONLY | FUSION
TRAINABLE = ("image", "large-image", "coordinates", "fusion", "gated-fusion",
             "tongue-gated-fusion", "large-gated-fusion",
             "large-transformer-fusion", "autoavsr-fusion")
