"""Audio-teacher labelling for visual phoneme datasets."""

from .core import TeacherTranscript, Word, build_alignment, select_consensus

__all__ = ["TeacherTranscript", "Word", "build_alignment", "select_consensus"]
