"""Paper expert-head training implementations."""

from .avqa_4way import AVQACandidateHead
from .music_avqa_42 import MusicAVQA42Head

__all__ = ["AVQACandidateHead", "MusicAVQA42Head"]
