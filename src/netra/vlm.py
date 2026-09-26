"""Mock scene descriptions used until the local GenieX VLM is integrated."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .detection import Detection


class MockSceneDescriber:
	"""Return sample descriptions; replace this with Qwen3-VL-2B-Instruct/GenieX."""

	_DESCRIPTIONS = (
		"A person is standing a few steps ahead in an indoor space.",
		"A room with a table and chairs is visible in front of you.",
		"The view shows an open area with a doorway in the distance.",
	)

	def __init__(self) -> None:
		self._next_description = 0

	def describe(self, image: np.ndarray, detections: Sequence[Detection]) -> str:
		"""Stand-in for sending the current frame to Qwen3-VL through GenieX."""
		description = self._DESCRIPTIONS[self._next_description]
		self._next_description = (self._next_description + 1) % len(self._DESCRIPTIONS)
		return "Mock scene description: {}".format(description)
