"""Currency-recognition demo placeholder; it does not identify denominations."""

from __future__ import annotations

import numpy as np


class CurrencyRecognizer:
	"""Honest stand-in for a future note detector and denomination classifier."""

	def recognize(self, image: np.ndarray) -> str:
		"""Do not infer a rupee value until a trained currency model is integrated."""
		return (
			"Currency recognition is a demo placeholder. I cannot identify the note or its denomination."
		)