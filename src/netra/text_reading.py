"""Local printed-text extraction using the Tesseract OCR engine."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pytesseract import TesseractNotFoundError


def extract_text(image: np.ndarray, minimum_confidence: float = 30.0) -> str:
	"""Return readable OCR words in approximate visual reading order."""
	configured_executable = os.getenv("TESSERACT_CMD")
	if configured_executable:
		pytesseract.pytesseract.tesseract_cmd = configured_executable

	if image.ndim == 3:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		gray = image
	if min(gray.shape[:2]) < 700:
		gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	try:
		words = pytesseract.image_to_data(
			processed,
			config="--oem 3 --psm 6",
			output_type=Output.DICT,
		)
	except TesseractNotFoundError as error:
		raise RuntimeError(
			"Tesseract OCR executable was not found. Install Tesseract OCR and add it to PATH, "
			"or set TESSERACT_CMD to the full path of tesseract.exe."
		) from error

	lines: dict[tuple[str, str, str], list[str]] = {}
	line_order: list[tuple[str, str, str]] = []
	for index, word in enumerate(words["text"]):
		word = word.strip()
		try:
			confidence = float(words["conf"][index])
		except (TypeError, ValueError):
			continue
		if not word or confidence < minimum_confidence:
			continue
		line_id = (
			words["block_num"][index],
			words["par_num"][index],
			words["line_num"][index],
		)
		if line_id not in lines:
			lines[line_id] = []
			line_order.append(line_id)
		lines[line_id].append(word)

	return " ".join(" ".join(lines[line_id]) for line_id in line_order)