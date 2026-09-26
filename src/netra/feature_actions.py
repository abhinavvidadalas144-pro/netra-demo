"""Background worker for frame-based OCR and currency demo actions."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

from .currency import CurrencyRecognizer
from .text_reading import extract_text

logger = logging.getLogger(__name__)


class FeatureActionWorker:
	"""Run read/currency requests away from the real-time detector loop."""

	def __init__(self, message_queue: queue.Queue[str]) -> None:
		self._message_queue = message_queue
		self._requests: queue.Queue[tuple[str, np.ndarray]] = queue.Queue(maxsize=4)
		self._currency_recognizer = CurrencyRecognizer()
		self._stop_event = threading.Event()
		self._thread: Optional[threading.Thread] = None

	def start(self) -> None:
		self._thread = threading.Thread(
			target=self._process_requests,
			name="netra-feature-actions",
			daemon=True,
		)
		self._thread.start()

	def submit(self, action: str, image: np.ndarray) -> bool:
		if action not in {"read", "currency"}:
			raise ValueError("Unsupported feature action: {}".format(action))
		try:
			self._requests.put_nowait((action, image.copy()))
			return True
		except queue.Full:
			logger.warning("Feature action queue is full; dropping %s request.", action)
			return False

	def stop(self, join_timeout: float = 3.0) -> None:
		self._stop_event.set()
		if self._thread is not None and self._thread.is_alive():
			self._thread.join(join_timeout)

	def _process_requests(self) -> None:
		while not self._stop_event.is_set() or not self._requests.empty():
			try:
				action, image = self._requests.get(timeout=0.2)
			except queue.Empty:
				continue
			try:
				if action == "read":
					text = extract_text(image)
					message = text if text else "no text detected"
				else:
					message = self._currency_recognizer.recognize(image)
			except RuntimeError as error:
				logger.error("%s", error)
				message = "Text reading unavailable. Install Tesseract OCR."
			except Exception:
				logger.exception("Could not complete %s action.", action)
				message = "Sorry, that request could not be completed."
			try:
				self._message_queue.put_nowait(message)
			except queue.Full:
				logger.warning("Speech queue is full; dropping feature response.")
			self._requests.task_done()