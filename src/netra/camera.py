"""Low-frame-rate camera capture with automatic reconnects."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraFrame:
	sequence: int
	captured_at: float
	image: np.ndarray


class CameraCapture:
	"""Capture webcam frames on a worker thread and retain only the newest frame."""

	def __init__(
		self,
		device_index: int = 0,
		fps: float = 3.0,
		retry_delay: float = 2.0,
		capture_factory: Optional[Callable[[int], cv2.VideoCapture]] = None,
	) -> None:
		if fps <= 0:
			raise ValueError("Camera FPS must be greater than zero.")
		if retry_delay <= 0:
			raise ValueError("Camera retry delay must be greater than zero.")

		self.device_index = device_index
		self.fps = fps
		self.retry_delay = retry_delay
		self._capture_factory = capture_factory or cv2.VideoCapture
		self._stop_event = threading.Event()
		self._condition = threading.Condition()
		self._latest_frame: Optional[CameraFrame] = None
		self._sequence = 0
		self._thread: Optional[threading.Thread] = None

	def start(self) -> None:
		if self._thread is not None and self._thread.is_alive():
			return
		self._stop_event.clear()
		self._thread = threading.Thread(
			target=self._capture_loop,
			name="netra-camera",
			daemon=True,
		)
		self._thread.start()

	def next_frame(
		self,
		after_sequence: int = -1,
		timeout: Optional[float] = None,
	) -> Optional[CameraFrame]:
		"""Wait for a frame newer than after_sequence, or return None on timeout."""
		deadline = None if timeout is None else time.monotonic() + timeout
		with self._condition:
			while not self._stop_event.is_set():
				if (
					self._latest_frame is not None
					and self._latest_frame.sequence > after_sequence
				):
					return self._latest_frame
				remaining = None if deadline is None else deadline - time.monotonic()
				if remaining is not None and remaining <= 0:
					return None
				self._condition.wait(remaining)
		return None

	def stop(self, join_timeout: float = 3.0) -> None:
		self._stop_event.set()
		with self._condition:
			self._condition.notify_all()
		if self._thread is not None and self._thread.is_alive():
			self._thread.join(join_timeout)

	def _capture_loop(self) -> None:
		frame_interval = 1.0 / self.fps
		while not self._stop_event.is_set():
			capture = self._capture_factory(self.device_index)
			if not capture.isOpened():
				logger.error(
					"No camera found at index %s; retrying in %.1f seconds.",
					self.device_index,
					self.retry_delay,
				)
				capture.release()
				self._stop_event.wait(self.retry_delay)
				continue

			logger.info("Camera %s connected at %.1f FPS.", self.device_index, self.fps)
			capture.set(cv2.CAP_PROP_FPS, self.fps)
			reconnect = False
			try:
				while not self._stop_event.is_set():
					started_at = time.monotonic()
					success, image = capture.read()
					if not success or image is None:
						logger.warning(
							"Camera %s stopped returning frames; reconnecting in %.1f seconds.",
							self.device_index,
							self.retry_delay,
						)
						reconnect = True
						break

					self._publish(image, time.time())
					delay = frame_interval - (time.monotonic() - started_at)
					if self._stop_event.wait(max(0.0, delay)):
						break
			finally:
				capture.release()

			if reconnect and not self._stop_event.is_set():
				self._stop_event.wait(self.retry_delay)

	def _publish(self, image: np.ndarray, captured_at: float) -> None:
		with self._condition:
			self._sequence += 1
			self._latest_frame = CameraFrame(self._sequence, captured_at, image)
			self._condition.notify_all()
