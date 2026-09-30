"""Local Moondream2 CPU captioning and visual question answering."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-06-21"


class MoondreamSceneDescriber:
	"""Run actual Moondream2 inference locally; never use a hosted API."""

	def __init__(self) -> None:
		self._model = None

	def _load(self):
		if self._model is not None:
			return self._model

		try:
			import torch
			from transformers import AutoModelForCausalLM
		except ImportError as error:
			raise RuntimeError(
				"Moondream2 dependencies are missing. Install with: "
				".\\.venv\\Scripts\\python.exe -m pip install -e .[vlm]"
			) from error

		local_only = os.getenv("NETRA_VLM_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
		cache_dir = os.getenv("NETRA_VLM_CACHE")
		try:
			self._model = AutoModelForCausalLM.from_pretrained(
				MOONDREAM_MODEL_ID,
				revision=MOONDREAM_REVISION,
				trust_remote_code=True,
				torch_dtype="auto",
				local_files_only=local_only,
				cache_dir=cache_dir,
			)
			self._model.to(torch.device("cpu"))
			self._model.eval()
		except Exception as error:
			if local_only:
				raise RuntimeError(
					"Moondream2 is not available in the local Hugging Face cache. Run "
					".\\scripts\\setup.ps1 -SetupCpuDemo to download it before starting Netra. "
					"Original error: {}".format(error)
				) from error
			raise RuntimeError("Could not load Moondream2 on CPU: {}".format(error)) from error

		logger.info(
			"Loaded real Moondream2 CPU model %s at revision %s (local_files_only=%s).",
			MOONDREAM_MODEL_ID,
			MOONDREAM_REVISION,
			local_only,
		)
		return self._model

	def describe(self, image: np.ndarray, question: Optional[str] = None) -> str:
		"""Caption the supplied camera frame or answer a question about it."""
		model = self._load()
		image_rgb = cv2_bgr_to_rgb(image)
		pil_image = Image.fromarray(image_rgb)
		try:
			if question and question.strip():
				result = model.query(pil_image, question.strip())
				answer = result.get("answer", "").strip()
				prefix = "Moondream answer:"
			else:
				result = model.caption(pil_image, length="normal")
				answer = result.get("caption", "").strip()
				prefix = "Moondream scene description:"
		except Exception as error:
			raise RuntimeError("Moondream2 inference failed: {}".format(error)) from error

		if not answer:
			answer = "I could not describe the image clearly."
		return "{} {}".format(prefix, answer)


def cv2_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
	"""Convert an OpenCV BGR frame to RGB without requiring another image package."""
	if image.ndim == 2:
		return np.repeat(image[:, :, None], 3, axis=2)
	return image[:, :, :3][:, :, ::-1].copy()


class SceneDescriptionWorker:
	"""Keep slow model loading/generation off the live detection and UI loop."""

	def __init__(
		self,
		message_queue: queue.Queue[str],
		question: Optional[str] = None,
		describer: Optional[MoondreamSceneDescriber] = None,
	) -> None:
		self._message_queue = message_queue
		self._question = question
		self._describer = describer or MoondreamSceneDescriber()
		self._requests: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
		self._lock = threading.Lock()
		self._busy_since: Optional[float] = None
		self._thinking_sent = False
		self._stop_event = threading.Event()
		self._thread: Optional[threading.Thread] = None

	def start(self) -> None:
		if self._thread is not None and self._thread.is_alive():
			return
		self._thread = threading.Thread(
			target=self._process_requests,
			name="netra-moondream-worker",
			daemon=True,
		)
		self._thread.start()

	def submit(self, image: np.ndarray) -> bool:
		with self._lock:
			if self._busy_since is not None:
				logger.warning("Moondream2 is still processing; ignoring this Space request.")
				return False
			self._busy_since = time.monotonic()
			self._thinking_sent = False
		try:
			self._requests.put_nowait(image.copy())
			return True
		except queue.Full:
			with self._lock:
				self._busy_since = None
			logger.warning("Moondream request queue is full.")
			return False

	def announce_thinking_if_slow(self, delay_seconds: float = 1.5) -> bool:
		with self._lock:
			if self._busy_since is None or self._thinking_sent:
				return False
			if time.monotonic() - self._busy_since < delay_seconds:
				return False
			self._thinking_sent = True
		try:
			self._message_queue.put_nowait("Thinking. Moondream is analyzing the camera image.")
			logger.info("Moondream2 is still working; queued a spoken thinking status.")
			return True
		except queue.Full:
			logger.warning("Speech queue is full; could not queue thinking status.")
			return False

	def stop(self, join_timeout: float = 3.0) -> None:
		self._stop_event.set()
		if self._thread is not None and self._thread.is_alive():
			self._thread.join(join_timeout)

	def _process_requests(self) -> None:
		while not self._stop_event.is_set() or not self._requests.empty():
			try:
				image = self._requests.get(timeout=0.2)
			except queue.Empty:
				continue
			try:
				logger.info("Running real Moondream2 CPU inference on the captured frame.")
				response = self._describer.describe(image, self._question)
				logger.info("Moondream2 response ready for TTS.")
				self._enqueue_message(response)
			except Exception as error:
				logger.exception("Moondream2 caption/question failed.")
				self._enqueue_message("Scene description unavailable. {}".format(error))
			finally:
				with self._lock:
					self._busy_since = None
					self._thinking_sent = False
				self._requests.task_done()

	def _enqueue_message(self, message: str) -> None:
		try:
			self._message_queue.put_nowait(message)
		except queue.Full:
			logger.warning("Speech queue is full; dropping Moondream response.")