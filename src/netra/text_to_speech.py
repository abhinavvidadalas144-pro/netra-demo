"""Asynchronous Windows speech output for detector alerts and descriptions."""

from __future__ import annotations

import queue
import logging
import sys
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SpeechOutput:
	"""Speak queued messages on one worker thread so detection remains responsive."""

	def __init__(
		self,
		message_queue: queue.Queue[str],
		engine_factory: Optional[Callable[[], object]] = None,
	) -> None:
		self._message_queue = message_queue
		self._engine_factory = engine_factory
		self._stop_event = threading.Event()
		self._thread: Optional[threading.Thread] = None

	def start(self) -> None:
		self._thread = threading.Thread(
			target=self._speak_messages,
			name="netra-speech-output",
			daemon=True,
		)
		self._thread.start()

	def stop(self, join_timeout: float = 2.0) -> None:
		self._stop_event.set()
		if self._thread is not None and self._thread.is_alive():
			self._thread.join(join_timeout)

	def _create_engine(self):
		if self._engine_factory is not None:
			return self._engine_factory()
		if sys.platform == "win32":
			try:
				import pythoncom
				pythoncom.CoInitialize()
			except ImportError:
				logger.debug("pywin32 COM helpers are unavailable; pyttsx3 will initialize SAPI directly.")
		import pyttsx3
		return pyttsx3.init()

	def _speak_messages(self) -> None:
		engine = None
		try:
			engine = self._create_engine()
		except Exception:
			logger.exception("Could not initialize pyttsx3; messages will be printed instead.")

		while not self._stop_event.is_set() or not self._message_queue.empty():
			try:
				message = self._message_queue.get(timeout=0.2)
			except queue.Empty:
				continue
			print("SPEAK: {}".format(message), flush=True)
			if engine is not None:
				try:
					logger.info("Sending queued message to pyttsx3/SAPI.")
					engine.say(message)
					engine.runAndWait()
				except Exception:
					logger.exception("Speech output failed; continuing with console messages.")
					engine = None
			self._message_queue.task_done()

		if sys.platform == "win32" and self._engine_factory is None:
			try:
				import pythoncom
				pythoncom.CoUninitialize()
			except ImportError:
				pass
