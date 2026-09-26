"""Camera-to-detector application loop and console alert orchestration."""

from __future__ import annotations

import argparse
import logging
import os
import queue
import time
from typing import Optional, Sequence

import cv2

from .camera import CameraCapture
from .detection import Detection, ObjectDetector, alerts_for_proximity
from .feature_actions import FeatureActionWorker
from .text_to_speech import SpeechOutput
from .vlm import MockSceneDescriber

logger = logging.getLogger(__name__)


def draw_detections(image, detections: Sequence[Detection]):
	preview = image.copy()
	for detection in detections:
		cv2.rectangle(
			preview,
			(detection.x1, detection.y1),
			(detection.x2, detection.y2),
			(42, 205, 145),
			2,
		)
		text = "{} {:.0f}%".format(detection.label, detection.confidence * 100)
		cv2.putText(
			preview,
			text,
			(detection.x1, max(20, detection.y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(42, 205, 145),
			2,
			cv2.LINE_AA,
		)
	help_text = "SPACE: scene    R: read text    C: currency    Q / ESC: quit"
	cv2.rectangle(
		preview,
		(0, max(0, preview.shape[0] - 34)),
		(preview.shape[1], preview.shape[0]),
		(20, 28, 32),
		thickness=-1,
	)
	cv2.putText(
		preview,
		help_text,
		(10, preview.shape[0] - 11),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.48,
		(245, 245, 245),
		1,
		cv2.LINE_AA,
	)
	return preview


def enqueue_alerts(
	messages: Sequence[str],
	alert_queue: queue.Queue[str],
	last_alert_at: dict[str, float],
	cooldown: float,
) -> None:
	now = time.monotonic()
	for message in messages:
		if now - last_alert_at.get(message, float("-inf")) < cooldown:
			continue
		try:
			alert_queue.put_nowait(message)
		except queue.Full:
			logger.warning("Alert queue is full; dropping alert: %s", message)
		else:
			last_alert_at[message] = now


def handle_preview_key(
	key: int,
	image,
	detections: Sequence[Detection],
	scene_describer: MockSceneDescriber,
	feature_actions: FeatureActionWorker,
	speech_queue: queue.Queue[str],
	last_action_at: dict[str, float],	action_cooldown: float = 0.6,
) -> bool:
	"""Log every key and dispatch preview actions; return True when quitting."""
	if key in (-1, 255):
		return False
	if key == ord(" "):
		key_name, action = "SPACE", "scene"
	elif key in (ord("r"), ord("R")):
		key_name, action = "R", "read"
	elif key in (ord("c"), ord("C")):
		key_name, action = "C", "currency"
	elif key in (ord("q"), ord("Q"), 27):
		key_name, action = ("ESC" if key == 27 else "Q"), "quit"
	else:
		key_name, action = chr(key) if 32 <= key < 127 else str(key), "other"

	print("KEY DETECTED: {}".format(key_name), flush=True)
	logger.info("Debug preview detected key %s (action=%s).", key_name, action)
	if action == "quit":
		return True
	if action == "other":
		return False

	now = time.monotonic()
	if now - last_action_at.get(action, float("-inf")) < action_cooldown:
		logger.info("Ignoring repeated %s key during the %.1fs debounce window.", key_name, action_cooldown)
		return False
	last_action_at[action] = now

	if action == "scene":
		logger.info("Calling mock scene describer for the current preview frame.")
		try:
			message = scene_describer.describe(image, detections)
			speech_queue.put_nowait(message)
			logger.info("Scene description queued for TTS: %s", message)
		except queue.Full:
			logger.warning("Speech queue is full; dropping the scene description.")
		except Exception:
			logger.exception("Scene description action failed.")
	else:
		queued = feature_actions.submit(action, image)
		logger.info("%s action request %s queued.", key_name, "was" if queued else "was not")
	return False


def run(
	camera_index: int = 0,
	fps: float = 3.0,
	debug: bool = False,
	model_path: Optional[str] = None,
	compute_unit: Optional[str] = None,
) -> None:
	detector = ObjectDetector(model_path=model_path, compute_unit=compute_unit)
	alert_queue: queue.Queue[str] = queue.Queue(maxsize=16)
	speech_output = SpeechOutput(alert_queue)
	feature_actions = FeatureActionWorker(alert_queue)
	scene_describer = MockSceneDescriber()
	camera = CameraCapture(device_index=camera_index, fps=fps)
	cooldown = max(0.0, float(os.getenv("NETRA_ALERT_COOLDOWN_S", "3.0")))
	last_alert_at: dict[str, float] = {}
	last_sequence = -1

	speech_output.start()
	feature_actions.start()
	camera.start()
	logger.info(
		"Netra CPU detection loop started at %.1f FPS. Press Ctrl+C to stop%s.",
		fps,
		"; Space scene, R read, C currency, Q/Esc quit" if debug else "",
	)
	if debug:
		print("Click the camera window before pressing hotkeys.", flush=True)
		last_action_at: dict[str, float] = {}
	try:
		while True:
			frame = camera.next_frame(after_sequence=last_sequence, timeout=1.0)
			if frame is None:
				continue
			last_sequence = frame.sequence
			detections = detector.detect(frame.image)
			messages = alerts_for_proximity(detections, detector.proximity_threshold)
			enqueue_alerts(messages, alert_queue, last_alert_at, cooldown)

			if debug:
				cv2.imshow("Netra camera / detections", draw_detections(frame.image, detections))
				key = cv2.waitKey(1) & 0xFF
				if handle_preview_key(
					key,
					frame.image,
					detections,
					scene_describer,
					feature_actions,
					alert_queue,
					last_action_at,
				):
					break
	except KeyboardInterrupt:
		logger.info("Stopping Netra.")
	finally:
		camera.stop()
		feature_actions.stop()
		speech_output.stop()
		if debug:
			cv2.destroyAllWindows()


def main() -> None:
	parser = argparse.ArgumentParser(description="Offline Netra camera detection prototype")
	parser.add_argument("--camera-index", type=int, default=0)
	parser.add_argument(
		"--fps",
		type=float,
		default=float(os.getenv("NETRA_CAMERA_FPS", "3")),
		help="Camera capture rate (default: 3 fps)",
	)
	parser.add_argument("--debug", action="store_true", help="Show the camera preview with detection boxes")
	parser.add_argument("--model", default=None, help="Path to the exported YOLOv8 ONNX model")
	parser.add_argument(
		"--compute-unit",
		choices=("auto", "npu", "cpu"),
		default=os.getenv("NETRA_COMPUTE_UNIT", "auto"),
		help="Prefer NPU (auto/npu) or force CPU inference",
	)
	arguments = parser.parse_args()

	logging.basicConfig(
		level=os.getenv("NETRA_LOG_LEVEL", "INFO").upper(),
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)
	try:
		run(
			camera_index=arguments.camera_index,
			fps=arguments.fps,
			debug=arguments.debug,
			model_path=arguments.model,
			compute_unit=arguments.compute_unit,
		)
	except (FileNotFoundError, RuntimeError, ValueError) as error:
		logger.error("%s", error)
		raise SystemExit(1) from error
