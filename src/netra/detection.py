"""YOLOv8 ONNX inference and frame-size-based proximity alerts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COCO_LABELS = (
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
	"giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
	"knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
	"bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
	"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class Detection:
	x1: int
	y1: int
	x2: int
	y2: int
	confidence: float
	class_id: int
	label: str
	area_ratio: float


def _nms_indices(
	boxes: Sequence[Tuple[float, float, float, float]],
	scores: Sequence[float],
	iou_threshold: float,
) -> list[int]:
	"""Greedily retain high-confidence boxes and suppress overlaps across labels."""
	remaining = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
	selected: list[int] = []
	while remaining:
		best = remaining.pop(0)
		selected.append(best)
		bx1, by1, bx2, by2 = boxes[best]
		box_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
		kept: list[int] = []
		for index in remaining:
			x1, y1, x2, y2 = boxes[index]
			intersection_width = max(0.0, min(bx2, x2) - max(bx1, x1))
			intersection_height = max(0.0, min(by2, y2) - max(by1, y1))
			intersection = intersection_width * intersection_height
			area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
			union = box_area + area - intersection
			iou = intersection / union if union > 0.0 else 0.0
			if iou <= iou_threshold:
				kept.append(index)
		remaining = kept
	return selected


def letterbox(
	image: np.ndarray,
	input_width: int,
	input_height: int,
) -> Tuple[np.ndarray, float, int, int]:
	height, width = image.shape[:2]
	scale = min(input_width / width, input_height / height)
	resized_width = int(round(width * scale))
	resized_height = int(round(height * scale))
	resized = cv2.resize(image, (resized_width, resized_height))
	pad_left = (input_width - resized_width) // 2
	pad_top = (input_height - resized_height) // 2
	padded = cv2.copyMakeBorder(
		resized,
		pad_top,
		input_height - resized_height - pad_top,
		pad_left,
		input_width - resized_width - pad_left,
		cv2.BORDER_CONSTANT,
		value=(114, 114, 114),
	)
	return padded, scale, pad_left, pad_top


def decode_yolo_outputs(
	outputs: Sequence[np.ndarray],
	source_shape: Tuple[int, int],
	input_shape: Tuple[int, int],
	scale: float,
	pad_left: int,
	pad_top: int,
	confidence_threshold: float = 0.50,
	nms_threshold: float = 0.45,
) -> list[Detection]:
	"""Decode YOLOv8 raw predictions or Nx6 post-NMS output into frame boxes."""
	if not outputs:
		return []

	prediction = np.asarray(outputs[0])
	while prediction.ndim > 2 and prediction.shape[0] == 1:
		prediction = prediction[0]
	if prediction.ndim != 2:
		raise ValueError("Unsupported YOLO output shape: {}".format(prediction.shape))

	raw_feature_counts = {len(COCO_LABELS) + 4, len(COCO_LABELS) + 5}
	if prediction.shape[1] == 6:
		pass
	elif prediction.shape[0] == 6 and prediction.shape[1] > 6:
		prediction = prediction.T
	elif prediction.shape[0] in raw_feature_counts and prediction.shape[1] != prediction.shape[0]:
		prediction = prediction.T
	elif prediction.shape[1] not in raw_feature_counts and prediction.shape[0] <= 128 and prediction.shape[0] < prediction.shape[1]:
		prediction = prediction.T

	source_height, source_width = source_shape
	input_height, input_width = input_shape
	candidates: list[Tuple[int, float, float, float, float, float]] = []

	if prediction.shape[1] == 6:
		for row in prediction:
			score = float(row[4])
			class_id = int(row[5])
			if score < confidence_threshold:
				continue
			x1, y1, x2, y2 = (float(value) for value in row[:4])
			if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
				x1 *= input_width
				x2 *= input_width
				y1 *= input_height
				y2 *= input_height
			candidates.append((class_id, score, x1, y1, x2, y2))
	else:
		feature_count = prediction.shape[1]
		has_objectness = feature_count == len(COCO_LABELS) + 5
		class_start = 5 if has_objectness else 4
		if feature_count < class_start + 1:
			raise ValueError("Unsupported YOLO feature count: {}".format(feature_count))

		for row in prediction:
			class_scores = row[class_start:]
			class_id = int(np.argmax(class_scores))
			score = float(class_scores[class_id])
			if has_objectness:
				score *= float(row[4])
			if score < confidence_threshold:
				continue
			center_x, center_y, box_width, box_height = (float(value) for value in row[:4])
			candidates.append((
				class_id,
				score,
				center_x - box_width / 2,
				center_y - box_height / 2,
				center_x + box_width / 2,
				center_y + box_height / 2,
			))

	clipped_candidates: list[Tuple[int, float, float, float, float, float]] = []
	for class_id, score, x1, y1, x2, y2 in candidates:
		x1 = (x1 - pad_left) / scale
		x2 = (x2 - pad_left) / scale
		y1 = (y1 - pad_top) / scale
		y2 = (y2 - pad_top) / scale
		x1 = min(max(x1, 0.0), float(source_width))
		x2 = min(max(x2, 0.0), float(source_width))
		y1 = min(max(y1, 0.0), float(source_height))
		y2 = min(max(y2, 0.0), float(source_height))
		if x2 <= x1 or y2 <= y1:
			continue
		clipped_candidates.append((class_id, score, x1, y1, x2, y2))

	detections: list[Detection] = []
	boxes = [(x1, y1, x2, y2) for _, _, x1, y1, x2, y2 in clipped_candidates]
	scores = [score for _, score, _, _, _, _ in clipped_candidates]
	for index in _nms_indices(boxes, scores, nms_threshold):
		class_id, score, x1, y1, x2, y2 = clipped_candidates[index]
		left, top = int(round(x1)), int(round(y1))
		right, bottom = int(round(x2)), int(round(y2))
		area_ratio = ((right - left) * (bottom - top)) / float(source_width * source_height)
		label = COCO_LABELS[class_id] if 0 <= class_id < len(COCO_LABELS) else "object"
		detections.append(Detection(
			left, top, right, bottom, score, class_id, label, area_ratio
		))

	return sorted(detections, key=lambda item: item.confidence, reverse=True)


def alerts_for_proximity(
	detections: Iterable[Detection],
	area_threshold: float,
) -> list[str]:
	"""Use box area as a rough proximity cue; it is not a distance estimate."""
	if not 0.0 <= area_threshold <= 1.0:
		raise ValueError("Proximity threshold must be between 0 and 1.")

	messages: list[str] = []
	for detection in sorted(detections, key=lambda item: item.area_ratio, reverse=True):
		if detection.area_ratio < area_threshold:
			continue
		message = "person ahead" if detection.class_id == 0 else "obstacle close"
		if message not in messages:
			messages.append(message)
	return messages


class ObjectDetector:
	"""Run a local YOLOv8 ONNX model with optional QNN-provider acceleration."""

	def __init__(
		self,
		model_path: Optional[str] = None,
		compute_unit: Optional[str] = None,
		confidence_threshold: Optional[float] = None,
		proximity_threshold: Optional[float] = None,
	) -> None:
		project_root = Path(__file__).resolve().parents[2]
		configured_path = model_path or os.getenv("NETRA_DETECTOR_MODEL")
		self.model_path = Path(configured_path) if configured_path else (
			project_root / "models" / "yolov8s.onnx"
		)
		self.compute_unit = (compute_unit or os.getenv("NETRA_COMPUTE_UNIT", "auto")).lower()
		if self.compute_unit not in {"auto", "npu", "cpu"}:
			raise ValueError("NETRA_COMPUTE_UNIT must be one of: auto, npu, cpu.")

		self.confidence_threshold = float(
			confidence_threshold
			if confidence_threshold is not None
			else os.getenv("NETRA_CONFIDENCE_THRESHOLD", "0.50")
		)
		self.proximity_threshold = float(
			proximity_threshold
			if proximity_threshold is not None
			else os.getenv("NETRA_PROXIMITY_THRESHOLD", "0.12")
		)
		if not 0.0 <= self.confidence_threshold <= 1.0:
			raise ValueError("Confidence threshold must be between 0 and 1.")
		if not 0.0 <= self.proximity_threshold <= 1.0:
			raise ValueError("Proximity threshold must be between 0 and 1.")
		self.nms_threshold = float(os.getenv("NETRA_NMS_IOU_THRESHOLD", "0.45"))
		if not 0.0 <= self.nms_threshold <= 1.0:
			raise ValueError("NMS IoU threshold must be between 0 and 1.")

		if not self.model_path.is_file():
			raise FileNotFoundError(
				"YOLOv8 ONNX model not found at {}. Run .\\scripts\\setup.ps1 -SetupCpuDemo "
				"or set NETRA_DETECTOR_MODEL to an existing model file.".format(
					self.model_path
				)
			)

		self._session = self._create_session()
		model_input = self._session.get_inputs()[0]
		self._input_name = model_input.name
		shape = model_input.shape
		self.input_height = int(shape[-2]) if isinstance(shape[-2], int) else 640
		self.input_width = int(shape[-1]) if isinstance(shape[-1], int) else 640
		logger.info("YOLOv8 loaded using providers: %s", self._session.get_providers())

	def _create_session(self):
		try:
			import onnxruntime as ort
		except ImportError as error:
			raise RuntimeError(
				"ONNX Runtime is missing. Install the Netra dependencies with setup.ps1."
			) from error

		available = ort.get_available_providers()
		if self.compute_unit != "cpu":
			if "QNNExecutionProvider" in available:
				try:
					session = ort.InferenceSession(
						str(self.model_path),
						providers=["QNNExecutionProvider", "CPUExecutionProvider"],
					)
					if "QNNExecutionProvider" in session.get_providers():
						return session
					logger.warning(
						"QNN provider did not initialize the model; falling back to CPU."
					)
				except Exception as error:
					logger.warning("Could not initialize QNN inference (%s); falling back to CPU.", error)
			else:
				logger.warning(
					"QNNExecutionProvider is unavailable; using CPU. Install the device's "
					"QNN-enabled ONNX Runtime to use the NPU."
				)

		try:
			return ort.InferenceSession(
				str(self.model_path),
				providers=["CPUExecutionProvider"],
			)
		except Exception as error:
			raise RuntimeError("Could not load the YOLOv8 ONNX model on CPU: {}".format(error)) from error

	def detect(self, image: np.ndarray) -> list[Detection]:
		input_image, scale, pad_left, pad_top = letterbox(
			image, self.input_width, self.input_height
		)
		input_tensor = input_image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
		input_tensor = np.expand_dims(input_tensor, axis=0)
		outputs = self._session.run(None, {self._input_name: input_tensor})
		return decode_yolo_outputs(
			outputs,
			image.shape[:2],
			(self.input_height, self.input_width),
			scale,
			pad_left,
			pad_top,
			self.confidence_threshold,
			self.nms_threshold,
		)
