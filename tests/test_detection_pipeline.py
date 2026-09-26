import contextlib
import io
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from netra.app import enqueue_alerts, handle_preview_key
from netra.camera import CameraCapture
from netra.detection import ObjectDetector, alerts_for_proximity, decode_yolo_outputs
from netra.feature_actions import FeatureActionWorker
from netra.text_to_speech import SpeechOutput
from netra.text_reading import extract_text
from netra.vlm import MockSceneDescriber


class FakeCapture:
    def __init__(self, opened, image=None):
        self.opened = opened
        self.image = image

    def isOpened(self):
        return self.opened

    def set(self, _property_id, _value):
        return True

    def read(self):
        return self.opened, self.image

    def release(self):
        return None


class DetectionPipelineTests(unittest.TestCase):
    def test_ocr_filters_low_confidence_words_and_preserves_lines(self):
        ocr_data = {
            "text": ["NETRA", "unreadable", "VISION"],
            "conf": ["94", "12", "88"],
            "block_num": ["1", "1", "1"],
            "par_num": ["1", "1", "1"],
            "line_num": ["1", "1", "2"],
        }
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        with mock.patch("pytesseract.image_to_data", return_value=ocr_data):
            self.assertEqual(extract_text(image), "NETRA VISION")

    def test_read_action_queues_no_text_message_when_ocr_is_empty(self):
        messages = queue.Queue()
        worker = FeatureActionWorker(messages)
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        with mock.patch("netra.feature_actions.extract_text", return_value=""):
            worker.start()
            self.assertTrue(worker.submit("read", image))
            worker._requests.join()
            worker.stop()
        self.assertEqual(messages.get_nowait(), "no text detected")

    def test_currency_action_does_not_claim_a_denomination(self):
        messages = queue.Queue()
        worker = FeatureActionWorker(messages)
        worker.start()
        self.assertTrue(worker.submit("currency", np.zeros((80, 80, 3), dtype=np.uint8)))
        worker._requests.join()
        worker.stop()
        response = messages.get_nowait()
        self.assertIn("cannot identify", response)
        self.assertNotIn("₹100", response)

    def test_read_action_reports_missing_tesseract_instead_of_crashing(self):
        messages = queue.Queue()
        worker = FeatureActionWorker(messages)
        with mock.patch(
            "netra.feature_actions.extract_text",
            side_effect=RuntimeError("Tesseract OCR executable was not found."),
        ):
            worker.start()
            self.assertTrue(worker.submit("read", np.zeros((80, 80, 3), dtype=np.uint8)))
            worker._requests.join()
            worker.stop()
        self.assertEqual(
            messages.get_nowait(),
            "Text reading unavailable. Install Tesseract OCR.",
        )

    def test_raw_yolov8_output_decodes_person_and_car_boxes(self):
        output = np.zeros((1, 84, 2), dtype=np.float32)
        output[0, :4, 0] = (320, 320, 240, 240)
        output[0, 4, 0] = 0.95
        output[0, :4, 1] = (160, 160, 100, 100)
        output[0, 6, 1] = 0.80

        detections = decode_yolo_outputs(
            [output],
            source_shape=(640, 640),
            input_shape=(640, 640),
            scale=1.0,
            pad_left=0,
            pad_top=0,
            confidence_threshold=0.3,
        )

        self.assertEqual([item.label for item in detections], ["person", "car"])
        self.assertEqual(detections[0].area_ratio, 0.140625)
        self.assertEqual(detections[0].x1, 200)

    def test_global_nms_suppresses_overlapping_competing_labels(self):
        output = np.zeros((1, 84, 2), dtype=np.float32)
        output[0, :4, 0] = (320, 320, 240, 240)
        output[0, 4, 0] = 0.90
        output[0, :4, 1] = (322, 320, 235, 238)
        output[0, 6, 1] = 0.82

        detections = decode_yolo_outputs(
            [output], (640, 640), (640, 640), 1.0, 0, 0, confidence_threshold=0.5
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")

    def test_confidence_threshold_drops_low_confidence_candidates(self):
        output = np.zeros((1, 84, 1), dtype=np.float32)
        output[0, :4, 0] = (320, 320, 240, 240)
        output[0, 4, 0] = 0.42

        detections = decode_yolo_outputs(
            [output], (640, 640), (640, 640), 1.0, 0, 0, confidence_threshold=0.5
        )

        self.assertEqual(detections, [])

    def test_post_nms_normalized_output_decodes_person(self):
        output = np.array([[[0.25, 0.25, 0.75, 0.75, 0.9, 0]]], dtype=np.float32)
        detections = decode_yolo_outputs(
            [output], (640, 640), (640, 640), 1.0, 0, 0, confidence_threshold=0.3
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertEqual(detections[0].area_ratio, 0.25)

    def test_alerts_use_area_threshold_and_deduplicate_messages(self):
        output = np.zeros((1, 84, 3), dtype=np.float32)
        output[0, :4, 0] = (320, 320, 240, 240)
        output[0, 4, 0] = 0.95
        output[0, :4, 1] = (320, 320, 180, 180)
        output[0, 4, 1] = 0.85
        output[0, :4, 2] = (480, 480, 160, 160)
        output[0, 6, 2] = 0.90
        detections = decode_yolo_outputs(
            [output], (640, 640), (640, 640), 1.0, 0, 0, confidence_threshold=0.3
        )

        self.assertEqual(alerts_for_proximity(detections, 0.1), ["person ahead"])
        self.assertEqual(
            alerts_for_proximity(detections, 0.05),
            ["person ahead", "obstacle close"],
        )

    def test_camera_retries_after_initial_open_failure(self):
        attempts = []
        image = np.zeros((16, 16, 3), dtype=np.uint8)

        def capture_factory(_device_index):
            attempts.append(True)
            if len(attempts) == 1:
                return FakeCapture(False)
            return FakeCapture(True, image)

        camera = CameraCapture(fps=30, retry_delay=0.01, capture_factory=capture_factory)
        camera.start()
        try:
            frame = camera.next_frame(timeout=1.0)
        finally:
            camera.stop()

        self.assertIsNotNone(frame)
        self.assertGreaterEqual(len(attempts), 2)
        self.assertEqual(frame.image.shape, image.shape)

    def test_alert_queue_to_console_does_not_run_in_detection_thread(self):
        alerts = queue.Queue(maxsize=2)
        output = SpeechOutput(alerts, engine_factory=lambda: None)
        printed = io.StringIO()
        detection_thread = threading.current_thread()
        last_alert_at = {}

        with contextlib.redirect_stdout(printed):
            output.start()
            enqueue_alerts(["person ahead"], alerts, last_alert_at, cooldown=3.0)
            alerts.join()
            output.stop()

        self.assertIn("SPEAK: person ahead", printed.getvalue())
        self.assertEqual(last_alert_at.keys(), {"person ahead"})
        self.assertIs(threading.current_thread(), detection_thread)

    def test_speech_output_speaks_on_worker_thread(self):
        class FakeEngine:
            def __init__(self):
                self.messages = []

            def say(self, message):
                self.messages.append(message)

            def runAndWait(self):
                return None

        engine = FakeEngine()
        messages = queue.Queue()
        output = SpeechOutput(messages, engine_factory=lambda: engine)
        output.start()
        messages.put("person ahead")
        messages.join()
        output.stop()
        self.assertEqual(engine.messages, ["person ahead"])

    def test_mock_scene_descriptions_rotate(self):
        describer = MockSceneDescriber()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        first = describer.describe(image, [])
        second = describer.describe(image, [])
        self.assertTrue(first.startswith("Mock scene description:"))
        self.assertNotEqual(first, second)

    def test_space_key_calls_scene_provider_and_queues_speech(self):
        messages = queue.Queue()
        describer = mock.Mock()
        describer.describe.return_value = "Mock scene description: a doorway is ahead."
        feature_actions = mock.Mock()
        last_action_at = {}
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        should_quit = handle_preview_key(
            ord(" "), image, [], describer, feature_actions, messages, last_action_at
        )

        self.assertFalse(should_quit)
        describer.describe.assert_called_once_with(image, [])
        self.assertEqual(messages.get_nowait(), "Mock scene description: a doorway is ahead.")

    def test_r_key_submits_read_request_independently(self):
        feature_actions = mock.Mock()
        feature_actions.submit.return_value = True
        last_action_at = {}

        should_quit = handle_preview_key(
            ord("r"), np.zeros((32, 32, 3), dtype=np.uint8), [], mock.Mock(),
            feature_actions, queue.Queue(), last_action_at
        )

        self.assertFalse(should_quit)
        feature_actions.submit.assert_called_once()
        self.assertEqual(feature_actions.submit.call_args.args[0], "read")

    def test_detector_uses_cpu_when_qnn_provider_is_unavailable(self):
        fake_session = mock.Mock()
        fake_session.get_providers.return_value = ["CPUExecutionProvider"]
        fake_input = mock.Mock()
        fake_input.name = "images"
        fake_input.shape = [1, 3, 640, 640]
        fake_session.get_inputs.return_value = [fake_input]
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "detector.onnx"
            model_path.touch()
            with mock.patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]), \
                    mock.patch("onnxruntime.InferenceSession", return_value=fake_session) as create_session:
                detector = ObjectDetector(model_path=str(model_path), compute_unit="auto")

        self.assertEqual(detector.compute_unit, "auto")
        self.assertEqual(create_session.call_args.kwargs["providers"], ["CPUExecutionProvider"])

    def test_detector_can_force_cpu_even_if_qnn_is_available(self):
        fake_session = mock.Mock()
        fake_session.get_providers.return_value = ["CPUExecutionProvider"]
        fake_input = mock.Mock()
        fake_input.name = "images"
        fake_input.shape = [1, 3, 640, 640]
        fake_session.get_inputs.return_value = [fake_input]
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "detector.onnx"
            model_path.touch()
            with mock.patch("onnxruntime.get_available_providers", return_value=["QNNExecutionProvider", "CPUExecutionProvider"]), \
                    mock.patch("onnxruntime.InferenceSession", return_value=fake_session) as create_session:
                ObjectDetector(model_path=str(model_path), compute_unit="cpu")

        self.assertEqual(create_session.call_args.kwargs["providers"], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()