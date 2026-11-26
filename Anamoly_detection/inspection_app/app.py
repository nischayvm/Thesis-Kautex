"""PyQt6 front end for single-frame ANNAlyzer inspection."""

import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QImageReader, QPixmap
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from comparison_view import ImageComparison
from lighting import SAMPLE_STEP, detect_lighting
from pipeline_view import PipelineView
from worker import ANOMALY, FRAME, LIGHTS, MODELS


class ImagePath(QLineEdit):
    def __init__(self, placeholder, load):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setAcceptDrops(True)
        self.load = load
        self.returnPressed.connect(self.submit)

    def submit(self):
        path = self.text().strip().strip('"')
        if path:
            self.load(Path(path))

    def browse(self, title):
        path, _ = QFileDialog.getOpenFileName(self.window(), title, "", "PNG images (*.png)")
        if path:
            self.setText(path)
            self.submit()

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if self.isEnabled() and len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if self.isEnabled() and len(urls) == 1 and urls[0].isLocalFile():
            self.setText(urls[0].toLocalFile())
            self.submit()
            event.acceptProposedAction()
        else:
            event.ignore()


class Inspection(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ANNAlyzer Tray Inspection")
        self.resize(1040, 680)
        self.session = TemporaryDirectory(prefix="annalyzer-inspection-")
        self.input_path = Path(self.session.name) / "tray.png"
        self.overlay_path = Path(self.session.name) / "overlay.png"
        self.heatmap_path = Path(self.session.name) / "heatmap.png"
        self.mask_overlay_path = Path(self.session.name) / "ground_truth.png"
        self.active = None
        self.ready = False
        self.operation = None
        self.detected = None
        self.image_size = None
        self.has_prediction = False
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.protocol = ""
        self.worker = QProcess(self)
        self.worker.readyReadStandardOutput.connect(self.receive)
        self.worker.readyReadStandardError.connect(self.read_errors)
        self.worker.finished.connect(self.worker_finished)
        self.worker.errorOccurred.connect(self.worker_error)

        self.image_path_field = ImagePath("Drop a tray PNG or paste its path and press Enter", self.load_image)
        self.open_button = QPushButton("Open image…")
        self.open_button.clicked.connect(lambda: self.image_path_field.browse("Open tray image"))
        self.inspect_button = QPushButton("Inspect")
        self.inspect_button.clicked.connect(self.inspect_image)
        self.pipeline_button = QPushButton("Show pipeline")
        self.pipeline_button.setCheckable(True)
        self.pipeline_button.toggled.connect(self.toggle_pipeline)
        self.model = QComboBox()
        for key, label in MODELS.items():
            self.model.addItem(label, key)
        self.light = QComboBox()
        self.light.addItem("Automatic", None)
        for key in LIGHTS:
            self.light.addItem(key.title(), key)
        self.model.currentIndexChanged.connect(self.selection_changed)
        self.light.currentIndexChanged.connect(self.selection_changed)

        self.filename = QLabel()
        self.filename.setTextFormat(Qt.TextFormat.PlainText)
        self.image_info = QLabel()
        self.image_info.setWordWrap(True)
        self.decision = QLabel("Awaiting image")
        decision_font = self.decision.font()
        decision_font.setPointSize(26)
        decision_font.setBold(True)
        self.decision.setFont(decision_font)
        self.decision.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail = QLabel()
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.status = QLabel("Load a tray image to begin.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(120)
        self.images = ImageComparison()
        self.pipeline = PipelineView()
        self.views = QStackedWidget()

        result_box = QGroupBox("Inspection result")
        result_box.setMinimumHeight(135)
        result_layout = QVBoxLayout(result_box)
        result_layout.addWidget(self.decision)
        result_layout.addWidget(self.detail)

        self.comparison_box = QGroupBox("Ground truth (optional)")
        self.mask_path_field = ImagePath("Drop a mask PNG or paste its path and press Enter", self.load_mask)
        self.open_mask_button = QPushButton("Open mask…")
        self.open_mask_button.clicked.connect(lambda: self.mask_path_field.browse("Open ground-truth mask"))
        mask_controls = QHBoxLayout()
        mask_controls.addWidget(self.mask_path_field, stretch=1)
        mask_controls.addWidget(self.open_mask_button)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        self.metrics.setTextFormat(Qt.TextFormat.PlainText)
        self.mask_hint = QLabel("Run Inspect to enable mask comparison.")
        comparison = QVBoxLayout(self.comparison_box)
        comparison.addLayout(mask_controls)
        comparison.addWidget(self.mask_hint)
        comparison.addWidget(self.metrics)
        comparison.addStretch()

        image_controls = QHBoxLayout()
        image_controls.addWidget(self.image_path_field, stretch=1)
        image_controls.addWidget(self.open_button)
        controls = QHBoxLayout()
        for widget in (QLabel("Model"), self.model, QLabel("Lighting"), self.light):
            controls.addWidget(widget)
        controls.addStretch()
        controls.addWidget(self.pipeline_button)
        controls.addWidget(self.inspect_button)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status, stretch=1)
        status_row.addWidget(self.progress)
        results = QHBoxLayout()
        results.addWidget(result_box, stretch=2)
        results.addWidget(self.comparison_box, stretch=3)
        self.views.addWidget(self.images)
        self.views.addWidget(self.pipeline)
        layout = QVBoxLayout(self)
        layout.addLayout(image_controls)
        layout.addWidget(self.filename)
        layout.addLayout(controls)
        layout.addWidget(self.image_info)
        layout.addLayout(status_row)
        layout.addLayout(results)
        layout.addWidget(self.views, stretch=1)
        self.refresh_controls()

    @property
    def busy(self):
        return self.operation is not None

    def selected_light(self):
        return self.light.currentData() or self.detected

    def refresh_controls(self):
        for widget in (self.image_path_field, self.open_button, self.model, self.light):
            widget.setEnabled(not self.busy)
        self.inspect_button.setEnabled(self.image_size is not None and self.selected_light() is not None and not self.busy)
        self.pipeline_button.setEnabled(self.image_size is not None)
        self.comparison_box.setEnabled(self.has_prediction and self.ready and not self.busy)
        self.progress.setVisible(self.busy)
        if self.image_size is None:
            self.image_info.clear()
            return
        if self.light.currentData():
            lighting = f"Using {self.selected_light()} lighting (manual). Automatic estimate: {self.detected or 'ambiguous'}."
        elif self.detected:
            lighting = f"Detected lighting: {self.detected.title()}."
        else:
            lighting = "Lighting is ambiguous. Select the render lighting before inspecting."
        transform = ""
        if self.image_size != (FRAME, FRAME):
            transform = ("\nResized to fit the model's 2048 × 2048 input with aspect ratio preserved. "
                         "The decision uses a threshold calibrated on native 2048 × 2048 renders.")
        self.image_info.setText(lighting + transform)

    def toggle_pipeline(self, enabled):
        self.pipeline_button.setText("Show images" if enabled else "Show pipeline")
        self.views.setCurrentIndex(int(enabled))
        if enabled:
            self.pipeline.show_pipeline(self.images.original.image, self.model.currentData())

    def clear_comparison(self):
        self.mask_path_field.clear()
        self.metrics.clear()
        self.images.ground_truth.show_image(QPixmap())
        self.mask_hint.setText("Run Inspect to enable mask comparison.")

    def selection_changed(self):
        self.has_prediction = False
        self.clear_comparison()
        self.images.clear_prediction()
        self.detail.clear()
        if self.image_size is None:
            self.decision.setText("Awaiting image")
            self.status.setText("Load a tray image to begin.")
        elif self.selected_light() is None:
            self.decision.setText("Select lighting")
            self.status.setText("Choose a lighting condition for this image.")
        else:
            self.decision.setText("Ready")
            self.status.setText("Click Inspect to run the selected model.")
        if self.pipeline_button.isChecked():
            self.pipeline.show_pipeline(self.images.original.image, self.model.currentData())
        self.refresh_controls()

    def load_image(self, path):
        if self.busy:
            return
        temporary = self.input_path.with_suffix(".new")
        try:
            shutil.copyfile(path, temporary)
            reader = QImageReader(str(temporary), b"png")
            # Check content through Qt before replacing the previous input snapshot.
            reader.setDecideFormatFromContent(True)
            if bytes(reader.imageFormat(str(temporary))).lower() != b"png":
                raise ValueError("Select a PNG image.")
            image = reader.read()
            del reader
            if image.isNull():
                raise ValueError("Could not read the PNG image.")
            temporary.replace(self.input_path)
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.image_size = image.width(), image.height()
        step = max(1, round(max(self.image_size) * SAMPLE_STEP / FRAME))
        pixels = (image.pixelColor(x, y).getRgb()[:3] for y in range(0, image.height(), step)
                  for x in range(0, image.width(), step))
        self.detected, _ = detect_lighting(pixels)
        self.filename.setText(f"{path.name}  ·  {image.width()} × {image.height()}")
        self.images.set_tray(QPixmap.fromImage(image))
        self.selection_changed()

    def load_mask(self, path):
        if not self.comparison_box.isEnabled():
            return
        self.images.ground_truth.show_image(QPixmap())
        self.metrics.clear()
        self.mask_hint.setText("Comparing with the ground-truth mask…")
        self.operation = "compare"
        self.status.setText("Reading the ground-truth mask…")
        self.refresh_controls()
        self.send("compare", mask=str(path.resolve()), overlay=str(self.mask_overlay_path))

    def inspect_image(self):
        key = self.model.currentData(), self.selected_light()
        if not self.inspect_button.isEnabled():
            return
        self.selection_changed()
        self.operation = "inspect"
        self.decision.setText("Working…")
        if self.active == key and self.ready and self.worker.state() == QProcess.ProcessState.Running:
            self.submit()
            return
        self.stop_worker()
        env_root = os.getenv("ANNALYZER_MODEL_ENV_ROOT")
        python = Path(env_root) / key[0] / "Scripts/python.exe" if env_root else None
        if python is None or not python.is_file():
            python = ANOMALY / key[0] / ".venv/Scripts/python.exe"
        if not python.is_file():
            self.fail(f"Model environment not found at {python}")
            return
        self.active = key
        self.stdout.clear()
        self.stderr.clear()
        self.status.setText(f"Starting {MODELS[key[0]]} for {key[1]} lighting…")
        self.refresh_controls()
        self.worker.start(str(python), ["-B", "-u", str(Path(__file__).with_name("worker.py")), *key])

    def submit(self):
        self.status.setText("Model loaded. Starting inspection…")
        self.refresh_controls()
        self.send("inspect", image=str(self.input_path), overlay=str(self.overlay_path), heatmap=str(self.heatmap_path))

    def send(self, action, **fields):
        request = json.dumps({"action": action, **fields})
        if self.worker.write((request + "\n").encode("utf-8")) < 0:
            self.fail("Could not send the request to the model worker.")

    def receive(self):
        self.stdout.extend(bytes(self.worker.readAllStandardOutput()))
        while b"\n" in self.stdout:
            line, _, remaining = self.stdout.partition(b"\n")
            self.stdout = remaining
            try:
                reply = json.loads(line)
                if not self.active:
                    continue
                if reply["event"] == "progress" and self.busy:
                    self.status.setText(reply["message"])
                elif reply["event"] == "ready" and self.operation == "inspect":
                    self.ready = True
                    self.protocol = f"{reply['protocol']} · {reply['run']}"
                    self.submit()
                elif reply["event"] == "result" and self.operation == "inspect":
                    self.show_prediction(reply)
                elif reply["event"] == "comparison" and self.operation == "compare":
                    self.show_comparison(reply)
                elif reply["event"] == "error" and self.busy:
                    self.fail(reply["message"])
            except (OSError, ValueError, KeyError) as exc:
                self.fail(f"Invalid model worker response: {exc}")

    def show_prediction(self, result):
        # Load from bytes so Qt cannot reuse a cached pixmap when this path is overwritten.
        overlay, heatmap = QPixmap(), QPixmap()
        if not overlay.loadFromData(self.overlay_path.read_bytes(), "PNG") or not heatmap.loadFromData(
                self.heatmap_path.read_bytes(), "PNG"):
            raise ValueError("Could not load the prediction images.")
        self.images.set_prediction(overlay, heatmap, result["display_max"])
        self.pipeline_button.setChecked(False)
        threshold = result["threshold"]
        label = "SCORE ONLY" if threshold is None else ("FAIL" if result["flagged"] else "PASS")
        threshold_text = "N/A" if threshold is None else f"{threshold:.6g}"
        self.decision.setText(label)
        self.detail.setText(f"Score {result['score']:.6g}  ·  Threshold {threshold_text}  ·  {result['seconds']:.2f} s\n"
                            f"{self.protocol}")
        self.has_prediction = True
        self.operation = None
        self.status.setText("Prediction complete.")
        self.mask_hint.setText("Add a matching mask to compare with this prediction.")
        self.refresh_controls()

    def show_comparison(self, result):
        preview = QPixmap()
        if not preview.loadFromData(self.mask_overlay_path.read_bytes(), "PNG"):
            raise ValueError("Could not load the ground-truth overlay.")
        self.images.ground_truth.show_image(preview)
        self.pipeline_button.setChecked(False)
        defect_pixels = result["defect_pixels"]
        agreement = result["agreement"]
        agreement_text = "N/A" if agreement is None else ("Agrees with ground truth" if agreement else "Disagrees with ground truth")
        if result["pixel_ap"] is None:
            pixel_text = "Pixel AP N/A  ·  Pixel AUROC N/A (requires both pixel classes)"
        else:
            pixel_text = f"Pixel AP {result['pixel_ap']:.4f}  ·  Pixel AUROC {result['pixel_auroc']:.4f}"
        label = "defect present" if defect_pixels else "clean"
        self.metrics.setText(f"Mask: {label} ({defect_pixels:,} defect pixels)  ·  {agreement_text}\n{pixel_text}")
        self.mask_hint.setText("Green marks defect pixels in the ground-truth overlay.")
        self.operation = None
        self.status.setText("Ground-truth comparison complete.")
        self.refresh_controls()

    def read_errors(self):
        self.stderr.extend(bytes(self.worker.readAllStandardError()))
        del self.stderr[:-4096]

    def worker_error(self, error):
        if error == QProcess.ProcessError.FailedToStart and self.active:
            self.fail(self.worker.errorString())

    def worker_finished(self, code, status):
        if self.active:
            self.receive()
            self.read_errors()
            self.ready = False
            if self.busy:
                lines = self.stderr.decode("utf-8", errors="replace").rstrip().splitlines()
                self.fail(lines[-1] if lines else f"Model worker exited with code {code}.")
            if self.has_prediction:
                self.mask_hint.setText("Run Inspect again to enable a new mask comparison.")
        self.active = None
        self.ready = False
        self.refresh_controls()

    def fail(self, message):
        if self.operation == "compare":
            self.mask_hint.setText("Choose a matching binary mask and try again.")
        else:
            self.decision.setText("Inspection failed")
        self.status.setText(message)
        self.operation = None
        self.refresh_controls()

    def stop_worker(self):
        self.active = None
        self.ready = False
        if self.worker.state() != QProcess.ProcessState.NotRunning:
            self.worker.closeWriteChannel()
            if not self.worker.waitForFinished(350):
                self.worker.kill()
                self.worker.waitForFinished(2000)

    def closeEvent(self, event):
        self.stop_worker()
        self.session.cleanup()
        super().closeEvent(event)


def main():
    application = QApplication(sys.argv)
    window = Inspection()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
