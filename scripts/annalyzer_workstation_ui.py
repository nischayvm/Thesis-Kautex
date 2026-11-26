"""PyQt6 editor and live job terminal for ANNAlyzer workstation jobs."""

import argparse
import codecs
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt6.QtGui import QTextCursor, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from annalyzer_master import MENU, log_file
from workstation_config import DEFAULT, LIGHTS, LOCAL, MODELS, ROOT, read_json, rooted, validate, write_json


MODEL_NAMES = dict(zip(MODELS, ("PatchCore", "AnomalyDINO", "U-Net", "Flownomaly"), strict=True))
TABS = {"paths": "Paths", "render": "Dataset creation", "split": "Split", "training": "Training",
        **MODEL_NAMES}
LABELS = {
    "dataset": "Dataset folder", "scene": "Blender scene", "blender": "Blender executable",
    "pairs": "Clean/defect pairs per light", "resolution": "Square resolution (px)",
    "cycles_samples": "Cycles samples", "lights": "Lighting", "gpu_match": "OptiX GPU name contains",
    "camera_height": "Camera height (m)", "led_height": "LED height (m)", "train": "Train (%)",
    "validation": "Validation (%)", "accept_near_duplicates": "Reviewed candidate pairs",
    "models": "Models", "overlap": "Tile overlap (px)", "gaussian_sigma": "Stitching sigma (px)",
    "cpu_threads": "PyTorch CPU threads", "candidates": "Candidates per tile",
    "coreset": "Coreset ratio (0–1)", "learning_rate": "Learning rate", "batch": "Batch size",
    "encoder_lr": "Encoder learning rate", "flow_lr": "Flow learning rate", "flow_batch": "Flow batch size",
}
ACTION_LABELS = {
    "setup": "Install requirements", "weights": "Download or verify weights", "check": "Check GPU and environments",
    "demo": "Render a demo image and mask", "render": "Render a custom dataset", "resume": "Resume dataset render",
    "audit": "Audit and split dataset", "fit": "Fit models and validate", "benchmark": "Run original V3/V4 benchmark",
    "validate": "Revalidate a saved custom run", "test": "Run held-out test", "html": "Generate HTML report",
    "excel": "Generate Excel reports", "inspect": "Open tray inspection", "journal": "Update journal",
    "publish": "Publish saved validation results",
}
CONFIRM = {"setup", "weights", "render", "resume", "fit", "benchmark", "test", "publish"}


class Choices(QWidget):
    def __init__(self, options):
        super().__init__()
        self.order = []
        self.boxes = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for name in options:
            box = QCheckBox(MODEL_NAMES.get(name, name.title()))
            self.boxes[name] = box
            layout.addWidget(box)
        layout.addStretch()

    def set_values(self, values):
        self.order = list(values)
        for name, box in self.boxes.items():
            box.setChecked(name in values)

    def values(self):
        order = self.order + [key for key in self.boxes if key not in self.order]
        return [name for name in order if self.boxes[name].isChecked()]


class PathEdit(QWidget):
    def __init__(self, title, *, directory=False, file_filter="All files (*)"):
        super().__init__()
        self.title, self.directory, self.file_filter = title, directory, file_filter
        self.field = QLineEdit()
        button = QPushButton("Browse…")
        button.clicked.connect(self.browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.field, stretch=1)
        layout.addWidget(button)

    def browse(self):
        initial = str(rooted(self.field.text())) if self.field.text().strip() else str(ROOT)
        if self.directory:
            selected = QFileDialog.getExistingDirectory(self, self.title, initial)
        else:
            selected, _ = QFileDialog.getOpenFileName(self, self.title, initial, self.file_filter)
        if selected:
            self.field.setText(selected)

    def text(self):
        return self.field.text().strip()

    def setText(self, value):
        self.field.setText(value)


def combo(values, labels=None):
    result = QComboBox()
    for value in values:
        result.addItem(labels[value] if labels else value, value)
    return result


def field_for(key, default):
    if key == "dataset":
        return PathEdit("Choose the dataset folder", directory=True)
    if key in ("blender", "scene"):
        filters = "Executable (*.exe)" if key == "blender" else "Blender scene (*.blend)"
        result = PathEdit(LABELS[key], file_filter=filters)
        if key == "blender":
            result.field.setPlaceholderText("Automatic if blank")
        return result
    if key == "precision":
        return combo(("fp16", "bf16", "fp32"))
    if isinstance(default, list):
        return Choices(MODELS if key == "models" else LIGHTS)
    if isinstance(default, bool):
        return QCheckBox("Accept cross-fold similarity candidates after review")
    if isinstance(default, int):
        field = QSpinBox()
        field.setRange(0, 2_147_483_647)
        field.setKeyboardTracking(False)
        return field
    field = QLineEdit()
    if default is None:
        field.setPlaceholderText("Use the default render setup when blank")
    elif key == "gpu_match":
        field.setPlaceholderText("First available OptiX GPU when blank")
    return field


class JobSettings(QWidget):
    def __init__(self):
        super().__init__()
        sections = {"paths": {key: DEFAULT[key] for key in ("blender", "scene")}}
        sections.update((key, value) for key, value in DEFAULT.items() if isinstance(value, dict))
        sections["render"] = {"dataset": DEFAULT["dataset"], **DEFAULT["render"]}
        self.fields = {}
        self.sections = combo(sections, TABS)
        self.pages = QStackedWidget()
        header = QHBoxLayout()
        header.addWidget(QLabel("Settings"))
        header.addWidget(self.sections, stretch=1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.pages, stretch=1)
        self.sections.currentIndexChanged.connect(self.pages.setCurrentIndex)
        for section, values in sections.items():
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for key, default in values.items():
                label = LABELS.get(key, key.replace("_", " ").capitalize())
                field = field_for(key, default)
                form.addRow(label, field)
                parent = "paths" if key == "dataset" else section
                self.fields[parent, key] = (field, default, label)
            if section == "split":
                form.addRow(QLabel("The remaining percentage is reserved for held-out test."))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.pages.addWidget(scroll)

    def load(self, config):
        for (section, key), (field, default, _) in self.fields.items():
            value = config[key] if section == "paths" else config[section][key]
            if isinstance(field, Choices):
                field.set_values(value)
            elif isinstance(field, QComboBox):
                field.setCurrentIndex(field.findData(value))
            elif isinstance(field, QCheckBox):
                field.setChecked(value)
            elif isinstance(field, QSpinBox):
                field.setValue(value)
            else:
                field.setText("" if value is None else str(value))

    def values(self):
        config = deepcopy(DEFAULT)
        for (section, key), (field, default, label) in self.fields.items():
            if isinstance(field, Choices):
                value = field.values()
            elif isinstance(field, QComboBox):
                value = field.currentData()
            elif isinstance(field, QCheckBox):
                value = field.isChecked()
            elif isinstance(field, QSpinBox):
                value = field.value()
            else:
                text = field.text().strip()
                if default is None and not text:
                    value = None
                elif default is None or isinstance(default, float):
                    try:
                        value = float(text)
                    except ValueError as exc:
                        raise ValueError(f"Enter a number for {label}.") from exc
                else:
                    value = text
            target = config if section == "paths" else config[section]
            target[key] = value
        return validate(config)


class Workstation(QWidget):
    def __init__(self, path):
        super().__init__()
        self.setWindowTitle("ANNAlyzer Workstation")
        self.resize(1280, 760)
        self.path = path
        self.log = None
        self.decoder = None
        self.cancelled = False
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)

        self.settings = JobSettings()
        self.settings.load(validate(read_json(path)) if path.exists() else DEFAULT)
        self.path_label = QLabel(str(path))
        self.path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        self.file_actions = QWidget()
        files = QHBoxLayout(self.file_actions)
        files.setContentsMargins(0, 0, 0, 0)
        files.addWidget(QLabel("Job"))
        files.addWidget(self.path_label, stretch=1)
        for label, slot in (("Open…", self.open_job), ("Save", self.save_job), ("Save as…", self.save_as)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            files.addWidget(button)

        self.action = combo((key for key, _ in MENU if key in ACTION_LABELS), ACTION_LABELS)
        self.arguments = QStackedWidget()
        self.arguments.addWidget(QWidget())
        self.components = Choices((*MODELS, "reporting"))
        self.components.set_values([*self.settings.values()["training"]["models"], "reporting"])
        setup_page = QWidget()
        setup_form = QFormLayout(setup_page)
        setup_form.addRow("Components", self.components)
        self.arguments.addWidget(setup_page)
        self.demo_light = combo(LIGHTS)
        self.demo_kind = combo(("clean", "blistered"))
        demo = QWidget()
        demo_form = QFormLayout(demo)
        demo_form.addRow("Lighting", self.demo_light)
        demo_form.addRow("Render type", self.demo_kind)
        self.arguments.addWidget(demo)
        self.saved_model = combo(MODELS, MODEL_NAMES)
        self.saved_run = PathEdit("Choose a completed V5 run", directory=True)
        saved = QWidget()
        saved_form = QFormLayout(saved)
        saved_form.addRow("Model", self.saved_model)
        saved_form.addRow("Run folder", self.saved_run)
        self.arguments.addWidget(saved)
        self.action.currentIndexChanged.connect(self.change_action)
        self.change_action()

        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.terminal.setMaximumBlockCount(4000)
        self.terminal.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.terminal_input = QLineEdit()
        self.terminal_input.setPlaceholderText("Send input to a terminal prompt")
        self.terminal_input.returnPressed.connect(self.send_input)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_input)
        send_row = QHBoxLayout()
        send_row.addWidget(self.terminal_input, stretch=1)
        send_row.addWidget(self.send_button)
        terminal_page = QWidget()
        terminal_layout = QVBoxLayout(terminal_page)
        terminal_layout.setContentsMargins(0, 0, 0, 0)
        terminal_layout.addWidget(QLabel("Job terminal"))
        terminal_layout.addWidget(self.terminal, stretch=1)
        terminal_layout.addLayout(send_row)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.settings)
        split.addWidget(terminal_page)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([640, 640])

        self.start_button = QPushButton("Run")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop job")
        self.stop_button.clicked.connect(self.stop)
        self.status = QLabel("Ready")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Action"))
        controls.addWidget(self.action, stretch=1)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.file_actions)
        layout.addLayout(controls)
        layout.addWidget(self.arguments)
        layout.addWidget(split, stretch=1)
        layout.addWidget(self.status)
        self.set_busy(False)

    def change_action(self):
        action = self.action.currentData()
        index = {"setup": 1, "demo": 2, "validate": 3, "test": 3}.get(action, 0)
        self.arguments.setCurrentIndex(index)
        self.arguments.setVisible(index != 0)
        section = {"render": "render", "resume": "render", "demo": "render",
                   "audit": "split", "fit": "training"}.get(action)
        if section is not None:
            self.settings.sections.setCurrentIndex(self.settings.sections.findData(section))

    def set_busy(self, busy):
        for widget in (self.file_actions, self.settings, self.action, self.arguments, self.start_button):
            widget.setEnabled(not busy)
        for widget in (self.stop_button, self.terminal_input, self.send_button):
            widget.setEnabled(busy)

    def open_job(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open workstation job", str(self.path.parent), "Job files (*.json)")
        if not path:
            return
        try:
            config = validate(read_json(path))
            self.settings.load(config)
            self.components.set_values([*config["training"]["models"], "reporting"])
            self.path = Path(path).resolve()
            self.path_label.setText(str(self.path))
            self.status.setText("Job opened")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def write_job(self, path):
        config = self.settings.values()
        if not path.is_file() or read_json(path) != config:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, config)

    def save_job(self):
        try:
            self.write_job(self.path)
            self.status.setText("Job saved")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save workstation job", str(self.path), "Job files (*.json)")
        if not path:
            return
        try:
            target = Path(path).resolve()
            self.write_job(target)
            self.path = target
            self.path_label.setText(str(target))
            self.status.setText("Job saved")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def command(self, action):
        args = ["-B", "-u", str(Path(__file__).with_name("annalyzer_master.py")), action, "--config", str(self.path)]
        if action == "setup":
            components = self.components.values()
            if not components:
                raise ValueError("Select at least one setup component.")
            args.extend(("--components", *components))
        elif action == "demo":
            args.extend(("--light", self.demo_light.currentData(), "--kind", self.demo_kind.currentData()))
        elif action in ("validate", "test"):
            run = self.saved_run.text()
            if not run or not (rooted(run) / "complete.json").is_file():
                raise ValueError("Choose a completed V5 run folder.")
            args.extend(("--model", self.saved_model.currentData(), "--run", str(rooted(run))))
        return args

    def start(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        action = self.action.currentData()
        try:
            args = self.command(action)
            if action not in ("validate", "test"):
                self.settings.values()
            if action in CONFIRM and QMessageBox.question(
                self, "Start job", f"{ACTION_LABELS[action]}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
            if action not in ("validate", "test"):
                self.write_job(self.path)
            destination = log_file(f"ui_{action}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.log = destination.open("x", encoding="utf-8")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.cancelled = False
        self.set_busy(True)
        self.status.setText(f"Running {ACTION_LABELS[action]}")
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        self.append_terminal(f"\n> {subprocess.list2cmdline([str(python), *args])}\nLog {destination}\n")
        self.process.start(str(python), args)

    def append_terminal(self, text):
        if not text:
            return
        if self.log is not None:
            self.log.write(text)
            self.log.flush()
        scroll = self.terminal.verticalScrollBar()
        follow = scroll.value() == scroll.maximum()
        cursor = QTextCursor(self.terminal.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        if follow:
            scroll.setValue(scroll.maximum())

    def read_output(self):
        output = bytes(self.process.readAllStandardOutput())
        if self.decoder is not None:
            self.append_terminal(self.decoder.decode(output))

    def send_input(self):
        if self.process.state() == QProcess.ProcessState.Running:
            self.process.write((self.terminal_input.text() + "\n").encode("utf-8"))
            self.terminal_input.clear()

    def end(self, status):
        self.read_output()
        if self.decoder is not None:
            self.append_terminal(self.decoder.decode(b"", final=True))
        self.append_terminal(f"\n{status}\n")
        if self.log is not None:
            self.log.close()
            self.log = None
        self.decoder = None
        self.status.setText(status)
        self.set_busy(False)

    def finished(self, code, status):
        if self.cancelled:
            message = "Job stopped"
        elif status == QProcess.ExitStatus.NormalExit and not code:
            message = "Job completed"
        else:
            message = f"Job failed (exit code {code})"
        self.end(message)

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.end(f"Could not start the job: {self.process.errorString()}")

    def stop(self):
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return True
        if QMessageBox.question(self, "Stop job", "Stop this job and its child processes?") != QMessageBox.StandardButton.Yes:
            return False
        pid = self.process.processId()
        if not pid:
            self.status.setText("Wait for the job to start, then try again.")
            return False
        try:
            if os.name == "nt":
                result = subprocess.run(["taskkill.exe", "/PID", str(pid), "/T", "/F"], capture_output=True,
                                        timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode and self.process.state() == QProcess.ProcessState.Running:
                    self.status.setText("Windows could not stop the job.")
                    return False
            else:
                self.process.kill()
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.status.setText(str(exc))
            return False
        self.cancelled = True
        self.status.setText("Stopping job…")
        return True

    def closeEvent(self, event):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            if not self.stop() or not self.process.waitForFinished(2000):
                event.ignore()
                return
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LOCAL / "jobs/default.json")
    args = parser.parse_args()
    app = QApplication([sys.argv[0]])
    try:
        window = Workstation(rooted(args.config))
    except (OSError, ValueError) as exc:
        QMessageBox.critical(None, "Could not open the job", str(exc))
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
