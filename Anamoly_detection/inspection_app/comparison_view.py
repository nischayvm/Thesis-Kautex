"""Side-by-side image views with a shared zoom and pan position."""

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)


class ImagePane(QWidget):
    def __init__(self, comparison, placeholder):
        super().__init__(comparison)
        self.comparison = comparison
        self.placeholder = placeholder
        self.image = QPixmap()
        self.drag_position = None
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def show_image(self, image):
        self.image = image
        self.update_cursor()
        self.update()

    def update_cursor(self):
        if self.drag_position is None:
            movable = self.comparison.zoom > 1 and not self.image.isNull()
            self.setCursor(Qt.CursorShape.OpenHandCursor if movable else Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        with QPainter(self) as painter:
            if self.image.isNull():
                painter.setPen(self.palette().windowText().color())
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            else:
                viewport, target = self.comparison.rectangles(self)
                painter.setClipRect(viewport)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                painter.drawPixmap(target, self.image, QRectF(self.image.rect()))

    def wheelEvent(self, event):
        if self.image.isNull():
            return super().wheelEvent(event)
        steps = event.angleDelta().y() / 120 or event.pixelDelta().y() / 40
        if steps:
            _, rect = self.comparison.rectangles(self)
            position = event.position() - rect.topLeft()
            anchor = QPointF(position.x() / rect.width(), position.y() / rect.height())
            self.comparison.zoom_by(1.25 ** steps, anchor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.comparison.zoom > 1 and not self.image.isNull():
            self.drag_position = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_position is None:
            return super().mouseMoveEvent(event)
        position = event.position()
        self.comparison.pan_by(position - self.drag_position)
        self.drag_position = position
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drag_position is not None:
            self.drag_position = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ImageComparison(QWidget):
    MAX_ZOOM = 12.0

    def __init__(self):
        super().__init__()
        self.zoom = 1.0
        self.center = QPointF(0.5, 0.5)
        self.predictions = {}
        self.original = ImagePane(self, "No image")
        self.prediction = ImagePane(self, "No prediction")
        self.ground_truth = ImagePane(self, "No mask")
        self.panes = (self.original, self.prediction, self.ground_truth)

        self.prediction_choice = QComboBox()
        self.prediction_choice.addItems(["Overlay", "Heatmap"])
        self.prediction_choice.currentIndexChanged.connect(self.show_prediction_choice)
        self.prediction_choice.setEnabled(False)
        self.scale_label = QLabel()
        prediction_header = QHBoxLayout()
        prediction_header.addStretch()
        prediction_header.addWidget(QLabel("Prediction"))
        prediction_header.addWidget(self.prediction_choice)
        prediction_header.addStretch()

        self.zoom_out = QPushButton("Zoom out")
        self.zoom_out.clicked.connect(lambda: self.zoom_by(1 / 1.25))
        self.zoom_in = QPushButton("Zoom in")
        self.zoom_in.clicked.connect(lambda: self.zoom_by(1.25))
        self.fit = QPushButton("Fit")
        self.fit.clicked.connect(self.reset_zoom)
        self.zoom_label = QLabel()
        zoom_controls = QHBoxLayout()
        zoom_controls.addWidget(QLabel("Scroll to zoom, drag to pan."))
        zoom_controls.addStretch()
        for widget in (self.zoom_out, self.zoom_label, self.zoom_in, self.fit):
            zoom_controls.addWidget(widget)

        previews = QGridLayout()
        previews.addWidget(QLabel("Tray image"), 0, 0, Qt.AlignmentFlag.AlignCenter)
        previews.addLayout(prediction_header, 0, 1)
        previews.addWidget(QLabel("Ground-truth overlay"), 0, 2, Qt.AlignmentFlag.AlignCenter)
        for column, pane in enumerate(self.panes):
            previews.addWidget(pane, 1, column)
            previews.setColumnStretch(column, 1)
        previews.addWidget(self.scale_label, 2, 1, Qt.AlignmentFlag.AlignCenter)
        previews.setRowStretch(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(zoom_controls)
        layout.addLayout(previews, stretch=1)
        self.update_view()

    def set_tray(self, image):
        self.original.show_image(image)
        self.reset_zoom()

    def set_prediction(self, overlay, heatmap, scale):
        self.predictions = {"Overlay": overlay, "Heatmap": heatmap}
        self.scale_label.setText(f"Color scale: 0 to {scale:.6g}")
        self.prediction_choice.setEnabled(True)
        self.show_prediction_choice()

    def clear_prediction(self):
        self.predictions.clear()
        self.prediction_choice.setEnabled(False)
        self.show_prediction_choice()

    def show_prediction_choice(self):
        choice = self.prediction_choice.currentText()
        self.prediction.show_image(self.predictions.get(choice, QPixmap()))
        self.scale_label.setVisible(bool(self.predictions))

    def dimensions(self):
        width = max(1, min(pane.width() for pane in self.panes))
        height = max(1, min(pane.height() for pane in self.panes))
        image = self.original.image
        scale = self.zoom * min(width / image.width(), height / image.height())
        return width, height, scale * image.width(), scale * image.height()

    def visible_center(self, width, height, image_width, image_height):
        x = min(0.5, width / (2 * image_width))
        y = min(0.5, height / (2 * image_height))
        return QPointF(max(x, min(1 - x, self.center.x())), max(y, min(1 - y, self.center.y())))

    def rectangles(self, pane):
        width, height, image_width, image_height = self.dimensions()
        midpoint = QPointF(pane.width() / 2, pane.height() / 2)
        center = self.visible_center(width, height, image_width, image_height)
        viewport = QRectF(midpoint.x() - width / 2, midpoint.y() - height / 2, width, height)
        target = QRectF(midpoint.x() - center.x() * image_width, midpoint.y() - center.y() * image_height,
                        image_width, image_height)
        return viewport, target

    def reset_zoom(self):
        self.zoom = 1.0
        self.center = QPointF(0.5, 0.5)
        self.update_view()

    def zoom_by(self, factor, anchor=None):
        if self.original.image.isNull():
            return
        zoom = max(1.0, min(self.MAX_ZOOM, self.zoom * factor))
        center = self.visible_center(*self.dimensions())
        if anchor is not None:
            center = anchor + (center - anchor) * (self.zoom / zoom)
        self.center = center
        self.zoom = zoom
        self.center = self.visible_center(*self.dimensions())
        self.update_view()

    def pan_by(self, delta):
        width, height, image_width, image_height = self.dimensions()
        self.center = self.visible_center(width, height, image_width, image_height) - QPointF(
            delta.x() / image_width, delta.y() / image_height)
        self.center = self.visible_center(width, height, image_width, image_height)
        self.update_view()

    def update_view(self):
        has_image = not self.original.image.isNull()
        self.zoom_label.setText(f"{self.zoom:.0%}")
        self.zoom_out.setEnabled(has_image and self.zoom > 1)
        self.fit.setEnabled(has_image and self.zoom > 1)
        self.zoom_in.setEnabled(has_image and self.zoom < self.MAX_ZOOM)
        for pane in self.panes:
            pane.update_cursor()
            pane.update()
