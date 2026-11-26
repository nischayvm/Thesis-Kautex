"""Draw the model's input geometry on the current tray image."""

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel,
                             QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from worker import FRAME, ORIGINS, OVERLAP, STRIDE, TILE, frame_layout


GRIDS = {
    "patchcore": (("Layer 2 feature grid (64 × 64)", 8), ("Layer 3 feature grid (32 × 32)", 16)),
    "anomalydino": (("ViT patch grid (37 × 37)", 14),),
    "unet": (("Encoder bottleneck grid (16 × 16)", 32),),
}

STAGES = {
    "patchcore": ("WideResNet50-2 extracts features at layers 2 and 3. Their aligned 64 × 64 grid is compared "
                  "with the fitted normal feature bank to generate an anomaly map."),
    "anomalydino": ("The tile gets a 3 px reflection border before DINOv2 splits it into 14 × 14 px patches. "
                    "Cosine distances to the normal bank form the anomaly map. The border is removed before stitching."),
    "unet": ("A ResNet18 encoder shrinks each tile from 512 × 512 to a 16 × 16 bottleneck. "
             "The decoder uses skip connections to recover one defect probability per input pixel."),
}


class Preview(QLabel):
    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.placeholder = placeholder
        self.image = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(220, 220)

    def show_image(self, image):
        self.image = image
        self.refresh()

    def refresh(self):
        if self.image.isNull():
            self.setText(self.placeholder)
        else:
            self.setPixmap(self.image.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()


class PipelineView(QWidget):
    def __init__(self):
        super().__init__()
        self.canvas = QImage()
        self.model = None
        self.tile = QSpinBox()
        self.tile.setRange(1, len(ORIGINS))
        self.grid = QComboBox()
        self.grid.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.tile.valueChanged.connect(self.draw)
        self.grid.currentIndexChanged.connect(self.draw)
        self.position = QLabel()
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.tray_preview = Preview("Tile positions")
        self.tile_preview = Preview("Selected tile")

        options = QHBoxLayout()
        for item in (QLabel("Tile"), self.tile, self.position, QLabel("Show"), self.grid):
            options.addWidget(item)
        options.addStretch()
        previews = QGridLayout()
        for column, (label, preview) in enumerate((("Overlapping tiles", self.tray_preview),
                                                   ("Selected tile", self.tile_preview))):
            previews.addWidget(QLabel(label), 0, column, Qt.AlignmentFlag.AlignCenter)
            previews.addWidget(preview, 1, column)
            previews.setColumnStretch(column, 1)
        previews.setRowStretch(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{len(ORIGINS)} tiles · {TILE} × {TILE} px · {OVERLAP} px overlap · {STRIDE} px stride"))
        layout.addLayout(options)
        layout.addLayout(previews, stretch=1)
        layout.addWidget(self.note)

    def show_pipeline(self, pixmap, model):
        source = pixmap.toImage()
        x, y, w, h = frame_layout(source.width(), source.height())
        self.canvas = QImage(FRAME, FRAME, QImage.Format.Format_RGB32)
        self.canvas.fill(QColor("#606060"))
        with QPainter(self.canvas) as painter:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(QRect(x, y, w, h), source)
        self.has_padding = (w, h) != (FRAME, FRAME)
        if self.model != model:
            self.model = model
            self.grid.blockSignals(True)
            self.grid.clear()
            for text, step in GRIDS[model]:
                self.grid.addItem(text, step)
            self.grid.addItem("Individual pixels (16 × 16 sample)", 0)
            self.grid.blockSignals(False)
        self.draw()

    def draw(self):
        if self.canvas.isNull() or self.model is None:
            return
        x, y = ORIGINS[self.tile.value() - 1]
        self.position.setText(f"of {len(ORIGINS)}  ·  x={x}, y={y}")
        tray = self.canvas.copy()
        with QPainter(tray) as painter:
            painter.setPen(QPen(QColor(Qt.GlobalColor.white), 4))
            for left, top in ORIGINS:
                painter.drawRect(left, top, TILE - 1, TILE - 1)
            painter.setPen(QPen(QColor("#ffdc3c"), 12))
            painter.drawRect(x + 6, y + 6, TILE - 12, TILE - 12)
        self.tray_preview.show_image(QPixmap.fromImage(tray))

        tile = self.canvas.copy(x, y, TILE, TILE)
        step = self.grid.currentData()
        tile_note = ""
        pen_width = 1
        if step == 0:
            tile = tile.copy(TILE // 2 - 8, TILE // 2 - 8, 16, 16).scaled(TILE, TILE)
            step, pen_width = TILE // 16, 2
            tile_note = "The pixel view magnifies a 16 × 16 sample from the center of the selected tile. "
        elif self.model == "anomalydino":
            padded = QImage(TILE + 6, TILE + 6, QImage.Format.Format_RGB32)
            padded.fill(QColor("#606060"))
            with QPainter(padded) as painter:
                painter.drawImage(3, 3, tile)
            tile = padded
        with QPainter(tile) as painter:
            painter.setPen(QPen(QColor(255, 255, 255, 130), pen_width))
            edge = tile.width() - 1
            for i in range(step, edge, step):
                painter.drawLine(i, 0, i, edge)
                painter.drawLine(0, i, edge, i)
        self.tile_preview.show_image(QPixmap.fromImage(tile))
        padding = ""
        if self.has_padding:
            padding = "Gray margins mark model input padding. "
        self.note.setText(f"{padding}{tile_note}{STAGES[self.model]} Gaussian stitching blends overlapping maps. "
                          "Grid lines show spatial layout.")
