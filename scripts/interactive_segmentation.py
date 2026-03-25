import sys
import numpy as np
from PIL import Image

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QFileDialog,
        QGraphicsScene,
        QGraphicsView,
        QLabel,
        QMessageBox,
    )
    from PyQt6.QtGui import QImage, QPixmap, QColor, QPen, QPainterPath, QBrush, QIcon
    from PyQt6.QtCore import Qt, pyqtSignal
except ImportError:
    print("=========================================================")
    print("Error: The interactive segmentation GUI requires PyQt6.")
    print("To run this demo, please install it via: pip install PyQt6")
    print("=========================================================")
    sys.exit(1)

sys.path.append("build")
try:
    import alpha_expansion_py as ae
except ImportError:
    print("Error: Could not import alpha_expansion_py. Did you compile the project?")
    sys.exit(1)


class DrawableScene(QGraphicsScene):
    scribble_drawn = pyqtSignal(int, int, int)

    def __init__(self):
        super().__init__()
        self.current_label = 0
        self.colors = []
        self.pen_width = 10
        self.can_draw = False

    def mousePressEvent(self, event):
        if not self.can_draw:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._draw_point(event.scenePos().x(), event.scenePos().y())

    def mouseMoveEvent(self, event):
        if not self.can_draw:
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._draw_point(event.scenePos().x(), event.scenePos().y())

    def _draw_point(self, x, y):
        r = self.pen_width // 2
        self.addRect(
            x - r,
            y - r,
            2 * r + 1,
            2 * r + 1,
            QPen(Qt.PenStyle.NoPen),
            QBrush(self.colors[self.current_label]),
        )
        self.scribble_drawn.emit(int(x), int(y), self.current_label)


class InteractiveSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alpha-Expansion Interactive Segmentation")
        self.resize(1000, 700)

        self.image_path = None
        self.original_image = None
        self.scribble_mask = None

        self.preset_colors = [
            QColor(255, 0, 0),
            QColor(0, 255, 0),
            QColor(0, 0, 255),
            QColor(255, 255, 0),
            QColor(0, 255, 255),
            QColor(255, 0, 255),
            QColor(255, 128, 0),
            QColor(128, 0, 255),
            QColor(0, 128, 255),
        ]

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        toolbar_layout = QVBoxLayout()

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.btn_load)

        self.btn_add_label = QPushButton("+ Add Label")
        self.btn_add_label.clicked.connect(self.add_label_ui)
        toolbar_layout.addWidget(self.btn_add_label)

        self.labels_layout = QVBoxLayout()
        toolbar_layout.addLayout(self.labels_layout)

        self.scene = DrawableScene()
        self.scene.scribble_drawn.connect(self.record_scribble)
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumSize(600, 600)

        # Pre-fill with two labels to start
        self.add_label_color()
        self.add_label_color()
        self.rebuild_labels_ui()

        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout, 1)

        main_layout.addWidget(self.view, 4)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.ppm *.pgm)"
        )
        if file_name:
            self.image_path = file_name
            self.original_image = Image.open(file_name).convert("RGB")
            self.display_image(self.original_image)

    def display_image(self, pil_image):
        self.scene.can_draw = True
        self.scribble_mask = np.full(
            (pil_image.height, pil_image.width), -1, dtype=np.int32
        )
        self.redraw_canvas()

    def redraw_canvas(self):
        if self.original_image is None:
            return

        self.scene.clear()

        data = np.array(self.original_image)
        height, width, channel = data.shape
        bytes_per_line = 3 * width

        q_img = QImage(
            data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)

        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, width, height)

        if self.scribble_mask is not None:
            overlay_data = np.zeros((height, width, 4), dtype=np.uint8)
            for i, color in enumerate(self.scene.colors):
                overlay_data[self.scribble_mask == i] = [
                    color.red(),
                    color.green(),
                    color.blue(),
                    255,
                ]

            q_overlay = QImage(
                overlay_data.data,
                width,
                height,
                4 * width,
                QImage.Format.Format_RGBA8888,
            )
            pixmap_overlay = QPixmap.fromImage(q_overlay)
            self.scene.addPixmap(pixmap_overlay)

    def add_label_color(self):
        if len(self.scene.colors) >= len(self.preset_colors):
            return

        used_colors = set(c.name() for c in self.scene.colors)
        for color in self.preset_colors:
            if color.name() not in used_colors:
                self.scene.colors.append(QColor(color))
                return

    def add_label_ui(self):
        if len(self.scene.colors) >= len(self.preset_colors):
            QMessageBox.warning(
                self, "Warning", f"Maximum {len(self.preset_colors)} labels reached!"
            )
            return

        self.add_label_color()
        self.rebuild_labels_ui()
        self.set_label(len(self.scene.colors) - 1)

    def remove_label(self, label_idx):
        if len(self.scene.colors) <= 2:
            QMessageBox.warning(
                self, "Warning", "You need at least two labels for alpha-expansion!"
            )
            return

        del self.scene.colors[label_idx]

        mask = self.scribble_mask
        if mask is not None:
            mask[mask == label_idx] = -1
            mask[mask > label_idx] -= 1

        if self.scene.current_label == label_idx:
            self.set_label(max(0, label_idx - 1))
        elif self.scene.current_label > label_idx:
            self.set_label(self.scene.current_label - 1)

        self.rebuild_labels_ui()
        self.redraw_canvas()

    def rebuild_labels_ui(self):
        # Clear layout
        while self.labels_layout.count():
            item = self.labels_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, color in enumerate(self.scene.colors):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            pixmap = QPixmap(16, 16)
            pixmap.fill(color)

            btn_select = QPushButton(f" Label {i}")
            btn_select.setIcon(QIcon(pixmap))
            btn_select.setCheckable(True)
            btn_select.setChecked(i == self.scene.current_label)

            if i == self.scene.current_label:
                btn_select.setStyleSheet("text-align: left; font-weight: bold;")
            else:
                btn_select.setStyleSheet("text-align: left;")

            btn_select.clicked.connect(lambda checked, idx=i: self.set_label(idx))
            row_layout.addWidget(btn_select)

            btn_del = QPushButton("X")
            btn_del.setFixedWidth(30)
            btn_del.clicked.connect(lambda checked, idx=i: self.remove_label(idx))
            row_layout.addWidget(btn_del)

            self.labels_layout.addWidget(row_widget)

    def set_label(self, label_idx):
        if label_idx < len(self.scene.colors):
            self.scene.current_label = label_idx
            self.rebuild_labels_ui()

    def record_scribble(self, x, y, label_idx):
        mask = self.scribble_mask
        if mask is None:
            return
        h, w = mask.shape
        r = self.scene.pen_width // 2

        x_min, x_max = max(0, x - r), min(w, x + r + 1)
        y_min, y_max = max(0, y - r), min(h, y + r + 1)
        mask[y_min:y_max, x_min:x_max] = label_idx


if __name__ == "__main__":
    app = QApplication(sys.path)
    window = InteractiveSegmentationApp()
    window.show()
    sys.exit(app.exec())
