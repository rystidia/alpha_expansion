import os
import sys

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGraphicsScene,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build")))
import alpha_expansion_py as ae

from .problem import Problem

_PRESET_COLORS = [
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


class DrawableScene(QGraphicsScene):
    scribble_drawn = pyqtSignal(int, int, int)

    def __init__(self):
        super().__init__()
        self.current_label = 0
        self.colors = []
        self.pen_width = 10
        self.can_draw = False
        self.ae_started = False

    def mousePressEvent(self, event):
        if not self.can_draw or self.ae_started:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._draw_point(event.scenePos().x(), event.scenePos().y())

    def mouseMoveEvent(self, event):
        if not self.can_draw or self.ae_started:
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._draw_point(event.scenePos().x(), event.scenePos().y())

    def _draw_point(self, x, y):
        r = self.pen_width // 2
        self.addRect(
            x - r, y - r, 2 * r + 1, 2 * r + 1,
            QPen(Qt.PenStyle.NoPen),
            QBrush(self.colors[self.current_label]),
        )
        self.scribble_drawn.emit(int(x), int(y), self.current_label)


class ImageSegmentationProblem(Problem):
    def __init__(self):
        self._original_image = None
        self._scribble_mask = None
        self._scene = DrawableScene()
        self._scene.colors = [QColor(c) for c in _PRESET_COLORS[:2]]
        self._scene.scribble_drawn.connect(self._record_scribble)
        self._labels_layout = None
        self._param_widget = self._build_param_widget()
        self._rebuild_labels_ui()

    def get_scene(self) -> QGraphicsScene:
        return self._scene

    def get_param_widget(self) -> QWidget:
        return self._param_widget

    def _build_param_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self._load_image)
        layout.addWidget(btn_load)
        btn_add = QPushButton("+ Add Label")
        btn_add.clicked.connect(self._add_label_ui)
        layout.addWidget(btn_add)
        self._labels_layout = QVBoxLayout()
        layout.addLayout(self._labels_layout)
        return widget

    def _load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Image File", "", "Images (*.png *.jpg *.bmp *.ppm *.pgm)"
        )
        if not file_name:
            return
        self._original_image = Image.open(file_name).convert("RGB")
        self._scene.can_draw = True
        self._scene.ae_started = False
        self._scribble_mask = np.full(
            (self._original_image.height, self._original_image.width), -1, dtype=np.int32
        )
        self._redraw_canvas()

    def _redraw_canvas(self):
        if self._original_image is None:
            return
        self._scene.clear()
        data = np.array(self._original_image)
        h, w, _ = data.shape
        q_img = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._scene.addPixmap(QPixmap.fromImage(q_img))
        self._scene.setSceneRect(0, 0, w, h)
        if self._scribble_mask is not None:
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            for i, color in enumerate(self._scene.colors):
                overlay[self._scribble_mask == i] = [
                    color.red(), color.green(), color.blue(), 255,
                ]
            q_overlay = QImage(overlay.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
            self._scene.addPixmap(QPixmap.fromImage(q_overlay))

    def _add_label_color(self):
        if len(self._scene.colors) >= len(_PRESET_COLORS):
            return
        used = {c.name() for c in self._scene.colors}
        for color in _PRESET_COLORS:
            if color.name() not in used:
                self._scene.colors.append(QColor(color))
                return

    def _add_label_ui(self):
        if self._scene.ae_started:
            QMessageBox.warning(None, "Warning", "Cannot add labels after algorithm has started.")
            return
        if len(self._scene.colors) >= len(_PRESET_COLORS):
            QMessageBox.warning(None, "Warning", f"Maximum {len(_PRESET_COLORS)} labels reached!")
            return
        self._add_label_color()
        self._rebuild_labels_ui()
        self._set_label(len(self._scene.colors) - 1)

    def _remove_label(self, label_idx):
        if self._scene.ae_started:
            QMessageBox.warning(None, "Warning", "Cannot remove labels after algorithm has started.")
            return
        if len(self._scene.colors) <= 2:
            QMessageBox.warning(None, "Warning", "You need at least two labels for alpha-expansion!")
            return
        del self._scene.colors[label_idx]
        if self._scribble_mask is not None:
            self._scribble_mask[self._scribble_mask == label_idx] = -1
            self._scribble_mask[self._scribble_mask > label_idx] -= 1
        if self._scene.current_label == label_idx:
            self._set_label(max(0, label_idx - 1))
        elif self._scene.current_label > label_idx:
            self._set_label(self._scene.current_label - 1)
        self._rebuild_labels_ui()
        self._redraw_canvas()

    def _rebuild_labels_ui(self):
        while self._labels_layout.count():
            item = self._labels_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, color in enumerate(self._scene.colors):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            btn_select = QPushButton(f" Label {i}")
            btn_select.setIcon(QIcon(pixmap))
            btn_select.setCheckable(True)
            btn_select.setChecked(i == self._scene.current_label)
            style = "text-align: left; font-weight: bold;" if i == self._scene.current_label else "text-align: left;"
            btn_select.setStyleSheet(style)
            btn_select.clicked.connect(lambda checked, idx=i: self._set_label(idx))
            row_layout.addWidget(btn_select)
            btn_del = QPushButton("X")
            btn_del.setFixedWidth(30)
            btn_del.clicked.connect(lambda checked, idx=i: self._remove_label(idx))
            row_layout.addWidget(btn_del)
            self._labels_layout.addWidget(row)

    def _set_label(self, label_idx):
        if label_idx < len(self._scene.colors):
            self._scene.current_label = label_idx
            self._rebuild_labels_ui()

    def _record_scribble(self, x, y, label_idx):
        if self._scribble_mask is None:
            return
        h, w = self._scribble_mask.shape
        r = self._scene.pen_width // 2
        x_min, x_max = max(0, x - r), min(w, x + r + 1)
        y_min, y_max = max(0, y - r), min(h, y + r + 1)
        self._scribble_mask[y_min:y_max, x_min:x_max] = label_idx

    def _compute_grid_edge_weights(self, data, w, h, lambda_smooth=50.0):
        data_f = data.astype(np.float64)
        sigma_sq = np.var(data_f)
        node_ids = np.arange(h * w).reshape(h, w)
        h_n1 = node_ids[:, :-1].flatten()
        h_n2 = node_ids[:, 1:].flatten()
        h_diff_sq = np.sum((data_f[:, :-1] - data_f[:, 1:]) ** 2, axis=2).flatten()
        h_weights = (lambda_smooth * np.exp(-h_diff_sq / (2 * sigma_sq))).astype(np.int32)
        v_n1 = node_ids[:-1, :].flatten()
        v_n2 = node_ids[1:, :].flatten()
        v_diff_sq = np.sum((data_f[:-1, :] - data_f[1:, :]) ** 2, axis=2).flatten()
        v_weights = (lambda_smooth * np.exp(-v_diff_sq / (2 * sigma_sq))).astype(np.int32)
        n1s = np.concatenate([h_n1, v_n1]).tolist()
        n2s = np.concatenate([h_n2, v_n2]).tolist()
        weights = np.concatenate([h_weights, v_weights]).tolist()
        return n1s, n2s, weights

    def build_model(self):
        if self._original_image is None:
            raise ValueError("Please load an image first.")
        data = np.array(self._original_image, dtype=np.float32)
        h, w, _ = data.shape
        num_labels = len(self._scene.colors)
        means, cov_invs = [], []
        for i in range(num_labels):
            mask = self._scribble_mask == i
            if not np.any(mask):
                raise ValueError(f"Label {i} has no scribbles! Please scribble all labels.")
            pixels = data[mask].reshape(-1, 3).astype(np.float64)
            mean = np.mean(pixels, axis=0)
            cov = np.cov(pixels.T) + np.eye(3) * 1e-6
            means.append(mean)
            cov_invs.append(np.linalg.inv(cov))
        flat_pixels = data.reshape(-1, 3).astype(np.float64)
        unary_costs = np.zeros((h, w, num_labels), dtype=np.float64)
        for i in range(num_labels):
            diff = flat_pixels - means[i]
            mahal = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_invs[i], diff))
            unary_costs[:, :, i] = mahal.reshape(h, w)
        MAX_COST = 2000
        for i in range(num_labels):
            unary_costs[:, :, i][self._scribble_mask == i] = 0.0
            for j in range(num_labels):
                if i != j:
                    unary_costs[:, :, i][self._scribble_mask == j] = MAX_COST
        model = ae.EnergyModel(h * w, num_labels)
        model.set_unary_costs(unary_costs.flatten().astype(np.int32).tolist())
        n1s, n2s, weights = self._compute_grid_edge_weights(data, w, h)
        model.add_grid_edges(w, h)
        model.set_edge_weights(n1s, n2s, weights)
        self._scene.ae_started = True
        optimizer = ae.AlphaExpansionInt(model)
        return model, optimizer

    def render(self, model) -> QPixmap:
        labels = model.get_labels()
        h, w = self._original_image.height, self._original_image.width
        labels_img = np.array(labels, dtype=np.int32).reshape((h, w))
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for i, color in enumerate(self._scene.colors):
            colored[labels_img == i] = [color.red(), color.green(), color.blue()]
        q_img = QImage(colored.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        base_pixmap = QPixmap.fromImage(q_img)
        if self._scribble_mask is not None:
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            for i, c in enumerate(self._scene.colors):
                overlay[self._scribble_mask == i] = [c.red(), c.green(), c.blue(), 255]
            q_overlay = QImage(overlay.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
            painter = QPainter(base_pixmap)
            painter.drawPixmap(0, 0, QPixmap.fromImage(q_overlay))
            painter.end()
        return base_pixmap

    def num_labels(self) -> int:
        return len(self._scene.colors)
