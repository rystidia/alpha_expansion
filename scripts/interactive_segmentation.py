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
    )
    from PyQt6.QtGui import QImage, QPixmap
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


class InteractiveSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alpha-Expansion Interactive Segmentation")
        self.resize(1000, 700)

        self.image_path = None
        self.original_image = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        toolbar_layout = QVBoxLayout()

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.btn_load)

        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout, 1)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumSize(600, 600)

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
        self.scene.clear()

        data = np.array(pil_image)
        height, width, channel = data.shape
        bytes_per_line = 3 * width

        q_img = QImage(
            data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)

        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, width, height)


if __name__ == "__main__":
    app = QApplication(sys.path)
    window = InteractiveSegmentationApp()
    window.show()
    sys.exit(app.exec())
