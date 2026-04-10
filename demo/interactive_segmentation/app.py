import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGraphicsView,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _ResizableView(QGraphicsView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build")))
import alpha_expansion_py as ae

from .community import CommunityDetectionProblem
from .problem import Problem
from .segmentation import ImageSegmentationProblem


class ExpansionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alpha-Expansion Demo")
        self.resize(1100, 750)
        self._problem: Problem | None = None
        self._model = None
        self._optimizer = None
        self._alpha = 0
        self._moves_without_change = 0
        self._setup_ui()
        self._switch_problem(ImageSegmentationProblem())

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        sidebar = QVBoxLayout()

        mode_row = QHBoxLayout()
        self._btn_seg = QPushButton("Segmentation")
        self._btn_seg.setCheckable(True)
        self._btn_seg.clicked.connect(lambda: self._switch_problem(ImageSegmentationProblem()))
        self._btn_comm = QPushButton("Community Detection")
        self._btn_comm.setCheckable(True)
        self._btn_comm.clicked.connect(lambda: self._switch_problem(CommunityDetectionProblem()))
        mode_row.addWidget(self._btn_seg)
        mode_row.addWidget(self._btn_comm)
        sidebar.addLayout(mode_row)

        self._param_container = QWidget()
        self._param_layout = QVBoxLayout(self._param_container)
        self._param_layout.setContentsMargins(0, 0, 0, 0)
        sidebar.addWidget(self._param_container)

        sidebar.addStretch()

        self._btn_init = QPushButton("Initialize")
        self._btn_init.clicked.connect(self._initialize)
        sidebar.addWidget(self._btn_init)

        self._btn_step = QPushButton("Step (1 Alpha Move)")
        self._btn_step.clicked.connect(self._step)
        self._btn_step.setEnabled(False)
        sidebar.addWidget(self._btn_step)

        self._btn_run = QPushButton("Run to Convergence")
        self._btn_run.clicked.connect(self._run)
        self._btn_run.setEnabled(False)
        sidebar.addWidget(self._btn_run)

        self._view = _ResizableView()
        self._view.setMinimumSize(700, 600)
        outer.addLayout(sidebar, 1)
        outer.addWidget(self._view, 4)

    def _switch_problem(self, problem: Problem):
        while self._param_layout.count():
            item = self._param_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self._problem = problem
        self._model = None
        self._optimizer = None
        self._alpha = 0
        self._moves_without_change = 0
        self._param_layout.addWidget(problem.get_param_widget())
        self._view.setScene(problem.get_scene())
        problem.get_scene().sceneRectChanged.connect(
            lambda rect: self._view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        )
        self._btn_seg.setChecked(isinstance(problem, ImageSegmentationProblem))
        self._btn_comm.setChecked(isinstance(problem, CommunityDetectionProblem))
        self._btn_init.setEnabled(True)
        self._btn_step.setEnabled(False)
        self._btn_run.setEnabled(False)
        if isinstance(problem, CommunityDetectionProblem):
            problem._on_graph_selected = self._initialize_and_render

    def _initialize_and_render(self):
        self._model, self._optimizer = self._problem.build_model()
        self._alpha = 0
        self._moves_without_change = 0
        self._btn_init.setEnabled(False)
        self._btn_step.setEnabled(True)
        self._btn_run.setEnabled(True)
        self._update_canvas()

    def _initialize(self):
        try:
            self._model, self._optimizer = self._problem.build_model()
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))
            return
        self._alpha = 0
        self._moves_without_change = 0
        self._btn_init.setEnabled(False)
        self._btn_step.setEnabled(True)
        self._btn_run.setEnabled(True)

    def _step(self):
        changed = self._optimizer.perform_expansion_move(self._alpha)
        if changed:
            self._moves_without_change = 0
        else:
            self._moves_without_change += 1
        self._update_canvas()
        if self._moves_without_change >= self._problem.num_labels():
            QMessageBox.information(
                self, "Converged",
                f"Algorithm converged!\nFinal Energy: {self._model.evaluate_total_energy()}",
            )
            self._btn_step.setEnabled(False)
            self._btn_run.setEnabled(False)
            return
        self._alpha = (self._alpha + 1) % self._problem.num_labels()

    def _run(self):
        strategy = ae.SequentialStrategyInt(20)
        strategy.execute(self._optimizer, self._model)
        self._update_canvas()
        self._btn_step.setEnabled(False)
        self._btn_run.setEnabled(False)
        QMessageBox.information(
            self, "Done",
            f"Optimization finished.\nFinal Energy: {self._model.evaluate_total_energy()}",
        )

    def _update_canvas(self):
        pixmap = self._problem.render(self._model)
        scene = self._problem.get_scene()
        scene.clear()
        scene.addPixmap(pixmap)
        scene.setSceneRect(pixmap.rect().toRectF())
