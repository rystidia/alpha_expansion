from abc import ABC, abstractmethod

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsScene, QWidget


class Problem(ABC):
    @abstractmethod
    def get_param_widget(self) -> QWidget:
        """Return the sidebar control widget for this problem."""

    @abstractmethod
    def get_scene(self) -> QGraphicsScene:
        """Return the QGraphicsScene used as the canvas (may be interactive)."""

    @abstractmethod
    def build_model(self):
        """
        Build and return (model, optimizer).
        Raise ValueError with a user-readable message if the problem is not ready.
        """

    @abstractmethod
    def render(self, model) -> QPixmap:
        """Convert current label state into a QPixmap for display."""

    @abstractmethod
    def num_labels(self) -> int:
        """Return the number of labels in the current problem instance."""
