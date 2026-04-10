import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build")))
import alpha_expansion_py as ae

from .problem import Problem

def _auto_seed(G: nx.Graph, num_labels: int) -> dict:
    """Greedy farthest-point seeding for unseeded graphs.

    Picks num_labels nodes that are maximally spread out in BFS-distance space
    so the pairwise terms can propagate meaningful community structure outward
    from each seed.
    """
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    start = max(G.degree(), key=lambda x: x[1])[0]
    seed_nodes = [start]
    for _ in range(num_labels - 1):
        best, best_dist = None, -1
        for node in G.nodes():
            if node in seed_nodes:
                continue
            d = min(lengths.get(node, {}).get(s, float("inf")) for s in seed_nodes)
            if d > best_dist:
                best_dist, best = d, node
        if best is not None:
            seed_nodes.append(best)
    return {node: i for i, node in enumerate(seed_nodes)}


_GRAPHS = [
    {
        "name": "Karate Club",
        "graph_fn": nx.karate_club_graph,
        "seeds": {0: 0, 33: 1},
        "num_labels": 2,
        "lambda_val": 10,
    },
    {
        "name": "Les Misérables",
        "graph_fn": nx.les_miserables_graph,
        "seeds": {
            "Valjean": 0,
            "Cosette": 0,
            "Marius": 0,
            "Fauchelevent": 0,
            "Javert": 1,
            "Enjolras": 1,
            "Gavroche": 1,
            "Combeferre": 1,
            "Thenardier": 2,
            "MmeThenardier": 2,
            "Boulatruelle": 2,
        },
        "num_labels": 3,
        "lambda_val": 8,
    },
]


class CommunityDetectionProblem(Problem):
    def __init__(self):
        self._config = None
        self._G = None
        self._node_to_idx = None
        self._idx_to_node = None
        self._scene = None
        self._param_widget = None
        self._btn_group = None
        self._on_graph_selected = None

    def get_scene(self) -> QGraphicsScene:
        if self._scene is None:
            self._scene = QGraphicsScene()
        return self._scene

    def get_param_widget(self) -> QWidget:
        if self._param_widget is not None:
            return self._param_widget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self._btn_group = QButtonGroup(widget)
        self._btn_group.setExclusive(True)
        for i, config in enumerate(_GRAPHS):
            btn = QPushButton(config["name"])
            btn.setCheckable(True)
            self._btn_group.addButton(btn, i)
            layout.addWidget(btn)
        self._btn_group.idClicked.connect(self._select_graph)

        sep = QLabel("— or —")
        sep.setStyleSheet("color: grey;")
        layout.addWidget(sep)

        self._btn_custom = QPushButton("Load Custom Graph…")
        self._btn_custom.clicked.connect(self._load_custom_graph)
        layout.addWidget(self._btn_custom)

        self._param_widget = widget
        return widget

    def _select_graph(self, idx: int):
        self._config = _GRAPHS[idx]
        G = self._config["graph_fn"]()
        self._G = G
        self._node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        self._idx_to_node = {i: node for node, i in self._node_to_idx.items()}
        if hasattr(self, "_btn_custom"):
            self._btn_custom.setText("Load Custom Graph…")
        if self._on_graph_selected:
            self._on_graph_selected()

    def _load_custom_graph(self):
        file_name, _ = QFileDialog.getOpenFileName(
            None,
            "Open Graph File",
            "",
            "Graph files (*.edgelist *.txt *.graphml *.gml);;All files (*)",
        )
        if not file_name:
            return

        ext = os.path.splitext(file_name)[1].lower()
        try:
            if ext in (".edgelist", ".txt"):
                G = nx.read_edgelist(file_name)
            elif ext == ".graphml":
                G = nx.read_graphml(file_name)
            elif ext == ".gml":
                G = nx.read_gml(file_name)
            else:
                G = nx.read_edgelist(file_name)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load graph:\n{e}")
            return

        dialog = QDialog()
        dialog.setWindowTitle("Custom Graph Settings")
        form = QFormLayout(dialog)
        spin_labels = QSpinBox()
        spin_labels.setRange(2, 9)
        spin_labels.setValue(2)
        spin_lambda = QSpinBox()
        spin_lambda.setRange(1, 1000)
        spin_lambda.setValue(10)
        form.addRow("Number of communities:", spin_labels)
        form.addRow("Lambda (smoothness):", spin_lambda)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        checked = self._btn_group.checkedButton()
        if checked:
            self._btn_group.setExclusive(False)
            checked.setChecked(False)
            self._btn_group.setExclusive(True)

        self._config = {
            "name": os.path.basename(file_name),
            "num_labels": spin_labels.value(),
            "lambda_val": spin_lambda.value(),
            "seeds": {},
        }
        self._G = G
        self._node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        self._idx_to_node = {i: node for node, i in self._node_to_idx.items()}
        self._btn_custom.setText(f"Custom: {os.path.basename(file_name)}")
        if self._on_graph_selected:
            self._on_graph_selected()

    def build_model(self):
        if self._config is None:
            raise ValueError("No graph selected. Please choose a graph first.")

        G = self._G
        num_nodes = G.number_of_nodes()
        num_labels = self._config["num_labels"]
        seeds = self._config["seeds"]
        lambda_val = self._config["lambda_val"]
        node_to_idx = self._node_to_idx
        idx_to_node = self._idx_to_node

        if not seeds:
            seeds = _auto_seed(G, num_labels)

        model = ae.EnergyModel(num_nodes, num_labels, "int32")
        for u, v in G.edges():
            model.add_neighbor(node_to_idx[u], node_to_idx[v])

        def unary_cost(idx, label):
            node = idx_to_node[idx]
            if node in seeds:
                return 0 if label == seeds[node] else 1000
            return 0

        def pairwise_cost(n1, n2, l1, l2):
            return 0 if l1 == l2 else lambda_val

        model.set_unary_cost_fn(unary_cost)
        model.set_pairwise_cost_fn(pairwise_cost)
        optimizer = ae.AlphaExpansionInt(model, "bk")
        return model, optimizer

    def render(self, model) -> QPixmap:
        labels = model.get_labels()
        G = self._G
        node_to_idx = self._node_to_idx
        color_palette = list(mcolors.TABLEAU_COLORS.values())
        node_colors = [
            color_palette[labels[node_to_idx[n]] % len(color_palette)] for n in G.nodes()
        ]

        fig, ax = plt.subplots(figsize=(8, 7))
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog="neato")
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx(
            G, pos,
            node_color=node_colors,
            with_labels=True,
            node_size=600,
            font_size=8,
            edge_color="#CCCCCC",
            ax=ax,
        )
        ax.set_title(f"{self._config['name']} — Energy: {model.evaluate_total_energy()}")
        ax.axis("off")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.read()))

    def num_labels(self) -> int:
        if self._config is None:
            return 0
        return self._config["num_labels"]
