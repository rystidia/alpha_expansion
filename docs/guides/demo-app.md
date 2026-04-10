# Demo App

The library includes an interactive PyQt6 application that shows the alpha-expansion
algorithm working step by step on two problems: image segmentation and community detection.

## Prerequisites

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The app needs PyQt6, numpy, Pillow, networkx and matplotlib, all listed in `requirements.txt`.

## Launching the App

From the project root (with the virtual environment active):

```bash
python -m demo.interactive_segmentation
```

A window opens with a sidebar on the left and a canvas on the right.

## Image Segmentation Mode

The app opens in this mode. You load an image and draw seed strokes to mark regions,
then the algorithm segments the image based on those strokes.

**Workflow:**

1. Click **Load Image** and open any PNG or JPEG file.
2. The image appears on the canvas. Two labels are pre-configured (red and green).
3. Select a label with the colored button in the sidebar, then paint strokes on the
   canvas to mark the region. Click **+ Add Label** to add up to 9 labels total,
   or **X** to remove one.
4. Click **Initialize**. The algorithm builds an energy model from the strokes:
   - *Unary costs* come from the Mahalanobis distance of each pixel's color to
     the Gaussian color model fitted on each label's strokes.
   - *Pairwise costs* are edge weights based on color similarity between neighbors.
5. Click **Step** to run one alpha-expansion move. The canvas updates to show the
   current labeling. Click it again to watch the algorithm converge step by step.
6. Click **Run to Convergence** to run `SequentialStrategy` (up to 20 cycles) and
   go straight to the final result.
7. A dialog appears when the algorithm converges, showing the final energy value.

Once you click Initialize you cannot add or remove labels or change strokes.
Start over with **Initialize** after switching the mode or reloading an image.

## Community Detection Mode

Click **Community Detection** at the top of the sidebar to switch modes. Here you
partition a social network graph into communities.

**Workflow:**

1. Choose a graph from the sidebar:
   - **Karate Club**: 34-node social network split into 2 communities. Nodes 0
     and 33 are pre-seeded as the two group leaders.
   - **Les Miserables**: 77-node character co-occurrence network split into 3
     communities. Key characters from each group are pre-seeded.
   - **Load Custom Graph…**: load your own graph from a file (see below).
2. Click **Initialize**. The energy model sets high unary costs for seeded nodes
   assigned to the wrong community, and penalizes adjacent nodes in different communities.
   Custom graphs have no seeds, so the algorithm partitions purely by minimizing
   the graph cut.
3. Click **Step** or **Run to Convergence**. After each step the graph renders with
   nodes colored by their current community assignment.

### Loading a Custom Graph

Click **Load Custom Graph…** to open a file dialog, then set the number of communities
and the smoothness weight (lambda) in the dialog that follows.

**Supported file formats:**

| Format | Extension | Description |
|--------|-----------|-------------|
| Edge list | `.edgelist`, `.txt` | One `node1 node2` pair per line; lines starting with `#` are comments |
| GraphML | `.graphml` | XML-based format exported by Gephi, igraph, and similar tools |
| GML | `.gml` | Graph Modelling Language, supported by most graph libraries |

**Tuning parameters:**

- **Number of communities** — how many partitions to produce (2–9).
- **Lambda (smoothness)** — pairwise penalty for placing adjacent nodes in different
  communities. Higher values push the algorithm toward fewer, larger communities.
  Start around 10 and adjust if the result is under- or over-segmented.

### Example Graph Files

`demo/graphs/` contains two ready-to-use edge lists:

| File | Nodes | Edges | Suggested settings |
|------|-------|-------|--------------------|
| `two_cliques.edgelist` | 12 | 31 | 2 communities, lambda 10–20 |
| `three_cliques.edgelist` | 15 | 33 | 3 communities, lambda 5–15 |

`two_cliques.edgelist` contains two 6-node cliques joined by a single bridge edge —
the algorithm should recover the two cliques as perfect communities in one step.
`three_cliques.edgelist` contains three 5-node cliques arranged in a ring, each
pair connected by one bridge edge.

## Controls Reference

| Control | Description |
|---------|-------------|
| **Segmentation / Community Detection** | Switch problem mode. Resets state. |
| **Initialize** | Build the energy model from the current configuration. |
| **Step** | Run one alpha-expansion move (cycles through labels 0, 1, ...). |
| **Run to Convergence** | Run `SequentialStrategy` up to 20 cycles. |
