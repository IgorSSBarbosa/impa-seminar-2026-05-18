"""
Spring-layout graph rendering for d >= 4 percolation clusters.

Pipeline per dimension
----------------------
1. Subsample cluster to max_spring_nodes (BFS order = innermost nodes).
2. Build a networkx Graph.
3. Compute Fruchterman-Reingold layout in R^3 (once, expensive).
4. Animate by revealing nodes in BFS-depth order (simulates zoom-out from
   origin outward).
5. During freeze phase, slowly rotate camera.

Why spring layout?
------------------
In d >= 4 (and especially d >= 6, the mean-field regime), the cluster
becomes tree-like (loops are rare). The spring layout naturally exposes
this branching, tree-like topology in a beautiful 3D embedding.
"""

import io
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from percolation.critical_params import (
    ANIM_3D, FIGURE_SIZE, FIGURE_DPI, CMAP_NAME, BG_COLOR,
    EDGE_COL, TARGET_SIZE, FRACTAL_DIMS,
)
from visualization.render_3d import fig_to_pil, style_3d_axes


def generate_graph_frames(cluster, cfg):
    """
    Generate all animation frames for d >= 4 as a list of PIL RGB images.
    """
    dim       = cluster['dim']
    max_nodes = cfg.get('max_spring_nodes', 400)

    # --- Subsample -------------------------------------------------------
    coords_list = cluster['coords'][:max_nodes]
    n           = len(coords_list)
    coord_set   = set(coords_list)

    depths_dict = {c: cluster['depths'][c] for c in coords_list}
    adj         = {
        c: [nb for nb in cluster['adjacency'][c] if nb in coord_set]
        for c in coords_list
    }
    max_depth = max(depths_dict.values())

    # --- Build graph -----------------------------------------------------
    print(f"  [d={dim}] Building graph ({n} nodes) ...", end=' ', flush=True)
    G = nx.Graph()
    G.add_nodes_from(coords_list)
    seen_edges = set()
    for node, neighbours in adj.items():
        for nb in neighbours:
            key = (min(node, nb), max(node, nb))
            if key not in seen_edges:
                seen_edges.add(key)
                G.add_edge(node, nb)
    n_edges = G.number_of_edges()
    print(f"{n_edges} edges")

    # --- Spring layout in R^3 -------------------------------------------
    print(f"  [d={dim}] Spring layout (n={n}, 50 iterations) ...", end=' ', flush=True)
    pos_dict = nx.spring_layout(
        G,
        dim=3,
        iterations=50,
        seed=42,
        scale=2.0,
    )
    print("done")

    # Align arrays with coords_list order
    positions   = np.array([pos_dict[nd] for nd in coords_list])   # (N, 3)
    node_depths = np.array([depths_dict[nd] for nd in coords_list]) # (N,)

    # Precompute per-edge depth arrays for fast masking
    edges_list  = list(G.edges())
    if edges_list:
        edge_segs   = np.array([[pos_dict[u], pos_dict[v]] for u, v in edges_list])
        edge_d_u    = np.array([depths_dict[u] for u, v in edges_list])
        edge_d_v    = np.array([depths_dict[v] for u, v in edges_list])
    else:
        edge_segs = np.zeros((0, 2, 3))
        edge_d_u  = edge_d_v = np.zeros(0, dtype=int)

    # Colours (by node BFS depth)
    cmap        = plt.get_cmap(CMAP_NAME)
    node_colors = cmap(node_depths / max(max_depth, 1))   # (N, 4)

    # Zoom: reveal by BFS depth
    start_depth = cfg.get('zoom_start_depth', 3)

    d_f     = FRACTAL_DIMS.get(dim, 4.0)
    d_f_str = f'{d_f:.3f}' if isinstance(d_f, float) else str(d_f)
    regime  = 'mean-field' if dim >= 6 else 'non-mean-field'

    timing = ANIM_3D
    total  = timing['total']

    print(f"  [d={dim}] Rendering {total} frames ...")

    frames = []
    for i in tqdm(range(total), desc=f'  d={dim}  '):
        # Depth threshold for this frame
        if i < timing['zoom_in_frames']:
            cur_max_d = start_depth
        elif i < timing['zoom_in_frames'] + timing['zoom_out_frames']:
            t         = (i - timing['zoom_in_frames']) / timing['zoom_out_frames']
            t_ease    = 1.0 - (1.0 - t) ** 3
            cur_max_d = int(round(start_depth + t_ease * (max_depth - start_depth)))
        else:
            cur_max_d = max_depth

        node_mask = node_depths <= cur_max_d
        edge_mask = (edge_d_u <= cur_max_d) & (edge_d_v <= cur_max_d)
        n_visible = int(node_mask.sum())

        # Camera
        if i < timing['zoom_in_frames'] + timing['zoom_out_frames']:
            azim = 30.0
        else:
            freeze_i = i - timing['zoom_in_frames'] - timing['zoom_out_frames']
            azim     = 30.0 + freeze_i * 4.5

        fig = plt.figure(figsize=FIGURE_SIZE, facecolor=BG_COLOR)
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(BG_COLOR)

        # Draw edges
        if edge_mask.any():
            lc = Line3DCollection(
                edge_segs[edge_mask],
                colors=EDGE_COL,
                linewidths=0.6,
                alpha=0.45,
            )
            ax.add_collection3d(lc)

        # Draw nodes
        if node_mask.any():
            vis = positions[node_mask]
            ax.scatter(
                vis[:, 0], vis[:, 1], vis[:, 2],
                c=node_colors[node_mask],
                s=14,
                alpha=0.92,
                linewidths=0,
                depthshade=True,
            )
            # Highlight origin (depth 0, always index 0 after sorting)
            if node_mask[0]:
                ax.scatter(
                    [positions[0, 0]], [positions[0, 1]], [positions[0, 2]],
                    c='white', s=70, alpha=1.0, linewidths=0, zorder=10,
                )

        # Axis limits: fit to visible nodes with margin
        if node_mask.any():
            vis  = positions[node_mask]
            margin = 0.25
            ax.set_xlim(vis[:, 0].min() - margin, vis[:, 0].max() + margin)
            ax.set_ylim(vis[:, 1].min() - margin, vis[:, 1].max() + margin)
            ax.set_zlim(vis[:, 2].min() - margin, vis[:, 2].max() + margin)

        style_3d_axes(ax)
        ax.view_init(elev=20, azim=azim)

        fig.suptitle(
            f'Dimension {dim}  —  Site Percolation at Criticality\n'
            f'$p_c = {cfg["p_c"]:.4f}$'
            f'   |   {n_visible}/{n} nodes'
            f'   |   $d_f = {d_f_str}$  ({regime})'
            f'   |   Spring layout in $\\mathbb{{R}}^3$',
            color='white', fontsize=10, y=0.97,
        )

        frames.append(fig_to_pil(fig))

    return frames