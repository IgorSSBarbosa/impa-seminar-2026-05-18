"""
Critical parameters for site percolation on Z^d.

p_c values are from high-precision numerical studies.
Fractal dimensions:
  d < 6  : non-mean-field regime  (d_f varies, < 4)
  d >= 6 : mean-field regime      (d_f = 4 exactly, upper critical dim)
"""

# Site percolation thresholds on the hypercubic lattice Z^d
CRITICAL_PROBS = {
    2: 0.5927462,
    3: 0.3116077,
    4: 0.1968861,
    5: 0.1406966,
    6: 0.1090820,
    7: 0.0786752,
    8: 0.0608227,
}

# Fractal dimensions of the critical cluster
FRACTAL_DIMS = {
    2: 91 / 48,   # Exact: ~1.896
    3: 2.523,
    4: 3.054,
    5: 3.674,
    6: 4.0,       # Mean-field onset
    7: 4.0,
    8: 4.0,
}

# Per-dimension configuration
DIMENSION_CONFIGS = {
    2: {
        'p_c':              CRITICAL_PROBS[2],
        'R':                400,       # L-inf exploration radius
        'min_cluster':      2000,      # Minimum accepted cluster size
        'max_retries':      50,
        'zoom_start_r':     30,        # L-inf radius shown initially
        'render_mode':      '2d',
    },
    3: {
        'p_c':              CRITICAL_PROBS[3],
        'R':                100,
        'min_cluster':      500,
        'max_retries':      1000,
        'zoom_start_r':     12,
        'max_display_nodes':3000,      # Cap for scatter rendering speed
        'render_mode':      '3d',
    },
    4: {
        'p_c':              CRITICAL_PROBS[4],
        'R':                50,
        'min_cluster':      500,
        'max_retries':      1000,
        'zoom_start_depth': 5,
        'max_spring_nodes': 2000,
        'render_mode':      'graph',
    },
    5: {
        'p_c':              CRITICAL_PROBS[5],
        'R':                40,
        'min_cluster':      500,
        'max_retries':      1000,
        'zoom_start_depth': 4,
        'max_spring_nodes': 2000,
        'render_mode':      'graph',
    },
    6: {
        'p_c':              CRITICAL_PROBS[6],
        'R':                40,
        'min_cluster':      500,
        'max_retries':      1000,
        'zoom_start_depth': 3,
        'max_spring_nodes': 2000,
        'render_mode':      'graph',
    },
    7: {
        'p_c':              CRITICAL_PROBS[7],
        'R':                40,
        'min_cluster':      200,
        'max_retries':      10000,
        'zoom_start_depth': 3,
        'max_spring_nodes': 2000,
        'render_mode':      'graph',
    },
    8: {
        'p_c':              CRITICAL_PROBS[8],
        'R':                40,
        'min_cluster':      100,
        'max_retries':      10000,
        'zoom_start_depth': 2,
        'max_spring_nodes': 2000,
        'render_mode':      'graph',
    },
}

# ---------------------------------------------------------------------------
# Animation timing
# ---------------------------------------------------------------------------
ANIM_FPS = 3

# d=2 uses fast PIL bitmap rendering — more frames
ANIM_2D = dict(zoom_in_frames=5, zoom_out_frames=20, freeze_frames=20)
ANIM_2D['total'] = (ANIM_2D['zoom_in_frames']
                    + ANIM_2D['zoom_out_frames']
                    + ANIM_2D['freeze_frames'])   # 85

# d>=3 uses matplotlib 3D rendering — fewer frames for speed
ANIM_3D = dict(zoom_in_frames=3, zoom_out_frames=20, freeze_frames=20)
ANIM_3D['total'] = (ANIM_3D['zoom_in_frames']
                    + ANIM_3D['zoom_out_frames']
                    + ANIM_3D['freeze_frames'])   # 40

# ---------------------------------------------------------------------------
# Figure / output settings
# ---------------------------------------------------------------------------
FIGURE_DPI  = 160
FIGURE_SIZE = (16, 12)                          # inches  →  640 × 480 px
TARGET_SIZE = (1280, 960)                      # all frames resized to this

# Colours
CMAP_NAME = 'plasma'
BG_COLOR  = '#050510'     # deep dark blue-black
EDGE_COL  = '#2a2a5a'     # dark purple for graph edges