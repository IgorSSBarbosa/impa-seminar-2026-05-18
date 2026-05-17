"""
Assembles per-dimension frame lists into a single looping GIF.
"""

import os
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from percolation.critical_params import ANIM_FPS, DIMENSION_CONFIGS
from visualization.render_2d    import generate_2d_frames
from visualization.render_3d    import generate_3d_frames
from visualization.render_graph import generate_graph_frames


def build_animation(clusters, output_path, fps=None):
    """
    Render every dimension and save as a single GIF.

    Parameters
    ----------
    clusters    : dict  { dim : cluster_dict }
    output_path : str   — where to save the .gif
    fps         : int   — frames per second (default: ANIM_FPS from config)
    """
    fps = fps or ANIM_FPS

    all_frames = []

    for dim in range(2, 9):
        print(f"\n{'='*55}")
        print(f"  Dimension {dim}")
        print(f"{'='*55}")

        cluster = clusters[dim]
        cfg     = DIMENSION_CONFIGS[dim]
        mode    = cfg['render_mode']

        if mode == '2d':
            frames = generate_2d_frames(cluster, cfg)
        elif mode == '3d':
            frames = generate_3d_frames(cluster, cfg)
        else:                                        # 'graph'
            frames = generate_graph_frames(cluster, cfg)

        print(f"  -> {len(frames)} frames generated for d={dim}")
        all_frames.extend(frames)

    # --- Assemble GIF ----------------------------------------------------
    total = len(all_frames)
    duration_ms = int(1000 / fps)

    print(f"\n{'='*55}")
    print(f"  Assembling GIF")
    print(f"  Frames   : {total}")
    print(f"  FPS      : {fps}  ->  {total/fps:.1f} s animation")
    print(f"  Output   : {output_path}")
    print(f"{'='*55}")

    # Quantise to 256-colour palette (required by GIF)
    print("  Quantising frames ...", flush=True)
    pal_frames = [f.quantize(colors=256) for f in tqdm(all_frames, desc='  quantise')]

    print("  Writing GIF ...", flush=True)
    pal_frames[0].save(
        output_path,
        format='GIF',
        save_all=True,
        append_images=pal_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved!  File size: {size_mb:.1f} MB")