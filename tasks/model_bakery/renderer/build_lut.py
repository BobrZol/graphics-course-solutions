from pathlib import Path
import argparse

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("graded")
parser.add_argument("output")
args = parser.parse_args()

source = np.asarray(Image.open(args.source).convert("RGB"))
graded = np.asarray(Image.open(args.graded).convert("RGB"))

if source.shape != graded.shape:
    raise ValueError("Images must have identical dimensions")

cells = np.minimum(source.astype(np.uint32) * 32 // 256, 31).reshape(-1, 3)
indices = cells[:, 0] + 32 * cells[:, 1] + 1024 * cells[:, 2]
counts = np.bincount(indices, minlength=32768)
sums = np.stack([
    np.bincount(indices, weights=graded.reshape(-1, 3)[:, channel], minlength=32768)
    for channel in range(3)
], axis=1)

filled = counts > 0
colors = np.zeros((32768, 4), dtype=np.uint8)
colors[filled, :3] = np.rint(sums[filled] / counts[filled, None]).astype(np.uint8)
colors[:, 3] = 255

coordinates = np.stack(np.unravel_index(np.arange(32768), (32, 32, 32)), axis=1)
if np.any(~filled):
    tree = cKDTree(coordinates[filled])
    nearest = tree.query(coordinates[~filled])[1]
    colors[~filled] = colors[filled][nearest]

Path(args.output).write_bytes(colors.tobytes())
