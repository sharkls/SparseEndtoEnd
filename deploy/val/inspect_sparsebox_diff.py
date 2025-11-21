import argparse, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("path", help="val/diff_node*_sample*.npy")
parser.add_argument("--topk", type=int, default=5)
args = parser.parse_args()

diff = np.load(args.path)          # 形状 (1, N, P, 3)
flat = diff.reshape(-1)
idx = np.argsort(flat)[::-1][:args.topk]

for rank, i in enumerate(idx, 1):
    b, anchor, kp, xyz = np.unravel_index(i, diff.shape)
    print(f"#{rank}: anchor={anchor}, keypoint={kp}, axis={xyz}, diff={flat[i]:.6f}")