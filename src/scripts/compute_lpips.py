"""This script computes the LPIPS metric between images in two directories."""

import argparse
import os

import lpips
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d0", "--dir0", type=str, default="./imgs/ex_dir0")
parser.add_argument("-d1", "--dir1", type=str, default="./imgs/ex_dir1")
parser.add_argument("-o", "--out", type=str, default="./imgs/example_dists.txt")
parser.add_argument("-v", "--version", type=str, default="0.1")
parser.add_argument("--use_gpu", action="store_true", help="turn on flag to use GPU")

args = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net="alex", version=args.version)
if args.use_gpu:
    loss_fn.cuda()

# crawl directories
f = open(args.out.replace(".txt", "_all.txt"), "w")  # noqa: SIM115
files = os.listdir(args.dir0)

distances = []
for file in tqdm(files):
    if os.path.exists(os.path.join(args.dir1, file)):
        # Load images
        img0 = lpips.im2tensor(
            lpips.load_image(os.path.join(args.dir0, file))
        )  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(os.path.join(args.dir1, file)))

        if args.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()

        # Compute distance
        with torch.no_grad():
            dist01 = loss_fn.forward(img0, img1)

        distances.append(dist01.squeeze().item())  # type: ignore
        f.writelines(f"{file}: {dist01.squeeze().item():.6f}\n")  # type: ignore


f.close()

f_agg = open(args.out, "w")  # noqa: SIM115
f_agg.writelines("Aggregated results:\n")
f_agg.writelines("Mean: %.6f" % (sum(distances) / len(distances)))
f_agg.writelines("Max: %.6f" % max(distances))
f_agg.writelines("Min: %.6f" % min(distances))
f_agg.close()
