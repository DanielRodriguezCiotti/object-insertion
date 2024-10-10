"""This script computes the SSIM metric between images in two directories."""

import argparse
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-d0", "--dir0", type=str, required=True, help="Path to the first image directory"
)
parser.add_argument(
    "-d1", "--dir1", type=str, required=True, help="Path to the second image directory"
)
parser.add_argument(
    "-o", "--out", type=str, required=True, help="Output file to store SSIM results"
)

args = parser.parse_args()

# Open the output file for writing results
with open(args.out, "w") as f:
    files = os.listdir(args.dir0)

    distances = []
    for file in tqdm(files):
        file_path1 = os.path.join(args.dir0, file)
        file_path2 = os.path.join(args.dir1, file)

        if os.path.exists(file_path2):
            # Load the images
            image1 = np.array(Image.open(file_path1).convert("L"))
            image2 = np.array(Image.open(file_path2).convert("L"))

            # Compute SSIM between the two images
            ssim_value = ssim(image1, image2)

            # Write the SSIM value to the file
            f.write(f"{file}: SSIM = {ssim_value:.6f}\n")
            distances.append(ssim_value)

# Aggregate the results and write to another file
with open(args.out.replace(".txt", "_agg.txt"), "w") as f_agg:
    f_agg.writelines("Aggregated results:\n")
    f_agg.writelines(f"Mean: {sum(distances) / len(distances):.6f}\n")
    f_agg.writelines(f"Max: {max(distances):.6f}\n")
    f_agg.writelines(f"Min: {min(distances):.6f}\n")
