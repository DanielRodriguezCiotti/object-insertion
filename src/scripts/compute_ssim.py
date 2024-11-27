"""This script computes the SSIM metric between images in two directories."""

import argparse
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from src.utils import resize_images


def compute_ssim(output_file: str, ground_truth_dir: str, generated_dir: str):
    """Compute the SSIM metric between images in two directories."""
    gt_files = os.listdir(ground_truth_dir)
    distances = []
    with open(output_file.replace(".txt", "_all.txt"), "w") as f:
        for file in tqdm(gt_files):
            gt_file_path = os.path.join(ground_truth_dir, file)
            gen_file_path = os.path.join(generated_dir, file)

            if os.path.exists(gen_file_path):
                # Load the images
                images = {}
                images["gt"] = np.array(Image.open(gt_file_path).convert("L"))
                images["gen"] = np.array(Image.open(gen_file_path).convert("L"))
                # Resize the images to the common size if needed
                if images["gt"].shape != images["gen"].shape:
                    images = resize_images(images)
                # Compute SSIM between the two images
                ssim_value = ssim(images["gt"], images["gen"])
                # Write the SSIM value to the file
                f.write(f"{file}: SSIM = {ssim_value:.6f}\n")
                distances.append(ssim_value)

    # Aggregate the results and write to another file
    with open(output_file, "w") as f_agg:
        f_agg.writelines("Aggregated results:\n")
        f_agg.writelines(f"Mean: {sum(distances) / len(distances):.6f}\n")
        f_agg.writelines(f"Max: {max(distances):.6f}\n")
        f_agg.writelines(f"Min: {min(distances):.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d0",
        "--dir0",
        type=str,
        required=True,
        help="Path to the ground truth image directory",
    )
    parser.add_argument(
        "-d1",
        "--dir1",
        type=str,
        required=True,
        help="Path to the generated image directory",
    )
    parser.add_argument(
        "-o", "--out", type=str, required=True, help="Output file to store SSIM results"
    )

    args = parser.parse_args()
    results_folder = "/".join(args.out.split("/")[:-1])
    os.makedirs(results_folder, exist_ok=True)

    compute_ssim(args.out, args.dir0, args.dir1)


if __name__ == "__main__":
    main()
