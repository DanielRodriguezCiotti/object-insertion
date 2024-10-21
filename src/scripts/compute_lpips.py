"""This script computes the LPIPS metric between images in two directories."""

import argparse
import os

import lpips
import torch
from tqdm import tqdm

from src.utils import resize_images  # Assuming you want to use resize logic if needed


def compute_lpips(
    output_file: str,
    ground_truth_dir: str,
    generated_dir: str,
    version: str,
) -> None:
    """Compute the LPIPS metric between images in two directories."""
    # Initialize the LPIPS model
    loss_fn = lpips.LPIPS(net="alex", version=version)
    if torch.cuda.is_available():
        loss_fn.cuda()

    gt_files = os.listdir(ground_truth_dir)
    distances = []

    # Open file to log individual LPIPS scores
    with open(output_file.replace(".txt", "_all.txt"), "w") as f:
        for file in tqdm(gt_files):
            gt_file_path = os.path.join(ground_truth_dir, file)
            gen_file_path = os.path.join(generated_dir, file)

            if os.path.exists(gen_file_path):
                # Load the images
                images = {}
                images["gt"] = lpips.load_image(gt_file_path)
                images["gen"] = lpips.load_image(gen_file_path)
                if images["gt"].shape != images["gen"].shape:
                    images = resize_images(images)
                # Convert the images to tensors
                img0 = lpips.im2tensor(images["gt"])
                img1 = lpips.im2tensor(images["gen"])

                # Move to GPU if required
                if torch.cuda.is_available():
                    img0 = img0.cuda()
                    img1 = img1.cuda()

                # Compute LPIPS between the two images
                with torch.no_grad():
                    lpips_value = loss_fn.forward(img0, img1).squeeze().item()

                # Write the LPIPS value to the file
                f.write(f"{file}: LPIPS = {lpips_value:.6f}\n")
                distances.append(lpips_value)

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
        "-o",
        "--out",
        type=str,
        required=True,
        help="Output file to store LPIPS results",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="0.1",
        help="Version of the LPIPS model to use",
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Turn on flag to use GPU"
    )

    args = parser.parse_args()

    compute_lpips(args.out, args.dir0, args.dir1, args.version)


if __name__ == "__main__":
    main()
