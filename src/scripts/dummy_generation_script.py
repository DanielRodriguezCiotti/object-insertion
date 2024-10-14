"""This script processes images by adding Gaussian noise to them.

Usage:
    python dummy_generation_script.py <input_folder> <output_folder> [--mean MEAN]
    [--std STD]

Arguments:
    input_folder: Path to the input image folder.
    output_folder: Path to the output folder where noisy images will be saved.
    --mean: Mean of the Gaussian noise (default: 0).
    --std: Standard deviation of the Gaussian noise (default: 25).
"""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def add_gaussian_noise(image, mean=0, std=25):
    """Add Gaussian noise to an image.

    :param image: Input image as a NumPy array.
    :param mean: Mean of the Gaussian noise.
    :param std: Standard deviation of the Gaussian noise.
    :return: Image with added Gaussian noise.
    """
    gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def process_images(input_folder, output_folder, mean, std):
    """Process each image by adding Gaussian noise and save to the output folder.

    :param input_folder: Path to the folder containing input images.
    :param output_folder: Path to the folder to save noisy images.
    :param mean: Mean of the Gaussian noise.
    :param std: Standard deviation of the Gaussian noise.
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)
    # Iterate through each file in the input folder
    for filename in tqdm(files):
        input_image_path = os.path.join(input_folder, filename)
        # Read the image
        image = cv2.imread(input_image_path)
        # Add Gaussian noise to the image
        noisy_image = add_gaussian_noise(image, mean, std)
        # Save the noisy image to the output folder
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, noisy_image)


def main():
    """Main function to parse arguments and process images."""
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to images in a folder."
    )
    parser.add_argument(
        "--input-folder", type=str, help="Path to the input image folder."
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Path to the output folder where noisy images will be saved.",
    )
    parser.add_argument(
        "--mean", type=float, default=0, help="Mean of the Gaussian noise (default: 0)."
    )
    parser.add_argument(
        "--std",
        type=float,
        default=25,
        help="Standard deviation of the Gaussian noise (default: 25).",
    )

    args = parser.parse_args()

    # Process images with the specified parameters
    process_images(args.input_folder, args.output_folder, args.mean, args.std)


if __name__ == "__main__":
    main()
