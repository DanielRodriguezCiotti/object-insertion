import os
import shutil

def copy_filtered_images(file_list_path, source_dir, target_dir):
    """
    Copies images from the source directory to the target directory, filtering by a list of files.
    
    Parameters:
        file_list_path (str): Path to the file containing the list of filenames to copy.
        source_dir (str): Path to the source directory containing the images.
        target_dir (str): Path to the target directory where the images will be copied.
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Read the file list
    with open(file_list_path, "r") as f:
        files_to_copy = [line.strip() for line in f.readlines()]

    print(f"Found {len(files_to_copy)} files to copy.")

    # Copy the files
    copied_count = 0
    for file_name in files_to_copy:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        if os.path.exists(source_path):
            # Ensure subdirectories are created in the target directory
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied_count += 1
        else:
            print(f"File not found: {source_path}")

    print(f"Copied {copied_count}/{len(files_to_copy)} files to {target_dir}.")

# Example usage
if __name__ == "__main__":
    # Input paths
    file_list = "list.txt"  # Path to file containing the list of filenames
    source = "/home/daniel/code/anydoor-refiners/dataset/test/image"         # Source folder containing images
    target = "data/anydoor_subset_gt"         # Target folder to copy images to

    # Copy the filtered images
    copy_filtered_images(file_list, source, target)
