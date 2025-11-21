import tifffile
import os

def load_tiff_files(data_folder):
    """
    Load all .tif files from the specified data folder into a dictionary.

    Args:
        data_folder (str): Path to the folder containing .tif files.

    Returns:
        dict: Dictionary with filenames as keys and image arrays as values.
    """
    # List all .tif files in the data folder
    tiff_files = [f for f in os.listdir(data_folder) if f.endswith('.tif')]

    # Dictionary to hold the loaded images
    images = {}

    for file in tiff_files:
        file_path = os.path.join(data_folder, file)
        # Load the tiff file
        image = tifffile.imread(file_path)
        images[file] = image
        print(f"Loaded {file} with shape {image.shape}")

    return images

# Example usage (can be removed or commented out)
if __name__ == "__main__":
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    images = load_tiff_files(data_folder)
    # Now you can access the images, e.g., images['Embryo_37_intrareg_fuse_t077.tif']
