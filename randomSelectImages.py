import os
import random
from pathlib import Path

def select_random_images(directory_path, num_images=100):
    """
    Selects a specified number of image files randomly from a directory,
    including its subdirectories.

    Args:
        directory_path (str): The path to the directory containing the images.
        num_images (int, optional): The number of images to select. Defaults to 100.

    Returns:
        list: A list of Path objects representing the selected image files.
              Returns an empty list if the directory is empty or doesn't exist,
              or if the number of images to select is invalid.
    """
    # Convert the directory path to a Path object for easier manipulation
    directory = Path(directory_path)

    # Check if the directory exists
    if not directory.is_dir():
        print(f"Error: Directory '{directory_path}' not found.")
        return []

    image_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # Create a Path object for each file
            file_path = Path(dirpath) / filename
            # Check if the file is an image (case-insensitive)
            if file_path.suffix.lower() in (".jpg", ".png"): # ".jpeg", ".gif", ".bmp", ".tiff", ".webp"
                image_files.append(file_path)

    # Check if there are enough images in the directory
    if len(image_files) < num_images:
        print(
            f"Warning: Not enough images in '{directory_path}'. Found {len(image_files)}, "
            f"but requested {num_images}. Returning all found images."
        )
        return image_files  # Return all available images

    # Select 'num_images' random images
    selected_images = random.sample(image_files, num_images)
    return selected_images
    

def main():
    """
    Main function to run the image selection process.  Prompts the user for the
    directory and displays the selected images.
    """
    # Get the directory path from the user
    #directory_path = input("Enter the path to the directory containing the images: ")
    directory_path = r"C:\Users\Derrick\Documents\School\Computer Vision\project\datasets\Celeb Images"

    # Get the number of images to select from the user
    #num_images_str = input(f"Enter the number of images to select (default is 100): ")
    num_images_str = 10
    if num_images_str:
        try:
            num_images = int(num_images_str)
            if num_images <= 0:
                print("Error: Number of images to select must be positive.  Using default value of 100.")
                num_images = 100 # set back to default
        except ValueError:
            print("Error: Invalid input for number of images. Using default value of 100.")
            num_images = 100  # Use the default value
    else:
        num_images = 100 # Default value if user enters nothing

    # Call the function to select the images
    selected_images = select_random_images(directory_path, num_images)

    # Print the selected image paths
    if selected_images:
        print("\nSelected images:")
        for image_path in selected_images:
            print(image_path)
        print(f"\nSuccessfully selected {len(selected_images)} images.")
    else:
        print("No images selected.")  # select_random_images already prints an error if needed

if __name__ == "__main__":
    main()
