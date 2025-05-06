import cv2
from skimage.feature import hog
import numpy as np
from skimage import io, color, exposure

def calculate_hog_from_face_frame(face_frame, target_size=(200, 300)):
    """
    Calculates the HOG descriptor for a Region of Interest (ROI) within a face frame, resizing to a target size.

    Args:
        frame (numpy.ndarray): The video frame as a numpy array (e.g., from cv2.VideoCapture).
        target_size (tuple, optional): The target size (width, height) to resize the ROI to.
            Defaults to (200, 300).

    Returns:
        tuple: A tuple containing the HOG descriptor and the visualization image.
               Returns (None, None) if there is an error.
    """
    try:
        
        # Extract the ROI from the frame, which is the whole frame
        roi = face_frame

        # Convert the ROI to grayscale
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Resize the ROI to the target size
        resized_roi = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

        # Calculate HOG features
        hog_features, hog_visualization = hog(resized_roi,
                                              orientations=8,
                                              pixels_per_cell=(16, 16),
                                              cells_per_block=(1, 1),
                                              visualize=True,
                                              feature_vector=True)  # Ensure feature vector is returned

        return hog_features, hog_visualization

    except Exception as e:
        print(f"Error processing frame ROI: {e}")
        return None, None


def calculate_hog_from_frame(frame, bbox, target_size=(200, 300)):
    """
    Calculates the HOG descriptor for a Region of Interest (ROI) within a video frame, resizing to a target size.

    Args:
        frame (numpy.ndarray): The video frame as a numpy array (e.g., from cv2.VideoCapture).
        bbox (tuple): A tuple representing the bounding box (x, y, w, h) of the ROI.
        target_size (tuple, optional): The target size (width, height) to resize the ROI to.
            Defaults to (200, 300).

    Returns:
        tuple: A tuple containing the HOG descriptor and the visualization image.
               Returns (None, None) if there is an error.
    """
    try:
        #x, y, w, h = bbox
        x, y, w, h = map(int, bbox) # unpacks the tensor [x,y,w,h, device=0]
        # Extract the ROI from the frame
        roi = frame[y:y+h, x:x+w]

        # Convert the ROI to grayscale
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Resize the ROI to the target size
        resized_roi = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

        # Calculate HOG features
        hog_features, hog_visualization = hog(resized_roi,
                                              orientations=8,
                                              pixels_per_cell=(16, 16),
                                              cells_per_block=(1, 1),
                                              visualize=True,
                                              feature_vector=True)  # Ensure feature vector is returned

        return hog_features, hog_visualization

    except Exception as e:
        print(f"Error processing frame ROI: {e}")
        return None, None

def calculate_hog_from_path(image_path, target_size=(200, 300)):
    """
    Calculates the HOG descriptor for an image, resizing to a target size.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple, optional): The target size (width, height) to resize the image to.
            Defaults to (200, 300).

    Returns:
        tuple: A tuple containing the HOG descriptor and the visualization image.
               Returns (None, None) if there is an error loading the image.
    """
    try:
        # Load the image
        img = io.imread(image_path)
        # If the image is RGB, convert it to grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        # Resize the image to the target size
        #original_size = img.shape
        new_size = target_size  # Use the provided target_size
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

        # Calculate HOG features
        hog_features, hog_visualization = hog(gray,
                                              orientations=8,
                                              pixels_per_cell=(16, 16),
                                              cells_per_block=(1, 1),
                                              visualize=True,
                                              feature_vector=True)  # Ensure feature vector is returned

        return hog_features, hog_visualization

    except Exception as e:
        print(f"Error processing image: {image_path} - {e}")
        return None, None



def compare_hog_features(hog1, hog2, method='euclidean'):
    """
    Compares two HOG feature vectors using a specified distance method.

    Args:
        hog1 (numpy.ndarray): The first HOG feature vector.
        hog2 (numpy.ndarray): The second HOG feature vector.
        method (str): The distance method to use ('euclidean', 'cosine', or 'correlation').
                        Defaults to 'euclidean'.

    Returns:
        float: The distance between the two HOG feature vectors.
               Returns None if an invalid method is specified or if input HOGs are None.
    """
    if hog1 is None or hog2 is None:
        return None

    if method == 'euclidean':
        return np.linalg.norm(hog1 - hog2)
    elif method == 'cosine':
        return 1 - np.dot(hog1, hog2) / (np.linalg.norm(hog1) * np.linalg.norm(hog2))
    elif method == 'correlation':
        hog1_centered = hog1 - np.mean(hog1)
        hog2_centered = hog2 - np.mean(hog2)
        return 1 - np.dot(hog1_centered, hog2_centered) / (np.std(hog1_centered) * np.std(hog2_centered) * len(hog1))
    else:
        print(f"Error: Invalid distance method '{method}'.  Choose 'euclidean', 'cosine', or 'correlation'.")
        return None


def main():
    """
    Main function to run the HOG comparison.
    """
    image_path1 = r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\whitelist\face\aligned_face.jpg"
    image_path2 = r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\whitelist\face\aligned_face2.jpg"

    # Calculate HOG features for both images
    hog1, hog_visualization1 = calculate_hog_from_path(image_path1)
    hog2, hog_visualization2 = calculate_hog_from_path(image_path2)

    if hog1 is None or hog2 is None:
        print("Comparison failed due to errors in HOG calculation.")
        return

    # Compare the HOG features
    distance = compare_hog_features(hog1, hog2, method='euclidean')  # You can change the method here

    if distance is not None:
        print(f"Distance between HOG features: {distance:.4f}")
        # You can set a threshold to determine similarity
        if distance < 0.15:  # Example threshold, adjust as needed
            print("The images are likely similar.")
        else:
            print("The images are likely different.")

        #Optionally display the HOG visualizations (for debugging or understanding the features)
        hog_vis1_rescaled = exposure.rescale_intensity(hog_visualization1)
        hog_vis2_rescaled = exposure.rescale_intensity(hog_visualization2)
        cv2.imshow("HOG Visualization 1", hog_vis1_rescaled) # Removed the cv2.imshow.
        cv2.imshow("HOG Visualization 2", hog_vis2_rescaled) # These would cause errors if running
        cv2.imwrite(r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\whitelist\face\hog_face1.jpg", hog_vis1_rescaled)
        cv2.imwrite(r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\whitelist\face\hog_face2.jpg", hog_vis2_rescaled)
        #print("HOG visualizations are not displayed.  Code uses skimage, not cv2, for HOG.") # Keep it simple and just print a message
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Comparison failed.")



if __name__ == "__main__":
    main()

