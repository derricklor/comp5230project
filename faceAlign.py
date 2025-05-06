import cv2
import numpy as np
import dlib
from imutils import face_utils


def align_face(image, gray, landmarks, desired_face_width=200, desired_left_eye=(0.25, 0.4)):
    """
    Aligns a face image based on detected facial landmarks.

    Args:
        image (numpy.ndarray): The input color image (BGR format).
        gray (numpy.ndarray): The grayscale version of the input image.
        landmarks (numpy.ndarray):  Array of (x, y) coordinates of facial landmarks.
        desired_face_width (int, optional): The desired width of the output face image. Defaults to 200.
        desired_left_eye (tuple, optional): The desired position of the left eye
            in the output image (as a fraction of the image size). Defaults to (0.3, 0.45).

    Returns:
        numpy.ndarray: The aligned face image (color, BGR format).  Returns None on error.
    """
    # Get key facial features
    left_eye_center = landmarks[36:42].mean(axis=0).astype("int")
    right_eye_center = landmarks[42:48].mean(axis=0).astype("int")
    #nose_tip = landmarks[30].astype("int")
    #mouth_center = landmarks[48:68].mean(axis=0).astype("int")

    # Calculate the rotation angle
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) #- 180

    # Calculate the desired right eye position
    desired_right_eye_x = 1.0 - desired_left_eye[0]

    # Determine the scale
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0]) * desired_face_width
    scale = desired_dist / dist

    # Compute the center point between the eyes
    eyes_center = ((int(left_eye_center[0] + right_eye_center[0])) // 2,
                   (int(left_eye_center[1] + right_eye_center[1])) // 2)

    # Build the transformation matrix
    M = cv2.getRotationMatrix2D(center=eyes_center, angle=angle, scale=scale)

    # Update the translation component of the matrix to ensure the left eye
    # is at the desired position in the output image
    tX = desired_face_width * 0.5 #desired_left_eye[0]
    tY = desired_face_width * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    # Apply the affine transformation to align and resize the face
    (w, h) = (desired_face_width, int(desired_face_width*1.4))
    try:
        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        return aligned_face
    except Exception as e:
        print(f"Error aligning face: {e}")
        return None


def align_cropped_face(image):
    """
    Aligns a cropped face image.  This function assumes the input image
    contains a single, cropped face.

    Args:
        image (numpy.ndarray): The cropped face image (BGR format).

    Returns:
        numpy.ndarray: The aligned face image (color, BGR format). Returns None on error.
    """
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from dlib

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the face.  Since we assume it's a cropped image, we expect one face only.
    rects = detector(gray, 1)  #detect the face

    if len(rects) != 1:
        print("Error: Expected one face in the cropped image, found {}".format(len(rects)))
        return None

    rect = rects[0] #get the bounding box

    # Get the facial landmarks
    try:
        shape = predictor(gray, rect)
        landmarks = face_utils.shape_to_np(shape)  # Convert to NumPy array
    except Exception as e:
        print(f"Error getting landmarks: {e}")
        return None

    # Align the face using the landmarks
    aligned_face = align_face(image, gray, landmarks)
    return aligned_face


def align_face_from_file(image_path):
    """
    Aligns a face image given its file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: The aligned face image (color, BGR format). Returns None on error.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    return align_cropped_face(image) #use the existing function


def align_face_from_frame(frame):
    """
    Aligns a face image given a frame (numpy.ndarray).

    Args:
        frame (numpy.ndarray): The image frame (BGR format).
        bbox tensor [x,y,w,h, device=0]: The bounding box for the ROI Region of Interest.

    Returns:
        numpy.ndarray: The aligned face image (color, BGR format). Returns None on error.
    """
    #x, y, w, h = bbox
    #x, y, w, h = map(int, bbox) # unpacks the tensor [x,y,w,h, device=0]
    # Extract the ROI from the frame
    #roi = frame[y:y+h, x:x+w]
    return align_cropped_face(frame)


if __name__ == '__main__':
    # Example usage:
    # Replace with the path to your cropped face image
    image_path = [
        r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\curated faces\1-1276-_jpg.rf.fd0f982dbb0fca9ab4c5a6506807dc29.jpg"
        ,r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\images\1-318-_jpg.rf.47778bbd5a1c99b51e71d36469cc9b0b.jpg"
        ,r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\whitelist\face\Josh.jpg"
        ,r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\blacklist\face\Tim.jpg"
        ,r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\whitelist\face\Derrick.jpg"
        ,r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\whitelist\face\Ben.jpg"
    ]
    cropped_face_image = cv2.imread(image_path[3])
    if cropped_face_image is not None:
        aligned_face = align_cropped_face(cropped_face_image)
        if aligned_face is not None:
            cv2.imshow("Aligned Face", aligned_face)
            print(aligned_face)
            print(len(aligned_face))
            print(f'height: {len(aligned_face)}')
            print(f'width: {len(aligned_face[0])}')
            print(f'channel: {len(aligned_face[0][0])}')
            cv2.waitKey(0)
            #cv2.imwrite(r"C:\Users\Derrick\Documents\School\Computer Vision\project\known_faces\whitelist\face\aligned_face2.jpg", aligned_face)
            cv2.destroyAllWindows()
        else:
            print("Face alignment failed.")
    else:
        print(f"Error: Could not read image at {image_path}")
