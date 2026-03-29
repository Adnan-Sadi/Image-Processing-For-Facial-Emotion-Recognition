import cv2
import numpy as np

def get_face_bounding_box(mask):
    """Returns a rectangular bounding box of the face region from the mask"""
    # get coordinates of the non-zero pixels in the mask
    non_zero = cv2.findNonZero(mask)

    bbox = None
    if non_zero is not None:
        # boundingRect returns the smallest possible reactangle that contains 
        # all the non-zero points in the mask
        x, y, w, h = cv2.boundingRect(non_zero)
        bbox = (x, y, w, h)

    return bbox

def extract_face_region(
    image, 
    lower_skin_1=[0, 20, 100],
    upper_skin_1=[10, 255, 255],
    lower_skin_2=[170, 20, 100],
    upper_skin_2=[180, 255, 255],
    open_kernel_size=(7, 7),
    close_kernel_size=(7, 7),
    ):
    """Extracts the face region from the image by skin tone and morphological operations"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_skin_1 = np.array(lower_skin_1, dtype=np.uint8)
    upper_skin_1 = np.array(upper_skin_1, dtype=np.uint8)

    # to capture reddish skin tones that wrap around the hue spectrum
    lower_skin_2 = np.array(lower_skin_2, dtype=np.uint8)
    upper_skin_2 = np.array(upper_skin_2, dtype=np.uint8)

    mask_1 = cv2.inRange(hsv, lower_skin_1, upper_skin_1)
    mask_2 = cv2.inRange(hsv, lower_skin_2, upper_skin_2)
    mask = cv2.bitwise_or(mask_1, mask_2)
    # print(f"Mask mean value: {mask.mean()}, min: {mask.min()}, max: {mask.max()}")

    # calculate morphological operation iterations automatically based on assumption
    total_pixels = mask.shape[0] * mask.shape[1]
    face_pixels = np.sum(mask == 255)
    face_pixel_percentage = max((face_pixels / total_pixels) * 100, 0.01) # added a small value to avoid 0
    #print(f"Face pixel percentage: {face_pixel_percentage:.2f}%")

    expected_face_percentage = 60
    base_open_iterations = 3
    base_close_iterations = 6
    
    # more opening operations if the face pixel percentage is higher than expected
    open_iterations = max(1, round((face_pixel_percentage / expected_face_percentage) * base_open_iterations))
    # more closing operations if the face pixel percentage is lower than expected
    close_iterations = max(1, round((expected_face_percentage / face_pixel_percentage) * base_close_iterations))

    # morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    
    # generally used a larger kernel for the closing operation
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=close_iterations)    
    
    # apply the mask to the original image (here source and destination are the same)
    face_region = cv2.bitwise_and(image, image, mask=mask)
    bbox = get_face_bounding_box(mask)

    return face_region, mask, bbox

def crop_and_resize_face(image, bbox, output_size=(32, 64)):
    """
    Crops the face region from the image using the bounding box and resizes it.
    """
    x, y, w, h = bbox
    # extract the face using array slicing
    face_crop = image[y:y + h, x:x + w]
    resized_face = cv2.resize(face_crop, output_size, interpolation=cv2.INTER_LINEAR)
    return resized_face
