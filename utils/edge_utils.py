import cv2
import numpy as np
import math

def canny_edge_detection(image, threshold, blur_ksize=5, alpha=0.5, sigma=1.5, size=3, L2gradient=False):
    """
    compute canny edges for the given image.
    - threshold: can be either a tuple of (low_threshold, high_threshold) or a string specifying the auto thresholding method to use.
    - alpha: only used for auto thresholding methods like otsu, median and mean. But not used when direct values are given.
    - blur_ksize: kernel size for Gaussian blur.
    - sigma: standard deviation for Gaussian blur.
    - size: aperture size for the sobel operator used internally by canny edge detection. It must be odd and in the range [3, 7].
    - L2gradient: whether to use L2 norm or not.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(image_gray, ksize=(blur_ksize, blur_ksize), sigmaX=sigma)

    if threshold == 'otsu':
        otsu_threshold_value, _ = cv2.threshold(
            blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        low_threshold = int(max(0, otsu_threshold_value * (1 - alpha)))
        high_threshold = int(min(255, otsu_threshold_value * (1 + alpha)))
    elif threshold == 'median':
        median_value = np.median(blurred_image)
        low_threshold = int(max(0, median_value * (1 - alpha)))
        high_threshold = int(min(255, median_value * (1 + alpha)))
    elif threshold == 'mean':
        mean_value = np.mean(blurred_image)
        low_threshold = int(max(0, mean_value * (1 - alpha)))
        high_threshold = int(min(255, mean_value * (1 + alpha)))
    elif isinstance(threshold, tuple) and len(threshold) == 2:
        # direct values from a tuple
        low_threshold = threshold[0]
        high_threshold = threshold[1]
    else:
        raise ValueError("Invalid threshold value. It must be either a tuple of (low_threshold, high_threshold) or a string specifying the auto thresholding method to use. Available auto thresholding methods are: 'otsu', 'median' and 'mean'.")
        
    print(f"Using Canny thresholds: low = {low_threshold}, high = {high_threshold}")
    out = cv2.Canny(blurred_image, threshold1=low_threshold, threshold2=high_threshold, apertureSize=size, L2gradient=L2gradient)
    return out


def roberts_edge_detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    filter_gx = np.array([[1, 0], 
                        [0, -1]]) 
    filter_gy = np.array([[0, 1], 
                        [-1, 0]]) 

    Gx = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=filter_gx, borderType=cv2.BORDER_CONSTANT)
    Gy = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=filter_gy, borderType=cv2.BORDER_CONSTANT)

    # Compute gradient magnitude to get combined image
    combined_image = cv2.magnitude(Gx, Gy)
    
    # Convert to uint8
    combined_image = cv2.convertScaleAbs(combined_image)

    return combined_image


def prewitt_edge_detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    filter_gx = np.array([[1, 0, -1], 
                          [1, 0, -1], 
                          [1, 0, -1]]) # vertical
    filter_gy = np.array([[1, 1, 1], 
                          [0, 0, 0], 
                          [-1, -1, -1]]) # horizontal

    Gx = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=filter_gx, borderType=cv2.BORDER_CONSTANT)
    Gy = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=filter_gy, borderType=cv2.BORDER_CONSTANT)

    # Compute gradient magnitude to get combined image
    combined_image = cv2.magnitude(Gx, Gy)
    
    # Convert to uint8
    combined_image = cv2.convertScaleAbs(combined_image)

    return combined_image


def sobel_edge_detection(image, kernel_size=3, border_type=cv2.BORDER_CONSTANT):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    Gx = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=kernel_size, borderType=border_type)
    Gy = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=kernel_size, borderType=border_type)

    # Compute gradient magnitude to get combined image
    combined_image = cv2.magnitude(Gx, Gy)
    
    # Convert to uint8
    combined_image = cv2.convertScaleAbs(combined_image)

    return combined_image