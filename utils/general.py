import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.feature import hog


def plot_images(images, titles, count_per_row, vmin, vmax, cmap, figsize=(15,12), filename="default", save=False):
    plt.figure(figsize=figsize)

    rows = math.ceil(len(images) / count_per_row)

    for i in range(len(images)):
        # N x N grid
        plt.subplot(rows, count_per_row, i+1)
        plt.imshow(images[i], cmap=cmap[i], vmin=vmin[i], vmax=vmax[i])
        if titles[i] is not None:
            plt.title(titles[i])
        plt.axis("off")
    
    plt.tight_layout()

    if save:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()

def read_image(image_path, width=None, height=None, scale_factor=0.15):    
    """Reads the image from a path"""    
    # loading image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # resize image based on width and height if provided
    if width is not None and height is not None:
        rgb_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)
    else:
        # resize image based on scale factor
        rgb_image = cv2.resize(rgb_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    return rgb_image

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies CLAHE to the input image."""
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Create a CLAHE object and apply it to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_l_channel = clahe.apply(l_channel)
    
    # Merge the channels back together
    merged_lab_image = cv2.merge((cl_l_channel, a_channel, b_channel))
    
    # Convert back to RGB color space
    enhanced_image = cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2RGB)
    
    return enhanced_image

def apply_bilateral_filter(image, d=5, sigma_color=75, sigma_space=75, border_type=cv2.BORDER_CONSTANT):
    """Applies bilateral filter to the input image. Good for noise reduction while preserving edges."""
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space, borderType=border_type)


def get_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, feature_vector=True):
    """
    Extract HOG features from the input image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    
    if visualize:
        hog_features, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                      cells_per_block=cells_per_block,
                                      visualize=visualize, feature_vector=feature_vector)
        return hog_features, hog_image

    else:
        hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           visualize=visualize, feature_vector=feature_vector)
        return hog_features, None