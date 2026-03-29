import cv2
import os
import numpy as np

from utils.general import read_image, apply_clahe, get_hog_features
from utils.face_region_utils import extract_face_region, crop_and_resize_face
from config import config

def extract_face_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # get the Lightness (L) channel
    L_channel, _, _ = cv2.split(image)
    # flatten the L channel to a 1D array
    features = L_channel.flatten()
    # normalize the pixel values to [0,1]
    features = features.astype(np.float32) / 255.0
    return features

def main():
    # processed images
    original_images_dir = config.original_images
    face_region_dir = config.face_regions
    face_mask_dir = config.face_region_masks
    canny_dir = config.canny_edge_maps
    roberts_dir = config.roberts_edge_maps
    prewitt_dir = config.prewitt_edge_maps
    sobel_dir = config.sobel_edge_maps
    
    folders = [
        f for f in os.listdir(face_region_dir)
        if os.path.isdir(os.path.join(face_region_dir, f)) and not f.startswith(".") and f.isdigit()
    ]
    # sorted numerically as the folder names are numbers
    folders = sorted(folders, key=int)
    print(f"Found folders: {folders}")
    print("=====================================================================")

    # store features for the entire dataset
    file_names = []
    labels = []

    region_features_all = []
    mask_features_all = []
    hog_features_all = []
    canny_features_all = []
    roberts_features_all = []
    prewitt_features_all = []
    sobel_features_all = []

    print()
    print("Extracting features from images...")

    for folder in folders:
        original_folder_path = os.path.join(original_images_dir, folder)
        face_folder_path = os.path.join(face_region_dir, folder)
        mask_folder_path = os.path.join(face_mask_dir, folder)
        canny_folder_path = os.path.join(canny_dir, folder)
        roberts_folder_path = os.path.join(roberts_dir, folder)
        prewitt_folder_path = os.path.join(prewitt_dir, folder)
        sobel_folder_path = os.path.join(sobel_dir, folder)

        image_files = [
            f for f in os.listdir(face_folder_path)
            if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')
        ]

        for image_file in image_files:
            # read face image and extract features
            face_image_path = os.path.join(face_folder_path, image_file)
            face_image = read_image(face_image_path, width=config.input_size[0], height=config.input_size[1])
            face_features = extract_face_features(face_image)

            # read face mask and extract features
            mask_image_path = os.path.join(mask_folder_path, image_file)
            mask_image = read_image(mask_image_path, width=config.input_size[0], height=config.input_size[1])
            mask_features = mask_image.flatten().astype(np.float32) / 255.0

            # read image and extract HOG features
            original_image_path = os.path.join(original_folder_path, image_file)
            original_image = read_image(original_image_path, scale_factor=1.0)
            original_image = apply_clahe(original_image)
            face, mask, bbox = extract_face_region(original_image)
            resized_face = crop_and_resize_face(original_image, bbox, output_size=config.hog_input_size)
  
            hog_features, _ = get_hog_features(
                            image=resized_face, 
                            orientations=config.hog_orientations,
                            pixels_per_cell=config.hog_pixels_per_cell,
                            cells_per_block=config.hog_cells_per_block,
                            visualize=False,
                            feature_vector=True
                            )

            # read edge maps and extract features
            # canny
            canny_image_path = os.path.join(canny_folder_path, image_file)
            canny_image = read_image(canny_image_path, width=config.input_size[0], height=config.input_size[1])
            canny_image = cv2.cvtColor(canny_image, cv2.COLOR_RGB2GRAY)
            canny_features = canny_image.flatten().astype(np.float32) / 255.0

            # roberts
            roberts_image_path = os.path.join(roberts_folder_path, image_file)
            roberts_image = read_image(roberts_image_path, width=config.input_size[0], height=config.input_size[1])
            roberts_image = cv2.cvtColor(roberts_image, cv2.COLOR_RGB2GRAY)
            roberts_features = roberts_image.flatten().astype(np.float32) / 255.0

            # prewitt
            prewitt_image_path = os.path.join(prewitt_folder_path, image_file)
            prewitt_image = read_image(prewitt_image_path, width=config.input_size[0], height=config.input_size[1])
            prewitt_image = cv2.cvtColor(prewitt_image, cv2.COLOR_RGB2GRAY)
            prewitt_features = prewitt_image.flatten().astype(np.float32) / 255.0

            # sobel
            sobel_image_path = os.path.join(sobel_folder_path, image_file)
            sobel_image = read_image(sobel_image_path, width=config.input_size[0], height=config.input_size[1])
            sobel_image = cv2.cvtColor(sobel_image, cv2.COLOR_RGB2GRAY)
            sobel_features = sobel_image.flatten().astype(np.float32) / 255.0

            label = os.path.splitext(image_file)[0].lower()

            file_names.append(f"{folder}/{image_file}")
            labels.append(label)

            region_features_all.append(face_features)
            mask_features_all.append(mask_features)
            canny_features_all.append(canny_features)
            roberts_features_all.append(roberts_features)
            prewitt_features_all.append(prewitt_features)
            sobel_features_all.append(sobel_features)
            hog_features_all.append(hog_features)

    # convert lists to numpy arrays
    file_names = np.array(file_names)
    labels = np.array(labels)

    region_features_all = np.array(region_features_all, dtype=np.float32)
    mask_features_all = np.array(mask_features_all, dtype=np.float32)
    canny_features_all = np.array(canny_features_all, dtype=np.float32)
    roberts_features_all = np.array(roberts_features_all, dtype=np.float32)
    prewitt_features_all = np.array(prewitt_features_all, dtype=np.float32)
    sobel_features_all = np.array(sobel_features_all, dtype=np.float32)
    hog_features_all = np.array(hog_features_all, dtype=np.float32)

    if not os.path.exists(config.feature_dir):
        os.makedirs(config.feature_dir)

    output_path = os.path.join(config.feature_dir, "dataset_features.npz")

    np.savez_compressed(
        output_path,
        file_names=file_names,
        labels=labels,
        region=region_features_all,
        mask=mask_features_all,
        canny=canny_features_all,
        roberts=roberts_features_all,
        prewitt=prewitt_features_all,
        sobel=sobel_features_all,
        hog=hog_features_all,
    )

    print(f"Feature extraction completed. Features saved to: {output_path}")
    print(f"region shape:{region_features_all.shape}")
    print(f"mask shape:{mask_features_all.shape}")
    print(f"canny shape:{canny_features_all.shape}")
    print(f"roberts shape:{roberts_features_all.shape}")
    print(f"prewitt shape:{prewitt_features_all.shape}")
    print(f"sobel shape:{sobel_features_all.shape}")
    print(f"hog shape:{hog_features_all.shape}")
    print(f"labels shape:{labels.shape}")

if __name__ == "__main__":
    main()