import cv2
import os

from utils.general import read_image, apply_clahe
from utils.face_region_utils import extract_face_region
from config import config

def main():
    # get image folders
    source_dir = config.original_images
    destination_dir = config.face_regions # for face regions
    destination_dir_2 = config.face_region_masks # for the binary mask


    # create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    if not os.path.exists(destination_dir_2):
        os.makedirs(destination_dir_2)

    folders = [
        f for f in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, f)) and not f.startswith(".") and f.isdigit()
    ]
    # sorted numerically as the folder names are nnumbers
    
    folders = sorted(folders, key=int)
    print(f"Found folders: {folders}")
    print("=====================================================================")

    for folder in folders:
        folder_path = os.path.join(source_dir, folder)
        if os.path.isdir(folder_path):
            # read all iamge_files in the folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
            print(f"Found image files: {image_files}")
            print("=====================================================================")
            print(f"Processing Images...")

            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                image = read_image(image_path, width=config.width, height=config.height)
                # apply clahe for better contrast
                image = apply_clahe(image, clip_limit=config.clahe_clip_limit, tile_grid_size=config.clahe_tile_grid_size)
                # extract face region
                face_region, mask, _ = extract_face_region(image, open_kernel_size=config.open_ksize_fe, close_kernel_size=config.close_ksize_fe)
                
                # save the extracted face region
                save_path = os.path.join(destination_dir, folder)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                cv2.imwrite(os.path.join(save_path, image_file), cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR))

                # save the binary mask
                save_path_2 = os.path.join(destination_dir_2, folder)
                if not os.path.exists(save_path_2):
                    os.makedirs(save_path_2)

                cv2.imwrite(os.path.join(save_path_2, image_file), mask)

if __name__ == "__main__":
    main()