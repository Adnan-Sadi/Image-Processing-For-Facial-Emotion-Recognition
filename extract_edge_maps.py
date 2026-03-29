import cv2
import os

from utils.general import read_image
from utils.face_region_utils import extract_face_region
from utils.edge_utils import canny_edge_detection, roberts_edge_detection, prewitt_edge_detection, sobel_edge_detection
from config import config

def main():
    # get image folders
    source_dir = config.face_regions # use the extracted face regions as input for edge detection
    canny_dir = config.canny_edge_maps # for canny edge maps
    roberts_dir = config.roberts_edge_maps # for roberts edge maps
    prewitt_dir = config.prewitt_edge_maps # for prewitt edge maps
    sobel_dir = config.sobel_edge_maps # for sobel edge maps


    # create destination directory if it doesn't exist
    if not os.path.exists(canny_dir):
        os.makedirs(canny_dir)
    if not os.path.exists(roberts_dir):
        os.makedirs(roberts_dir)
    if not os.path.exists(prewitt_dir):
        os.makedirs(prewitt_dir)
    if not os.path.exists(sobel_dir):
        os.makedirs(sobel_dir)

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
                image = read_image(image_path, scale_factor=1.0) # no resizing needed here
                
                # save the edge maps
                canny_edges = canny_edge_detection(image, threshold="otsu", blur_ksize=5, alpha=0.33, sigma=1.5, size=3, L2gradient=False)
                roberts_edges = roberts_edge_detection(image)
                prewitt_edges = prewitt_edge_detection(image)
                sobel_edges = sobel_edge_detection(image, kernel_size=3, border_type=cv2.BORDER_CONSTANT)

                # save the edge maps
                save_path_canny = os.path.join(canny_dir, folder)
                if not os.path.exists(save_path_canny):
                    os.makedirs(save_path_canny)

                cv2.imwrite(os.path.join(save_path_canny, image_file), canny_edges)

                save_path_roberts = os.path.join(roberts_dir, folder)
                if not os.path.exists(save_path_roberts):
                    os.makedirs(save_path_roberts)

                cv2.imwrite(os.path.join(save_path_roberts, image_file), roberts_edges)

                save_path_prewitt = os.path.join(prewitt_dir, folder)
                if not os.path.exists(save_path_prewitt):
                    os.makedirs(save_path_prewitt)

                cv2.imwrite(os.path.join(save_path_prewitt, image_file), prewitt_edges)

                save_path_sobel = os.path.join(sobel_dir, folder)
                if not os.path.exists(save_path_sobel):
                    os.makedirs(save_path_sobel)

                cv2.imwrite(os.path.join(save_path_sobel, image_file), sobel_edges)

if __name__ == "__main__":
    main()