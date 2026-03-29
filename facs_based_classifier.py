import os
import pandas as pd
import json
import time

from utils.facs_utils import get_face_landmarks, au_detector, au_to_emotion, LANDMARK_DICT, AU_DICT
from utils.general import read_image
from config import config
from sklearn.metrics import classification_report, accuracy_score

def calculate_metrics(df):
    y_true = df["label"]
    y_pred = df["prediction"]

    print("Classification Report:")
    clf_report = classification_report(y_true, y_pred, output_dict=True)

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, clf_report
    

def main():
    start_time = time.time()
    source_dir = config.original_images
    output_path = config.result_save_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    folders = [
        f for f in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, f)) and not f.startswith(".") and f.isdigit()
    ]

    # sorted numerically as the folder names are nnumbers
    folders = sorted(folders, key=int)
    print(f"Found folders: {folders}")
    print("=====================================================================")
    
    # to store all results across folders for final csv
    all_results = []

    print(f"Classifying Images...")

    for folder in folders:
        folder_path = os.path.join(source_dir, folder)
        if os.path.isdir(folder_path):
            # read all iamge_files in the folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

            selected_emotions = ["happy", "sad", "surprised", "neutral"]
            selected_files = [
                f for f in image_files
                if any(emotion in f.lower() for emotion in selected_emotions)
            ]

            print()
            print("=====================================================================")
            print(f"Found image {len(selected_files)} files in folder: {folder}:{selected_files}")
            print("=====================================================================")

            results = []
            for image_file in selected_files:

                image_path = os.path.join(folder_path, image_file)
                # read with the original size first
                image = read_image(image_path, scale_factor=config.scale_factor)

                # remove file extension from image file name and apeend folder name
                label = os.path.splitext(image_file)[0]
                sample_name = f"{folder}_{label}"

                landmarks = get_face_landmarks(image, bbox_method=None)
                # if no face is detected, skip the image and save results with None values for AU and prediction
                if landmarks is None:
                    print(f"No face detected in image: {folder}/{image_file}, Defaulting AU values and prediction to Neutral.")
                    results.append({
                        "image_file": sample_name,
                        **AU_DICT, # default all AU values to False
                        "label": label.lower(),
                        "prediction": "neutral"
                    })
                    continue

                au_dict = au_detector(landmarks, LANDMARK_DICT)
                emotion = au_to_emotion(au_dict)

                results.append({
                    "image_file": sample_name,
                    **au_dict,
                    "label": label.lower(),
                    "prediction": emotion.lower()
                })

            # save results to csv
            all_results.extend(results)

    # save all results across folders to a single csv
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_path, config.result_filename), index=False)

    accuracy, clf_report = calculate_metrics(df_all)
    end_time = time.time()

    hyper_params = {
        "au12_threshold": config.au12_threshold,
        "au15_threshold": config.au15_threshold,
        "au26_threshold": config.au26_threshold,
        "au1_threshold": config.au1_threshold,
        "au2_threshold": config.au2_threshold,
        "au4_threshold": config.au4_threshold,
    }

    # save metrics to a json file
    metrics = {
        "accuracy": accuracy,
        "classification_report": clf_report,
        "execution_time_seconds": end_time - start_time,
        "hyper_parameters": hyper_params
    }

    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()