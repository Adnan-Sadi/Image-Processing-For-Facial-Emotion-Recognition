# Facial Emotion Recognition — CSC-481 Project

A facial emotion recognition system that classifies emotions from face images using two independent pipelines: a rule-based **FACS classifier** and a **machine learning classifier** (Random Forest / SVM).

---

## Dataset

The dataset set is publicly available on [kaggle](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition), please download the extract the zip file. Next, please **copy the "iamges" folder and place inside it the "data" (which is currently empty) folder** of the project root directly. The images are arranged by subjects and stored in numbered folders (`0`–`18`). Image filenames encode the emotion label (e.g., `happy.jpg`, `sad.jpg`, etc.).

---

## Installation

### 1. Install dependencies
Create a new virutal environment with python 3.12.9 using your virtual environment manager (anaconda, pyenv, or directly with python). A sample python command would be-
```bash
python3 -m venv venv_py312
```
Next activate the environment.

### 2. Install DLib
Dlib is one of the packages that is used in this project, the package requires the [cmake](https://cmake.org/download/) C++ development toolkit to run. Setting up Dlib can be tricky. Here is how i set it up on my MacOS environment-

- Install Homebrew if not already installed.
```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
```bash
brew update
```
- Considering you already have Python Installed. Run the following 3 commands-
```bash
brew install cmake
brew install boost
brew install boost-python --with-python3
```

Again, intalling Dlib can get tricky, so if the above commands doesn't work please check the following blog for [linux and mac](https://pyimagesearch.com/2017/03/27/how-to-install-dlib/), and this blog for [windows](https://www.geeksforgeeks.org/python/how-to-install-dlib-library-for-python-in-windows-10/).

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dlib landmark model
Download [shape_predictor_68_face_landmarks.dat](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) model from dlib github. It will be in a zip format, so extract and place it in `models/landmark_detector/` folder, which is currently empty.

## Run Codes
#### 1. Ensure that you have the dataset images in the `data/images` folder and the dlib landmark model in `models/landmark_detector/` folder.
#### 2. Source file locations and hyperparameters can be adjusted in `config.py` file. However default values are already set up, and the codes should run as long as the dataset and models are in the right places.
#### 3. Run the preprocessing scripts in the following sequence
```bash
# Extract face regions and masks
python extract_face.py

# Generate edge maps from face regions
python extract_edge_maps.py

# Build the feature matrix for the ML pipeline
python extract_dataset_features.py
```
These preprocessing scripts should create the following new folders inside the `data\` directory:
- `face_regions/` : That stores the extracted face region images with background removed.
- `fr_masks/` : That stores the binary masks of the extracted face regions.
- `/edge_maps/canny/` : That stores canny edge mask.
- `/edge_maps/roberts/` : That stores roberts edge mask.
- `/edge_maps/prewitt/` : That stores prewitt edge mask.
- `/edge_maps/sobel/` : That stores sobel edge mask.
- `features/`: that stores the numpy feature matrix saved by `extract_dataset_features.py`, This is used by the ML classifier.


#### 4. Run a classifier
```bash
# FACS rule-based classifier
python facs_based_classifier.py

# ML classifier with fixed hyperparameters
python train.py

# ML classifier with grid search over feature combinations
python train_grid_search.py
```

Results should be saved to the paths specified in `config.py` (`result_save_path` / `ml_result_save_path`). Usually they should be in results folder which already contains results from some my **final runs whihc reported in the project report.**

The ML Classifier The model choice can also be configured in `config.py`.

---

## File & Folder Structure

```
.
├── config.py                   # Central configuration — all paths, hyperparameters, and model settings
├── extract_face.py             # Extracts face regions and binary masks from original images, and saves inside data folder
├── extract_edge_maps.py        # Generate and save edge maps (Canny, Roberts, Prewitt, Sobel) from face regions
├── extract_dataset_features.py # Extract and save HOG + edge features as a compressed .npz file
├── facs_based_classifier.py    # FACS pipeline: rule-based emotion classification using facial landmarks
├── train.py                    # ML pipeline: train Random Forest or SVM with fixed hyperparameters
├── train_grid_search.py        # ML pipeline: train with grid search across feature combinations
├── requirements.txt            # Python package dependencies
│
├── data/
│   ├── emotions.csv            # Subject metadata (set_id, gender, age, country)
│   ├── images/                 # Original input images, organized by subject ID (0–18)
│   │   ├── 0/
│   │   ├── 1/
│   │   └── ...
│   ├── face_regions/           # Extracted face region images (output of extract_face.py)
│   │   ├── 0/
│   │   └── ...
│   ├── fr_masks/               # Binary masks of the extracted face regions (output of extract_face.py)
│   │   ├── 0/
│   │   └── ...
│   ├── edge_maps/              # Edge detection outputs (output of extract_edge_maps.py)
│   │   ├── canny/              # Canny edge maps, per subject
│   │   ├── sobel/              # Sobel edge maps, per subject
│   │   ├── prewitt/            # Prewitt edge maps, per subject
│   │   └── roberts/            # Roberts cross edge maps, per subject
│   └── features/
│       └── dataset_features.npz  # Compressed numpy feature matrix (output of extract_dataset_features.py)
│
├── models/
│   └── landmark_detector/      # dlib's 68-point facial landmark model
│       └── shape_predictor_68_face_landmarks.dat  (download separately)
│
├── results/
│   ├── facs_classifier/
│   │   └── final_1/              # results from one of my final runs        
│   │       ├── facs_results.csv   
│   │       └── metrics.json      
│   └── ml_classifier/
│       ├── final_rf_grid_1/       # results from one of my final runs
│       │   ├── random_forest_best_model.joblib
│       │   ├── ml_results.json
│       │   └── model_config.json
│       └── final_svm_grid_1/      # results from one of my final runs
│           ├── svm_best_model.joblib
│           ├── ml_results.json
│           └── model_config.json
│
├── utils/
│   ├── general.py              # helper functions: read_image, apply_clahe, get_hog_features, plot_images
│   ├── face_region_utils.py    # Face extraction helpers
│   ├── edge_utils.py           # Edge detection helpers
│   └── facs_utils.py           # FACS helpers (landmark extraction, AU detection, AU-to-emotion mapping)
│
└── test_notebooks/             # Jupyter notebooks I used for experimentation and figure generation
    ├── edge_detection.ipynb
    ├── face_landmarks.ipynb
    ├── remove_background.ipynb
    ├── report_figures.ipynb
    └── test_hog.ipynb
```

---

## Key Modules

### `config.py`
Single source of truth for the entire project. Defines:
- `config` — data/model paths, image resize parameters, CLAHE settings, FACS AU thresholds, and ML training parameters (model type, feature selection, result paths).
- `RFConfig` — Random Forest hyperparameters (`n_estimators`, `max_depth`, `criterion`, etc.).
- `SVMConfig` — SVM hyperparameters (`kernel`, `C`, `gamma`).
- `GridSearchConfig` — feature combination grid and hyperparameter search space for `train_grid_search.py`.

### `extract_face.py`
Reads images from `data/images/`, applies CLAHE contrast enhancement, then uses `face_region_utils.extract_face_region` (HSV skin-tone segmentation + morphological opening/closing) to isolate the face. Saves cropped face regions to `data/face_regions/` and binary masks to `data/fr_masks/`.

### `extract_edge_maps.py`
Takes the extracted face regions as input and applies four edge detectors to each image, saving the results to the corresponding subdirectory under `data/edge_maps/`:
- **Canny** — Gaussian blur + Otsu auto-thresholding.
- **Roberts** — 2×2 cross-gradient kernels.
- **Prewitt** — 3×3 horizontal/vertical gradient kernels.
- **Sobel** — 3×3 gradient kernels via `cv2.Sobel`.

### `extract_dataset_features.py`
Aggregates all processed images into a single feature matrix and saves it as `data/features/dataset_features.npz`. Extracted feature types (selectable via `config.features_to_use`):
| Key | Description |
|-----|-------------|
| `region` | Flattened L-channel (LAB) of the face region |
| `mask` | Flattened binary face mask |
| `hog` | HOG descriptors from the face region |
| `canny` | Flattened Canny edge map |
| `roberts` | Flattened Roberts edge map |
| `prewitt` | Flattened Prewitt edge map |
| `sobel` | Flattened Sobel edge map |

### `facs_based_classifier.py`
Rule-based classifier using dlib's 68-point facial landmarks. Detects six Action Units (AUs) based on geometric distances between landmark points, then maps AU combinations to emotions (happy, sad, surprised, neutral, etc.). Results are saved as a CSV and a JSON metrics file.

### `train.py`
Loads `dataset_features.npz`, concatenates the selected feature types, applies `SelectKBest` feature selection, then trains either a `RandomForestClassifier` or an SVM `Pipeline` using 5-fold cross-validation. Reports accuracy, precision, recall, and F1 (macro-averaged).

### `train_grid_search.py`
Extends `train.py` by iterating over all feature combinations defined in `GridSearchConfig` and saving the best-performing model configuration to disk.

---