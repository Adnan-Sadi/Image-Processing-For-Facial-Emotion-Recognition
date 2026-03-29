from typing import Optional

# global random state value
RANDOM_STATE = 42

class config:
    # data paths
    original_images = "data/images" # path to the original images
    face_regions = "data/face_regions" # path to the extracted face regions
    face_region_masks = "data/fr_masks" # path to the binary masks of the extracted face regions
    canny_edge_maps = "data/edge_maps/canny" # path to the canny edge maps
    roberts_edge_maps = "data/edge_maps/roberts" # path to the roberts edge maps
    prewitt_edge_maps = "data/edge_maps/prewitt" # path to the prewitt edge maps
    sobel_edge_maps = "data/edge_maps/sobel" # path to the sobel edge maps
    feature_dir = "data/features" # path to save the extracted features in npz format (for ML pipeline)

    # model paths
    dlib_landmark_model = "models/landmark_detector/shape_predictor_68_face_landmarks.dat" # path to dlib's 68 face landmark model

    # resize parameters (used by the read_image function in utils/general.py)
    width = 388 #  resized width of the images
    height = 518 # resized height of the images
    scale_factor = 0.15 # scale factor for resizing, instead of using fixed width and height

    #Clahe parameters
    clahe_clip_limit = 2.0 
    clahe_tile_grid_size = (8, 8)

    # face extraction parameters
    open_ksize_fe = (7, 7) # kernel size opening operation in face extraction
    close_ksize_fe = (7, 7) # kernel size closing operation in face extraction

    ##################################
    # FACS Classifier Hyperparameters
    ##################################

    au12_threshold = 0.03 # vertical distance between lip corners and upper lip center, higher than threshold = au12 (lip corner puller up)
    au15_threshold = 0.02 # vertical distance between lip corners and lower lip center, higher than threshold = au15 (lip corner puller down)
    au26_threshold = 0.08 # vertical distance between upper and lower lip centers, higher than threshold = au26 (jaw drop)
    au1_threshold = 0.6 # vertical distance between inner brow and center brow, lower than theshold = au1 (inner brow raiser)
    au2_threshold = 0.10 # vertical distance between outer brow and center brow lower than theshold = au2 (outer brow raiser)
    au4_threshold = 0.10 # vertical distance between center brow and upper eye lid, lower than theshold = au4 (brow lowerer)

    result_save_path = "results/facs_classifier/final_1" # path to save the FACS classifier results
    result_filename = "facs_results.csv" # filename to save the FACS classifier results (csv format)



    ##################################
    # ML Classifier Hyperparameters
    ##################################

    # feature extraction parameters
    input_size = (64, 64) # input size for ML classifier
    hog_input_size = (64, 128) # input size for HOG feature extraction
    hog_orientations = 9
    hog_pixels_per_cell = (8, 8)
    hog_cells_per_block = (2, 2)

    # training parameters
    model="random_forest" # "random_forest", "svm"
    features_to_use = ["hog", "canny"] # all option: "region", "mask", "canny", "roberts", "prewitt", "sobel"
    num_features_to_select = 2000 # number of top features to select using SelectKBest
    ml_result_save_path = "results/ml_classifier/final_rf_grid_1" # path to save the ML classifier results
    ml_result_filename = "ml_results.json" # filename to save the ML classifier results (json format)

# random forest hyperparameters
class RFConfig:
    def __init__(self):
        self.n_estimators: int = 200
        self.max_features: Optional[float] | str = "sqrt"
        self.max_depth: Optional[int] = None
        self.min_samples_split: int = 3
        self.criterion: str = "entropy"
        self.random_state: int = RANDOM_STATE

# svm hyperparameters
class SVMConfig:
    def __init__(self):
        self.kernel: str = "linear"       # "rbf", "linear", "poly", "sigmoid"
        self.C: float = 10.0           # regularization strength
        self.gamma: str = "scale"      # "scale", "auto", or a float
        self.random_state: int = RANDOM_STATE

# grid search hyperparameters
class GridSearchConfig:
    features_combinations = [
        ["hog"],
        ["sobel"],
        ["roberts"],
        ["canny"],
        ["prewitt"],
        ["region"],
        ["hog", "region"],
        ["hog", "roberts"],
        ["hog", "canny"],
        ["hog", "prewitt"],
        ["hog", "sobel"],
        ["hog", "region", "roberts"],
        ["hog", "region", "canny"],
        ["hog", "region", "prewitt"],
        ["hog", "region", "sobel"],
        ["hog", "region", "roberts", "sobel"],
        ["hog", "region", "canny", "prewitt", "sobel", "roberts"],
    ]
    num_features_to_select_combinations = [500, 800, 1000, 2000, 3000]