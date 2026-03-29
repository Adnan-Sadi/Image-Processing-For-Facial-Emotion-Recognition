
import cv2
import numpy as np
import dlib

from utils.general import apply_clahe
from utils.face_region_utils import extract_face_region
from config import config


LANDMARK_DICT = {
    "left_lip_corner": 48,
    "right_lip_corner": 54,
    "upper_lip_center": 51,
    "lower_lip_center": 66,
    "left_inner_brow": 21,
    "right_inner_brow": 22,
    "left_outer_brow": 17,
    "right_outer_brow": 26,
    "left_center_brow": 19,
    "right_center_brow": 24,
    "left_upper_eye_lid": 38,
    "right_upper_eye_lid": 44,
    "nose_bridge": 27,
    "jaw_center": 8
}

AU_DICT = {
        "au1": False, # inner brow raiser
        "au2": False, # outer brow raiser
        "au4": False, # brow lowerer
        "au12": False, # lip corner puller (smile)
        "au15": False, # lip corner depressor (frown)
        "au26": False, # jaw drop (mouth open)
    }
    

def landmarks_as_np_array(shape, dtype="int"):
    """Converts the dlib shape object containing the 68 facial landmarks into a numpy array of shape (68, 2)"""
    # initialize zero array with (x,y) coordinates for 68 landmarks
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert to numpy array
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def get_face_landmarks(image, bbox_method=None):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if bbox_method == "custom":
        # apply clahe for better contrast
        eq_image = apply_clahe(image, clip_limit=config.clahe_clip_limit, tile_grid_size=config.clahe_tile_grid_size)
        # extract face region
        _, _, bbox = extract_face_region(eq_image, open_kernel_size=config.open_ksize_fe, close_kernel_size=config.close_ksize_fe)
        x, y, w, h = bbox
        rect = dlib.rectangle(int(x), int(y), int(x + w - 1), int(y + h - 1))
    else:
        # use dlib's default face detector to get bounding box
        face_detector = dlib.get_frontal_face_detector()
        rects = face_detector(img_gray, 1)
        
        if len(rects) > 0:
            rect = rects[0] # take the first detected face
        else:
            return None # return None if no face is detected
        
    # load the facial landmark predictor
    predictor = dlib.shape_predictor(f"{config.dlib_landmark_model}")
    shape = predictor(img_gray, rect)
    shape = landmarks_as_np_array(shape)
    
    return shape



def au_detector(shape, land_dict):
    """Detects the facial action units (AUs) based on the positions of the facial landmarks.
    Args:
        shape: the 68 facial landmarks detected by dlib's shape predictor, in the form of a numpy array of shape (68, 2)
        land_dict: a dictionary mapping the relevant facial landmarks, it contians the pixel(x,y) position for each landmark
    Returns:
        au_dict: a dictionary indicating the presence of specific AUs.
    """

    au_dict = {
        "au1": False, # inner brow raiser
        "au2": False, # outer brow raiser
        "au4": False, # brow lowerer
        "au12": False, # lip corner puller (smile)
        "au15": False, # lip corner depressor (frown)
        "au26": False, # jaw drop (mouth open)
    }

    ####################################################################################
    # analyze the positions of the filtered landmarks to determine the facial expression
    ####################################################################################

    filtered_landmarks = {}

    for key, landmark_index in land_dict.items():
        filtered_landmarks[key] = shape[landmark_index]

    # calculate reference distance to be used for normalization
    nose_bridge = filtered_landmarks["nose_bridge"]
    jaw_center = filtered_landmarks["jaw_center"]
    ref_dist = abs(nose_bridge[1] - jaw_center[1]) 

    ####################################################################################
    # Check Lip Corners and Jaw Drop (au12 = lip corner puller, au15 = lip corner depressor, au26 = jaw drop)
    ####################################################################################

    left_lip_corner = filtered_landmarks["left_lip_corner"]
    right_lip_corner = filtered_landmarks["right_lip_corner"]
    upper_lip_center = filtered_landmarks["upper_lip_center"]
    lower_lip_center = filtered_landmarks["lower_lip_center"]

    # calculate the distances between lip corners and lip centers
    left_lip_to_upper_dist = abs(upper_lip_center[1] - left_lip_corner[1]) # vertical distance from left lip corner to upper lip center
    right_lip_to_upper_dist = abs(upper_lip_center[1] - right_lip_corner[1]) # vertical distance from right lip corner to upper lip center
    left_lip_to_lower_dist = abs(lower_lip_center[1] - left_lip_corner[1]) # vertical distance from left lip corner to lower lip center
    right_lip_to_lower_dist = abs(lower_lip_center[1] - right_lip_corner[1]) # vertical distance from right lip corner to lower lip center
    
    # calculate the overall upper distance 
    upper_distance = (left_lip_to_upper_dist + right_lip_to_upper_dist) / 2
    # calculate the overall lower distance
    lower_distance = (left_lip_to_lower_dist + right_lip_to_lower_dist) / 2

    # threshold for distance between lip corners and lip centers to determine if lips are pulled up or down
    au12_threshold = config.au12_threshold
    au15_threshold = config.au15_threshold

    #print()
    #print(f"normalized difference between lower and upper lip corner distances: {(lower_distance-upper_distance)/ref_dist}")
    #print(f"normalized difference between upper and lower lip corner distances: {(upper_distance-lower_distance)/ref_dist}")
    # lower distance is greater than upper distance, meaning lip corners are likely puller up(au12)
    if (lower_distance-upper_distance)/ref_dist > au12_threshold:
        au_dict["au12"] = True
    # lips are likely to be pulled down(au15)
    elif (upper_distance-lower_distance)/ref_dist > au15_threshold:
        au_dict["au15"] = True
    else:
        au_dict["au12"] = False
        au_dict["au15"] = False
    
    # check jaw drop (au26 = jaw drop)
    # threshold for distance between upper and lower lip centers to determine if jaw is dropped
    au26_threshold = config.au26_threshold

    lip_center_dist = abs(upper_lip_center[1] - lower_lip_center[1]) # vertical distance between upper and lower lip centers

    #print(f"normalized distance between upper and lower lip centers: {lip_center_dist / ref_dist}")
    if lip_center_dist / ref_dist > au26_threshold:
        au_dict["au26"] = True
    else:
        au_dict["au26"] = False


    ####################################################################################
    # Check Eye Brows (au1 = inner brow raiser, au2= outer brow raiser, au4 = brow lowerer)
    ####################################################################################
    left_inner_brow = filtered_landmarks["left_inner_brow"]
    left_outer_brow = filtered_landmarks["left_outer_brow"]
    left_center_brow = filtered_landmarks["left_center_brow"]
    right_inner_brow = filtered_landmarks["right_inner_brow"]
    right_outer_brow = filtered_landmarks["right_outer_brow"]
    right_center_brow = filtered_landmarks["right_center_brow"]
    left_upper_eye_lid = filtered_landmarks["left_upper_eye_lid"]
    right_upper_eye_lid = filtered_landmarks["right_upper_eye_lid"]

    # vertical distance from left inner brow(lib) to left center brow(lcb)
    lib_to_lcb_dist = abs(left_inner_brow[1] - left_center_brow[1])
    # vertical distance from left outer brow(lob) to left center brow(lcb)
    lob_to_lcb_dist = abs(left_outer_brow[1] - left_center_brow[1])
    # vertical distance from right inner brow(rib) to right center brow(rcb)
    rib_to_rcb_dist = abs(right_inner_brow[1] - right_center_brow[1])
    # vertical distance from right outer brow(rob) to right center brow(rcb)
    rob_to_rcb_dist = abs(right_outer_brow[1] - right_center_brow[1])

    au1_threshold = config.au1_threshold
    au2_threshold = config.au2_threshold

    #print(f"normalized distance from inner brows to center brow: left: {lib_to_lcb_dist / ref_dist}, right: {rib_to_rcb_dist / ref_dist}")

    if (lib_to_lcb_dist / ref_dist < au1_threshold or rib_to_rcb_dist / ref_dist < au1_threshold):
        au_dict["au1"] = True
    # if outer brows are raised, the distance between the outer brow and brow center should decrease (lower than a threshold)
    if(lob_to_lcb_dist / ref_dist < au2_threshold or rob_to_rcb_dist / ref_dist < au2_threshold):
        au_dict["au2"] = True

    # vertical distance from left center brow to left upper eye lid
    lcb_to_lue_dist = abs(left_center_brow[1] - left_upper_eye_lid[1])
    # vertical distance from right center brow to right upper eye lid
    rcb_to_rue_dist = abs(right_center_brow[1] - right_upper_eye_lid[1])

    au4_threshold = config.au4_threshold

    #print(f"normalized distance from center brows to upper eye lids: left: {lcb_to_lue_dist / ref_dist}, right: {rcb_to_rue_dist / ref_dist}")

    # if brows are lowered, the distance between the center brow and upper eye lid should decrease (lower than a threshold)    
    if lcb_to_lue_dist / ref_dist < au4_threshold or rcb_to_rue_dist / ref_dist < au4_threshold:
        au_dict["au4"] = True
    else:
        au_dict["au4"] = False

    return au_dict


def au_to_emotion(au_dict):
    """Maps the detected AUs to basic emotions based on common FACS interpretations.
    Args:
        au_dict: a dictionary indicating the presence of specific AUs.
    Returns:
        emotion: the detected emotion label based on the AUs.
    """
    # if smile is detected, then emotion is happy
    if au_dict.get("au12"):
        emotion = "Happy"
    # if jaw drop is detected or both inner brow and outer brow are raised, then surprised
    elif au_dict.get("au26") or (au_dict.get("au1") and au_dict.get("au2")):
        emotion = "Surprised"
    # if frown is detected or either inner brow raise or brow lowered is detected, then sad
    elif au_dict.get("au15") or (au_dict.get("au1") or au_dict.get("au4")):
        emotion = "Sad"
    # default to neutral if no other emotion is detected
    else:
        emotion = "Neutral"
    
    return emotion