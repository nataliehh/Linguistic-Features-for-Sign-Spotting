import itertools
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# MEDIAPIPE
# Mediapipe models and drawing styles
ms = mp.solutions
mp_drawing, mp_ds, mp_holistic, mp_hands, mp_face_mesh = (ms.drawing_utils, ms.drawing_styles, ms.holistic, ms.hands, ms.face_mesh)
drawing_spec = mp_drawing.DrawingSpec(thickness=0.5, circle_radius=0.5)

# Getting the shoulder indices from the mediapipe pose model
# We subtract 1 from it because these landmark indices start at 1 instead of 0
l_shoulder_ind = mp_holistic.PoseLandmark.LEFT_SHOULDER.value - 1
r_shoulder_ind = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value - 1

# Can change min_detection_confidence, min_tracking_confidence (both default to 0.5)
# Model complexity of holistic model set to 2 for more fine-grained (but slower) estimations
holistic = mp_holistic.Holistic(model_complexity=2)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1) # ,refine_landmarks=True # Adds 10 extra landmarks for irises

# Drawing specs
nose_style = mp_ds.DrawingSpec(color=(224, 224, 224), thickness=2, circle_radius=2)
 
'''
   ---Show the captured frame, either in a new window (cv2) or as an inline plot (pyplot)---
   img:      an RGB image
   title:    a title to display with the plot (str)
   plot_plt: whether we're using pyplot instead of cv2 to plot the image (bool)
   
'''
def show(img, title:str = '', plot_plt:bool = False): 
    if plot_plt: # Use pyplot to show image
        plt.imshow(img)
        plt.show()
        return
    # Use cv2 to display image (in new window)
    cv2.imshow(title, img)

'''
    ---Draw the landmarks of mediapipe (simply used to abbreviate some code later)---
    img:                     an RGB image
    landmarks:               a list of Mediapipe landmarks, e.g. of the hand and face
    connections:             a list specifying how the landmarks should be connected
    landmark_drawing_spec:   specifies the drawing style of the landmarks (e.g. color, thickness)
    connection_drawing_spec: specifies the drawing style of the connections (e.g. color, thickness)
    connection:              whether the thing to draw is a landmark or a connection
'''
def draw_landmarks(img, landmarks, connections, style, connection:bool = True):
    if connection:
        mp_drawing.draw_landmarks(img, landmarks, connections, landmark_drawing_spec=None, connection_drawing_spec=style)
    else:
        mp_drawing.draw_landmarks(img, landmarks, connections, landmark_drawing_spec=style)
        

'''
    ---Process the current frame using the specified methods---
    img:     an RGB image
    methods: a list of mediapipe methods (e.g. face, hand detection) with which to process the image.
    -----------------------------------------------------------
    return: the results of the processing with the mediapipe methods and the image
'''
def process_frame(img, methods:list = []):
    # Convert to RGB, flip image and set to not-writable (improves performance)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False

    # Detections (looping through detection models)
    results = []
    for method in methods:
        results.append(method.process(img))

    # Set writable back to true
    img.flags.writeable = True

    # RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return results, img
 
'''
    ---Convert the mediapipe landmark format to something that's easier to process (i.e. a list format)---
    landmarks:     a collection of mediapipe landmarks for a specific region (body, face or hands)
    num_keypoints: the number of keypoints that the region typically contains (e.g. 21 for the hands) (int)
    ------------------------------------------------------------------------------------------------------
    return: the (x, y) coordinates, for each mediapipe landmark, as a list
'''
def get_landmark_xy(landmarks, num_keypoints:int):
    if landmarks: # If the landmarks aren't 'None'
        xy = []
        if type(landmarks) != list: 
            # Get the x, y coordinates of the landmarks
            xy = [[l.x, l.y] for l in landmarks.landmark]
        else:
            for landmark in landmarks: # For faces, we always get a list 
                xy+= [[l.x, l.y] for l in landmark.landmark]
        return xy
    else: # If the landmarks are empty, return a list of (x,y) coordinates that are NaN
        # We use num_keypoints to make the number of (NaN, NaN) coordinates the same as for the non-NaN landmarks
        return [[np.nan, np.nan]]*num_keypoints
        
'''
    ---Process the video, obtaining the landmarks of the hands, pose and face in a dictionary---
    video_path: the path to the video to be processed (str)
    cngt_vid:   whether the video is part of the Corpus NGT (and thus should be processed differently) (bool)
    draw:       if the video should be shown while being processed, with landmarks (bool)
    only_face:  whether to only process the face landmarks or not (bool)
    frame_skip: how many frames to skip in the processing, useful if we want to visually inspect a specific part (int)
    --------------------------------------------------------------------------------------------
    return: the landmarks of the hands, pose and face as well as the width and height of the videos
'''
def process_video(video_path:str, cngt_vid:bool, draw:bool = False, only_face:bool = False, frame_skip: int = 0):
    landmark_areas = ['face'] if only_face else ['l_hand', 'r_hand', 'pose', 'face']
    cap = cv2.VideoCapture(video_path)
    
    # Get the video's dimensions (width x height) to store for later
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Get the number of frames and initialize a frame counter
    no_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a dictionary to store the landmarks in
    landmarks = {}
    # Instantiate as empty (will have 21, 21, 33, 468 keypoints respectively)
    for key in landmark_areas:
        landmarks[key] = []
    
    # Additionally, store the video dimensions in the dictionary
    landmarks['video_dim'] = (w,h)
    
    # Make a dictionary with the number of keypoints for each region of the landmarks (e.g. face, hands)
    # 'face' key currently doesn't output keypoints with the irises
    num_keypoints = {'face' : len(get_keypoints_from_connections(mp_face_mesh.FACEMESH_TESSELATION)), 
                    'l_hand': len(get_keypoints_from_connections(mp_hands.HAND_CONNECTIONS)), 
                    'r_hand': len(get_keypoints_from_connections(mp_hands.HAND_CONNECTIONS)),
                    'pose': len(get_keypoints_from_connections(mp_holistic.POSE_CONNECTIONS))}

    # I created a custom list for the landmark connections of the nose
    # Because MediaPipe doesn't show the nose by itself
    nose_connections = get_nose_connections()

    while cap.isOpened():
        success, img = cap.read() # Read frame
        no_frames +=1 # Increase frame counter
        print('frames: {}/{}'.format(no_frames, total_frames), end = '\r')

        # Skip the first 3 seconds (25 frames/sec = 75 frames) to save some processing
        # Since they are just a disclaimer at the start of all CNGT videos
        if cngt_vid and no_frames <= 75 + frame_skip:
            continue
            
        # Skip the last 1 second (25 frames/sec = 25 frames)
        # Because the disclaimer from the start repeats
        if cngt_vid and no_frames + 25 >= total_frames:
            break

        # This checks if there are still frames to process, otherwise we stop
        if not success:
            # If loading a video, use 'break', for webcam use 'continue'
            break

        # Resize image for larger display (and proportionately smaller landmarks later)
        # img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)

        # Process the frame, given which models to use
        results, img = process_frame(img, [holistic])
        r = results[0]  # Get results from list
        if only_face:
            r_dict = {'face':r.face_landmarks}
        else:
            r_dict = {'l_hand':r.left_hand_landmarks,'r_hand':r.left_hand_landmarks,'pose':r.pose_landmarks,'face':r.face_landmarks}

        # Store the landmarks in the dictionary
        for key in landmarks:
            if not 'dim' in key:
                landmarks[key].append(get_landmark_xy(r_dict[key], num_keypoints[key]))

        # Drawing landmark annotation on frame
        if draw:
            if not only_face:
                draw_landmarks(img, r.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_ds.get_default_pose_landmarks_style(), False)
                draw_landmarks(img,r.right_hand_landmarks,mp_hands.HAND_CONNECTIONS, mp_ds.get_default_hand_landmarks_style(), False)
                draw_landmarks(img,r.left_hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_ds.get_default_hand_landmarks_style(), False)
            draw_landmarks(img, r.face_landmarks, nose_connections, nose_style)
            draw_landmarks(img, r.face_landmarks, mp_holistic.FACEMESH_CONTOURS, mp_ds.get_default_face_mesh_contours_style())
            
            # Show horizontally flipped image (otherwise output is mirrored)
            show(cv2.flip(img, 1), 'MediaPipe Holistic + Face Mesh')

        # Pressing 'esc' will stop the video from being processed further
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    if draw:
        cv2.destroyAllWindows()
    return landmarks

        
'''
    ---Process the video, obtaining the landmarks only the face---
    video_path: the path to the video to be processed (str)
    cngt_vid:   whether the video is part of the Corpus NGT (and thus should be processed differently) (bool)
    draw:       if the video should be shown while being processed, with landmarks (bool)
    --------------------------------------------------------------------------------------------
    return: the landmarks of only the face as well as the width and height of the videos
'''
def process_face(video_path:str, cngt_vid:bool, draw:bool = False):    
    return process_video(video_path, cngt_vid, draw, only_face = True)

'''
    ---Get a fixed list of connections of the points around the nose---
    -------------------------------------------------------------------
    return: a fixed list of connections of the points around the nose
'''
def get_nose_connections():
    # Get the list of the tesselation connections (tuples in a list)
    facemesh_tesselation_tuples = list(mp_face_mesh.FACEMESH_TESSELATION)
    
    # Identify a potential list of nose points, based on:
    # https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    nose_points = [48, 64, 240, 99, 20, 242, 141, 94, 370, 462, 250, 328, 460, 294, 278]
    
    # Get any tuples in the tesselation connection tuples which contain these nose-points
    facemesh_nose_tuples = [x for x in facemesh_tesselation_tuples if x[0] in nose_points or x[1] in nose_points]  
    return frozenset(facemesh_nose_tuples)

'''
    ---Get the keypoints from a list of connections---
    connections: a list of keypoint connections
    --------------------------------------------------
    return: a list of keypoints from the list of connections
'''
def get_keypoints_from_connections(connections):
    # Convert to list, then 'flatten' the tuples: [(1,2), (3,4)] -> [1,2,3,4]
    connection_tuples = list(connections)
    keypoints = list(itertools.chain(*connection_tuples))
    # Get only the unique instances of keypoints
    unique_keypoints = list(set(keypoints))
    return unique_keypoints  

'''
    ---Get the normalisers for the landmarks ---
    coordinates: a dictionary of all the coordinates extracted with mediapipe (dict)
    --------------------------------------------
    return: two lists of normalisers for the landmarks
'''
def get_normalisers(coordinates:dict):
    norms1 = [] # shoulder_L - shoulder_R for every frame
    norms2 = [] # shoulder_mid (normalised by norm1) for every frame
    for pose in coordinates['pose']:
        # Retrieve the shoulder coordinates
        shoulder_l = np.array(pose[l_shoulder_ind]) # pose 11 = index 10?
        shoulder_r = np.array(pose[r_shoulder_ind]) # pose 12 = index 11?
        hip_l = np.array(pose[23])
        
        # Normalisation 1: divide all points by the distance between the shoulders
        norm1_x = abs(shoulder_l[0] - shoulder_r[0])
        norm1_y = abs(shoulder_l[1] - hip_l[1])
        norm1 = np.array([norm1_x, norm1_y])
        
        # Normalisation 2: subtract the midpoint of the shoulders from all points
        norm2 = abs(shoulder_l/norm1 + shoulder_r/norm1)/2
        
        # The normalisations are per frame, so make a list of all the norms
        norms1.append(list(norm1))
        norms2.append(list(norm2))
    return norms1, norms2 

'''
    ---Get the normalisers for the only-face landmarks ---
    coordinates: a dictionary of all the face coordinates extracted with mediapipe (dict)
    --------------------------------------------
    return: two lists of normalisers for the only-face landmarks
'''
def get_face_normalisers(coordinates:dict):
    norms1 = [] # head_leftmost - head_rightmost for every frame
    norms2 = [] # nose (normalised by norm1) for every frame
    norms3 = [] # head_topmost - head_bottommost for every frame
    for face in coordinates['face']: # 93 323
        # Retrieve the head extremity coordinates (taken from facemesh)
        head_leftmost = np.array(face[323]) 
        head_rightmost = np.array(face[93]) 
        head_topmost = np.array(face[10]) 
        head_bottommost = np.array(face[152]) 
        
        # Normalisation 1 + 3: divide all points by the width * height of the face
        norm1_x = abs(head_leftmost[0] - head_rightmost[0])
        norm1_y = abs(head_topmost[1] - head_bottommost[1])
        norm1 = np.array([norm1_x, norm1_y])
        # Normalisation 2: subtract the nose
        norm2 = np.array(face[4]) # tip of the nose
        
        # The normalisations are per frame, so make a list of all the norms
        norms1.append(norm1)
        norm2/=norm1
        norms2.append(list(norm2))
    return norms1, norms2 

'''
    ---Convert the Mediapipe landmarks (normalized by image size) to (pixel) coordinates---
    landmarks: the dictionary of landmarks of the hands, body and face (dict)
    ---------------------------------------------------------------------------------------
    return: the dictionary of pixel coordinate landmarks of the hands, body and face
'''
def get_pixel_coordinates(landmarks:dict):
    coordinates = {}
    # Instantiate as empty (will have 21, 21, 33, 468 keypoints respectively)
    for key in ['l_hand', 'r_hand', 'pose', 'face']:
        coordinates[key] = []
    
    # Retrieve the width and height of the video
    width, height = landmarks['video_dim']
    
    for key in landmarks: # loop through face, hand, pose landmarks
        if 'dim' in key: 
            # Don't normalize the non-landmark keys, just store them as-is
            coordinates[key] = landmarks[key]
            continue
        
        coords = np.array(landmarks[key])
        
        # Multiply x coordinates by width, y coordinates by height
        coords[:,:,0] *= width 
        coords[:,:,1] *= height 
        # Store the pixel coordinates
        coordinates[key] = coords

    return coordinates

'''
    ---Normalize the (pixel coordinate) landmarks using the shoulder normalisers---
    coordinates: the dictionary of (pixel coordinate) landmarks of the hands, body and face (dict)
    -------------------------------------------------------------------------------
    return: the dictionary of normalised landmarks of the hands, body and face
'''
def normalise_coordinates(coordinates:dict):
    # Getting the normalisers
    norms1, norms2 = get_normalisers(coordinates)
    # Normalizing the pixel coordinates (twice)
    for key in coordinates: # Looping over face, hand, pose keypoints
        if 'present' in key or 'dim' in key: # Don't normalize the boolean entries
            continue

        # Getting the coordinates of the keypoints for every frame
        coords = coordinates[key]
        for i in range(len(coords)): # looping over the frames
            # Normalise all keypoints twice (division by norm1, then subtract norm2)
            coords[i] = (coords[i]/norms1[i])-norms2[i] #  (coords[i]/norms1[i])-norms2[i]
        coordinates[key] = coords
    
    return coordinates

'''
    ---Normalize the (pixel coordinate) only-face landmarks using the face normalisers---
    coordinates: the dictionary of (pixel coordinate) landmarks of only the face (dict)
    -------------------------------------------------------------------------------
    return: the dictionary of normalised landmarks the face
'''
def normalise_face_coordinates(coordinates:dict):
    # Getting the normalisers
    norms1, norms2 = get_face_normalisers(coordinates)
    # Normalizing the pixel coordinates (twice)
    for key in coordinates: # Looping over face keypoints
        if 'dim' in key: # Don't normalize the boolean entries
            continue

        # Getting the coordinates of the keypoints for every frame
        coords = coordinates[key]
        for i in range(len(coords)): # looping over the frames
            # Normalise all keypoints twice (division by norm1, then subtract norm2)
            coords[i] = (coords[i]/norms1[i])-norms2[i]
        coordinates[key] = coords
    
    return coordinates

'''
    ---Save the landmarks of a video (without any normalisation) as a pickle file---
    path:     the path to the video file (str)
    file:     the name of the file to store (str)
    new_path: the path where we want to store the pickle file (str)
    log:      whether to log (print) while processing the file (bool)
    draw:     if the video should be shown while being processed, with landmarks (bool)
    frame_skip: how many frames to skip in the processing, useful if we want to visually inspect a specific part (int)
    -------------------------------------------------------------------------------
    return: the dictionary of non-normalised landmarks of the hands, body and face
'''        
def save_landmarks_for_video_path(path:str, file:str, new_path:str, log:bool = True, draw:bool = True, frame_skip:int = 0):
    # Grab the file name without the extension
    file_no_extension = file.split('.')[0]
    # Whether the video is part of CNGT (requires specific pre-processing)
    cngt_vid = 'CNGT' in file
    # Reserve the pose file, because it is used to check (in another file) if we've already processed the video 
    np.save(new_path+file_no_extension+'_pose.npy', np.array([]))
    # Process the footage with Mediapipe
    if log: print('Getting landmarks...')
    landmarks = process_video(path + file, cngt_vid, draw = draw, frame_skip = frame_skip)
    if log: print('Storing new files at:', new_path)
    # Store the landmarks as separate numpy files (one for each hand, for the pose and face)
    for lmrk_key in landmarks:
        np.save(new_path+file_no_extension+'_'+lmrk_key, np.array(landmarks[lmrk_key]).astype(np.float32))
    if log: print('Done.')
    return landmarks

'''
    ---Compute the normalised landmarks and (optionally) save as a pickle file---
    landmarks:   the non-normalised landmarks of the hands, body and face (dict)
    new_path:    the path where we want to store the pickle file (str)
    log:         whether to log (print) while processing the landmarks (bool)
    save:        whether to save the computed normalised landmarks as a pickle file (bool)
    face_only:   whether to only store the face landmarks
    -------------------------------------------------------------------------------
    return: the dictionary of normalised landmarks of the hands, body and face - or only the face (if face_only = True)
''' 
# def preprocess_landmarks(landmarks:dict, new_path:str, log:bool = True, save:bool=True, face_only:bool = False):
#     # Convert the Mediapipe (width-height-normalised) landmarks to pixel coordinates
#     if log: print('Converting to pixel coordinates...')
#     coordinates = get_pixel_coordinates(landmarks)
#     # Normalise the pixel coordinates by the shoulders
#     if log: print('Normalising pixel coordinates...')
#     coordinates = normalise_coordinates(coordinates)

#     if face_only: # Only keep the face landmarks
#         if log: print('Storing only the face landmarks...')
#         target_keys = ['face', 'video_dim']
#         # Filter out the hand, pose (empty) landmark keys that have been added
#         coordinates = {k: coordinates[k] for k in coordinates if k in target_keys}

#     if save:
#         if log: print('Storing preprocessed file at:', new_path)
#         store_dict(new_path, coordinates)
#     return coordinates

'''
    ---Mirror the pixel coordinate landmarks horizontally---
    landmarks:   the non-normalised landmarks of the hands, body and face (dict)
    ------------------------------------------------------
    return: the dictionary of mirrored landmarks of the hands, body and face
'''
def horizontal_mirror_landmarks(landmarks:dict):
    width, _ = landmarks['video_dim']
    for key in landmarks:
        if not 'dim' in key:
            l = np.array(landmarks[key])
            l[:,:,0] = width - l[:,:,0] - 1
            landmarks[key] = l
    # Swap the left and the right hands
    l_temp, r_temp = landmarks['l_hand'], landmarks['r_hand']
    landmarks['l_hand'], landmarks['r_hand'] = r_temp, l_temp
    return landmarks

# Convert to pixel coordinates, mirror the image (e.g. x = width - x - 1), then normalize
def mirror_and_norm(landmarks):
    pixel_coords = get_pixel_coordinates(landmarks)
    mirror_landmarks = horizontal_mirror_landmarks(pixel_coords)
    norm_mirror = normalise_coordinates(mirror_landmarks)
    return norm_mirror