import numpy as np
import math
import mediapipe as mp
from scipy.spatial.distance import cdist

mp_hands = mp.solutions.hands

# Get the euclidean distance between two points (x1, y1) and (x2, y2)
def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# Compute the velocities of the wrists based on the Mediapipe Pose model (more accurate than Hand model)
def compute_wrist_velocities_from_pose(pose, t1 = None, t2 = None):
    # Get the wrists from the Pose (more accurate than the Hands wrist landmark)
    l_wrist, r_wrist = pose[:, 15, :], pose[:, 16, :]
    # Compute wrist velocity, NOT removing any NaN-frames (need velocity for each frame for our features)
    wrist_l_vel = compute_wrist_velocity(l_wrist, t1, t2, remove_nan = False)
    wrist_r_vel = compute_wrist_velocity(r_wrist, t1, t2, remove_nan = False)
    return wrist_l_vel, wrist_r_vel

def compute_wrist_velocity(wrist, t1 = None, t2 = None, remove_nan = False, pad = False, fused_only = True):
    if t1 and t2: # Crop to specific timestamp if specified
        wrist = wrist[t1:t2, :]
    # If remove_nan = True, find where the wrist is detected (not NaN) and use only those frames
    if remove_nan:
        frames_present  = np.unique(np.where(~np.isnan(wrist))[0])
        wrist = wrist[frames_present, :]
        if frames_present.shape[0] <= 1: # No frames or max. 1 frame detected -> 0 velocity
            return 0
    
    wrist_velocity = np.diff(wrist, axis=0)
    # For x frames, we will get x-1 velocities, we pad to x velocities
    if pad: 
        wrist_velocity = np.insert(wrist_velocity, 0, [0,0], axis = 0)

    # We compute a fused velocity: sqrt((x1-x2)**2 + (y1-y2)**2)
    wrist_velocity_fused = np.sqrt(np.sum(wrist_velocity**2, axis = 1))

    # Either only return the velocity that's fused, or return fused and non-fused
    if fused_only:
        return wrist_velocity_fused
    else:
        wrist_velocity_fused = wrist_velocity_fused.reshape(-1,1)
        # print(wrist_velocity_fused.shape)
        # print(wrist_velocity.shape)
        wrist_velocities = np.append(wrist_velocity_fused, wrist_velocity, axis = 1)
        return wrist_velocities

# Computing the average wrist velocity based on the euclidean distance between each time point t and t+1
# This used to be called compute_wrist_velocity
def compute_both_wrist_velocities(landmarks, t1 = None, t2 = None, remove_nan = False):
    # Compute velocity of both hands
    l_hand = np.array(landmarks['l_hand'])
    r_hand = np.array(landmarks['r_hand'])
    # Pass the wrist of both hands to get their velocities
    wrist_l_vel = compute_wrist_velocity(l_hand[:, 0, :], t1, t2, remove_nan)
    wrist_r_vel = compute_wrist_velocity(r_hand[:, 0, :], t1, t2, remove_nan)
    
    # Sum the velocity for each frame, average at the end
    wrist_l_vel = np.sum(wrist_l_vel, axis = 0)/l_hand.shape[0]
    wrist_r_vel = np.sum(wrist_r_vel, axis = 0)/r_hand.shape[0]
    
    # Also return which hand has more velocity (aka the dominant hand)
    return wrist_l_vel, wrist_r_vel, wrist_l_vel <= wrist_r_vel

# https://stackoverflow.com/questions/8905501/extract-upper-or-lower-triangular-part-of-a-numpy-matrix
# https://stackoverflow.com/questions/47877530/calculate-distance-of-2-list-of-points-in-numpy
def calc_hand_distance(hand_data, selection = True):
    # Getting the wrist, palm indices from the mediapipe pose model
    wrist_ind = mp_hands.HandLandmark.WRIST.value
    palm_ind = mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value
    x1, y1 = hand_data[wrist_ind]
    x2, y2 = hand_data[palm_ind]
    palm_size=get_distance(x1, y1, x2, y2)
    
    # Then, we select only the hand landmarks we actually want to calculate the distances between
    # These are the landmarks of the handpalm, wrist, and fingertips
    if selection:
        hand_selection = [0,4,8,12,16,20] # [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
        hand_data = hand_data[hand_selection, :]
    
    # Computes sqrt((x1-x2)**2 + (y1-y2)**2) for all point pairs, result is N x N matrix 
    # We only want the upper triangle because the matrix is symmetrical and diagonal sim. is self-similarity (=1)
    # We normalize by the palm size to make this robust to scaling 
    dist_matrix = cdist(hand_data, hand_data)/palm_size
    N = dist_matrix.shape[0]
    return dist_matrix[np.triu_indices(N, k = 1)]

# Compute the distance for each frame of the data
def calc_distance_per_frame(hand_data):
    # Loop over the frames in list comprehension, compute distance features for each frame
    distance_per_frame = [calc_hand_distance(np.array(d)) for d in hand_data]
    # hand_data = np.array(hand_data)
    # distance_per_frame = []
    # for frame in range(hand_data.shape[0]):
    #     d = calc_hand_distance(hand_data[frame])
    #     distance_per_frame.append(d)
    return np.array(distance_per_frame)

# https://github.com/nicknochnack/MediaPipePoseEstimation/blob/main/Media%20Pipe%20Pose%20Tutorial.ipynb
def angle_between(start, mid, end):
    start, mid, end = np.array(start), np.array(mid), np.array(end)
    # Compute radians, then convert to angle
    radians = np.arctan2(end[:, 1]-mid[:, 1], end[:, 0]-mid[:, 0]) - np.arctan2(start[:, 1]-mid[:, 1], start[:, 0]-mid[:, 0])
    angle = np.abs(radians*180.0/np.pi)
    # Rectify angles that are large than 180
    too_large = np.where(angle > 180.0)
    angle[too_large] = 360-angle[too_large]  
    return angle/180 

def slope(x1, y1, x2, y2): # Line slope given two points:
    # To avoid division by 0, add a small constant
    return (y2-y1)/(x2-x1 + 0.000001)

# Given two list of slopes (for each frame)
def angle_lists(slopes1, slopes2): 
    ang = abs(np.degrees(np.arctan((slopes2-slopes1)/(1+(slopes2*slopes1)))))
    # Angles bigger than 180 are fixed
    too_large = np.where(ang>180)
    ang[too_large] = 360-ang[too_large]
    return ang/180

def get_slope(arr):
    return slope(*arr.flatten())

# Compute the angle of a midpoint between two points
def calc_hand_angle_midpoint(hand_data, pose = None):
    # Compute angles within the finger 
    # Taking base of finger as start, fingertip as end and a point in-between as the middle
    selection = [[1,2,4], [5,6,8], [9,10,12], [13,14,16], [17,18,20]]
    angles = []
    for s_1 in selection:
        # Get the beginning, middle and end point
        begin, mid, end = s_1
        # With the handpalm, we use the fingertip, finger base and wrist
        s_2 = [0, begin, end]
        # Flipping the hand data so we have it in the form
        # (specific_landmark (e.g. wrist landmark), frame, x and y coords)
        # This is done so we can flatten it into three lists of landmarks
        angle_1 = angle_between(*hand_data[:, s_1].transpose(1,0,2))
        angle_2 = angle_between(*hand_data[:, s_2].transpose(1,0,2))
        angles.append(angle_1)
        angles.append(angle_2)

    # We do the same with some specific angles taken from the paper "Real-time Dynamic Sign Recognition using MediaPipe"
    specific_angles = [[2,3,4], [5,6,7], [6,7,8], [9,10,11], [10,11,12], [13,14,15],[14,15,16], [17,18,19], 
                       [18,19,20], [4,0,8], [8,0,20], [16,17,20], [8,5,12], [4,5,20], [8,13,20]]
    for s in specific_angles:
        angle = angle_between(*hand_data[:, s].transpose(1,0,2))
        angles.append(angle)

    # Compute middle of shoulders and hips for each frame
    shoulders = np.append(pose[:,11], pose[:,12], axis = 1)
    shoulder_mids = (pose[:,11]+pose[:,12])/2
    hip_mids = (pose[:,23]+pose[:,24])/2
    
    # We make a list of the (hip_middle, shoulder_middle) landmarks for each frame
    mids = np.append(hip_mids, shoulder_mids,axis=1)
    
    # Get the vertical and horizontal axes within the hand for each frame
    # Vertical = from wrist to hand palm, horizontal = from left- to rightmost hand palm landmark
    hand_verticals = hand_data[:, [0,9]]
    hand_horizontals = hand_data[:, [5,17]]

    # Get the slopes of the hand vertical & horizontal line, same for the torso
    slopes_verts = np.apply_along_axis(get_slope, 1, hand_verticals.reshape(-1,4))
    slopes_hors = np.apply_along_axis(get_slope, 1, hand_horizontals.reshape(-1,4))
    slopes_mids = np.apply_along_axis(get_slope, 1, mids.reshape(-1,4))
    slopes_shoulders = np.apply_along_axis(get_slope, 1, shoulders.reshape(-1,4))
    
    # If we have the line in the middle of the torso, compute how the hand is angled
    # W.r.t. the torso, from the wrist-handpalm line and the horizontal handpalm line
    hoz_angles = angle_lists(slopes_hors, slopes_mids)
    vert_angles = angle_lists(slopes_verts, slopes_mids)

    # Append these angles as rows to the angles list, too
    angles.append(hoz_angles)
    angles.append(vert_angles)
    
    # Do the same for the line between the shoulders
    hoz_angles = angle_lists(slopes_hors, slopes_shoulders)
    vert_angles = angle_lists(slopes_verts, slopes_shoulders)
    angles.append(hoz_angles)
    angles.append(vert_angles)
        
    # Angles are currently appended as rows, but we want them as columns so we transpose
    return np.array(angles).T
        
# Calculate the angles for a hand, for each frame
def calc_angles_per_frame(hand_data, pose):
    hand_data = np.array(hand_data)
    angles = calc_hand_angle_midpoint(hand_data, pose)
    return np.array(angles)#.reshape(-1, 2)

# Compute how the x, y coordinates change over the next few frames
def calc_loc_change(hand_data, diff_length = 5):
    wrist_data = hand_data[:, 0, :]
    wrist_data = np.nan_to_num(wrist_data)
    wrist_x = wrist_data[:, 0]
    wrist_y = wrist_data[:, 1]
    # Find difference between all frames
    diff_x = np.subtract.outer(wrist_x, wrist_x) # normalizing by wrist size???
    diff_y = np.subtract.outer(wrist_y, wrist_y)
    # abs_diff_x = np.abs(diff_x)
    # abs_diff_y = np.abs(diff_y)
    sum_x, sum_y = [], []

    for i in range(diff_x.shape[0]):
        to_sum_x = diff_x[max(0,i-diff_length): i+1]
        to_sum_y = diff_y[max(0,i-diff_length): i+1]
        sum_x.append(np.sum(to_sum_x))
        sum_y.append(np.sum(to_sum_y))
        
    return np.array([sum_x, sum_y]).T

# Feature extraction to represent phonetics as proposed by Stokoe
# Gets distances, angles between the hand lmrks and location of wrist and fingertips
# Together these represent handshape, orientation, location and movement of the sign
def get_wrist_angle_distance(lmrks, key, pose = None):
    lmrk = lmrks[key] # Get the specified hand
    left = key == 'l_hand'

    # Replace the Hand model's fingertip coords if possible with the Pose model
    # Because the Pose model of Mediapipe is more accurate
    # The Pose model only keeps track of the wrist, thumb, index and pinky (not middle and ring finger)
    # if pose is not None:
    #     # From the Pose model, we either grab index 15 (left) or 16 (right hand)
    #     lmrk[:,0,:] = pose[:, 15, :] if left else pose[:, 16, :]
    #     # Grab the indices of the pinky, index finger and thumb (indices depend on which hand we are using)
    #     fingertips = pose[:, [17,19,21], :] if left else pose[:, [18,20,22], :]
    #     for f in range(fingertips.shape[1]):
    #         finger = fingertips[:, f, :]
    #         # Get the base index (pinky, index, or thumb) indices from the Hand model of Mediapipe
    #         lmrk_equivalent = [17,5,1][f]
    #         lmrk[:, lmrk_equivalent, :] = finger

    # Compute distances and angles between hand's landmarks
    angle = calc_angles_per_frame(lmrk, pose)
    dist = np.array(calc_distance_per_frame(lmrk))

    wrist = lmrk[:, 0, :] # wrist of the landmarks
    
    # Putting them together for each frame as (angles, distances, wrist)
    features = np.append(angle, dist, axis = 1)
    features = np.append(features, wrist, axis = 1)
    
    # Loop over fingertip indices for the middle and ring finger
    for f in [4,8,12,16,20]:
        # Get their position relative to the wrist, then add to the features
        fingertip = lmrk[:, f, :] - wrist
        features = np.append(features, fingertip, axis = 1)

    # Get the velocity and append it as a feature for each frame
    velocity = compute_wrist_velocity(wrist, pad = True, fused_only = False)#.reshape(-1,1)
    features = np.append(features, velocity, axis = 1)
    # print('{} dist, {} angle, {} wrist, {} velocity'.format(dist.shape, angle.shape, wrist.shape, velocity.shape))

    # Getting the centroid of all the landmarks of the hand
    # centroid = np.mean(lmrk, axis = 1)
    # features = np.append(features, centroid, axis = 1)
    
    # if pose is not None:
    #     shoulder = pose[:, 11, :] if left else pose[:, 12, :]
    #     features = np.append(features, shoulder, axis = 1)
    
    # loc_change = calc_loc_change(lmrk)
    # features = np.append(features, loc_change, axis = 1)

    # print('Distance shape', dist.shape)
    # print('Angle shape', angle.shape)
    # print('Wrist loc shape', wrist.shape)
    # print('Velocity shape', velocity.shape)
    # print('Centroid shape', centroid.shape)
    # print('Shoulder shape', shoulder.shape)
    return features