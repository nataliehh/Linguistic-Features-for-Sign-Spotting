# DATASET CREATION
import numpy as np
import copy
from tools.tools import ms_to_frame

# This gives indices for zero padding
def zero_padding(ann_len, fixed_length, too_short):
    # If the annotation is too short, we want to pad it with zeros
    if too_short:
        frames_short = fixed_length - ann_len
        chosen_ind = frames_short*[-1] # Flag the annotation to be padded with zeros
    # With annotations that are too long, we undersample them
    else:
        div, mod = divmod(ann_len, fixed_length)
        ind_kept = list(range(0, ann_len, div)) # Keep every X-th frame
        # If we still have too many frames after taking every X-th frame, randomly remove some frames
        # Until we reach the target duration
        chosen_ind = sorted(np.random.choice(ind_kept, fixed_length, replace = False))
    return [chosen_ind]

# Gives indices of frames from oversampling
def over_sampling(ann_len, fixed_length, too_short):
    chosen_inds = []
    if too_short:
        div, mod = divmod(fixed_length, ann_len)
        duplicated = np.array(list(range(ann_len)) * div)
        chosen_ind = np.random.choice(ann_len, mod)
        chosen_ind = sorted(np.append(duplicated,chosen_ind))
    else:
        div, mod = divmod(ann_len, fixed_length)
        ind_kept = list(range(0, ann_len, div))
        chosen_ind = sorted(np.random.choice(ind_kept, fixed_length, replace = False))
    chosen_inds.append(chosen_ind)
    return chosen_inds

# Convert all annotations (variable duration) to a fixed length (X frames)
def ann_to_fixed_length(ann_len, fixed_length, zero_pad = False):
    small, precise, large = [False] * 3
    if ann_len != fixed_length:
        too_short = ann_len < fixed_length
        small, large = too_short, (1-too_short)
        if zero_pad: # Use zero padding (padded frames contain only zeros)
            return zero_padding(ann_len, fixed_length, too_short), small, precise, large
        return over_sampling(ann_len, fixed_length, too_short), small, precise, large
        # return list(range(10)), small, precise, large
    else:
        precise = True
        return [np.arange(ann_len)], small, precise, large
           
# This adds all the data in the list and corresponding labels
# To the data matrix X, and label list y
def add_lmrks(X, y, data_list, label_list):
    for i in range(len(data_list)):
        data, label = data_list[i], label_list[i]
        X.append(data)
        y.append(label)
    return X, y

# Make dataset with the annotations
# Fixed_length = the target size of an annotation (we scaled it to this)
def make_dataset(anns_with_tiers, features, df, val_vids, test_vids, top_signs, id_split, fixed_length = 10, 
                 zero_pad = False, ling_features = True):
    # Creates empty list for train, train (no augmentation), validation and test sets, as well as the glosses
    X_train, X_train_no_mirr, X_val, X_test, y_train, y_train_no_mirr, y_val, y_test, glosses  = [[] for _ in range(9)]
    # Create lists specifically for the train set where we keep variable annotation lengths
    X_train_var_len, y_train_var_len = [], []
    empty, larger, smaller, precise = [0]*4
    ann_lengths = []

    # To ignore the handedness of the annotation, we can use this dictionary instead
    # But the code has to also be adapted a bit to suit this analysis instead
    # anns_dicts = load_dict(destination)

    # Here, we analyze how often landmarks for the hands are detected given the annotations
    # Looping over the .eaf files (i.e. the different signer videos)
    for i, k in enumerate(anns_with_tiers):
        print('Processing video {}/{}'.format(i+1, len(list(anns_with_tiers.keys()))), end = '\r')

        # Get the glosses of the file
        anns_dict = anns_with_tiers[k]

        # Adding all elements of the annotations to a dictionary
        # In the format: (start_time, end_time, [side], sign_label)
        items = []
        for key in anns_dict:
            gloss = key.replace('__2H', '')
            # Check if it's an English annotation - we map to Dutch based on Signbank for these
            dutch_equivalent = df.loc[df['Annotation ID Gloss (English)']==gloss]['Annotation ID Gloss (Dutch)']
            # Also check if it isn't already Dutch
            already_dutch = df.loc[df['Annotation ID Gloss (Dutch)']==gloss]['Annotation ID Gloss (Dutch)']
            # Check if there's any English entries equal to this annotation, if so we map to Dutch
            if dutch_equivalent.shape[0] > 0:
                dutch_gloss = dutch_equivalent.tolist()[0]
            elif already_dutch.shape[0] > 0:
                dutch_gloss = gloss   
            # We ignore instances where the annotation doesn't exist in Dutch or English in Signbank
            else: 
                continue

            # If we are using a filtering for top signs, we check if our gloss is in that top list
            if dutch_gloss in top_signs or len(top_signs) == 0:
                # Add the key as an element to each tuple of timespans
                tup_with_key = [(v)+(dutch_gloss,) for v in anns_dict[key]]
                items += tup_with_key

        if len(items) == 0: # If no annotations, skip the rest
            continue

        # Format map e.g. 'S085_CNGT2143.eaf' ->'CNGT2143_S085'
        # To match with the id_split key format
        vid_id = '_'.join(k.replace('.eaf', '').split('_')[::-1])
        train_element = vid_id in id_split['Train']

        # Load the normalised landmarks
        lmrk_dict = features[k]

        # Get the landmarks
        l_lmrk, r_lmrk = lmrk_dict['l_hand'], lmrk_dict['r_hand']

        if train_element:
            # We mirror the hands for the mirrored variant
            mirror_l_lmrk, mirror_r_lmrk = copy.deepcopy(lmrk_dict['r_hand']), copy.deepcopy(lmrk_dict['l_hand'])
            # x coordinates have to be mirrored (so we factorize by -1)
            if ling_features:
                # Wrist (x coord = ind 44) + fingertips (x coord = every other ind in 46...55)
                mirror_ind = [44] + list(range(46,56,2)) #list(range(61,71,2))
            else: # If we use landmarks instead of linguistic features, we can mirror all x-coordinates
                mirror_ind = np.array(range(0, mirror_l_lmrk.shape[-1], 2)) # every other index is an x-coord
            # Due to the way we normalised, multiplying by -1 gets us a mirrored equivalent
            mirror_l_lmrk[:,mirror_ind] *=-1
            mirror_r_lmrk[:,mirror_ind] *=-1

        # Looping over the annotation items in the video
        for item in items:

            # Two-handed signs unpack differently, don't include a 'side' element
            if len(item) == 3:
                start, end, key = item
            else:
                start, end, side, key = item

            # Convert ms to frames to be compatible with mediapipe framewise landmarks
            start_frame = ms_to_frame(start)
            end_frame = ms_to_frame(end)

            # Get frames for the given annotation window
            l = l_lmrk[start_frame:end_frame+1]
            r = r_lmrk[start_frame:end_frame+1]

            # If the annotation is completely out of bounds, we make a note of this and then skip
            if l.shape[0] == 0 or r.shape[0] == 0:
                empty += 0
            else:  

                # Fuse the features of the right & left hand together
                # Then check how often no features are extracted at all
                # This is the case if no landmarks are detected for a frame
                lmrks = np.append(l, r, axis = 1)
                print(l.shape, r.shape)
                if ling_features:
                    diff_wrists = r[:,[44,45]] - l[:,[44,45]]
                    lmrks = np.append(lmrks, diff_wrists, axis = 1)
                num_lmrks = lmrks.shape[0]
                lmrks_present = np.unique(np.where(~np.isnan(lmrks))[0]) 

                # If it's a train example, get the landmarks of both hands 
                # For the mirrored data too and then put the landmarks in one array
                if train_element:
                    mirror_l = mirror_l_lmrk[start_frame:end_frame+1]
                    mirror_r = mirror_r_lmrk[start_frame:end_frame+1]
                    mirror_lmrks = np.append(mirror_l, mirror_r, axis = 1)
                    if ling_features:
                        diff_wrists = mirror_r[:,[44,45]] - mirror_l[:,[44,45]]
                    mirror_lmrks = np.append(mirror_lmrks, diff_wrists, axis = 1)

                if lmrks_present.shape[0] == 0:
                    empty+=1
                # If at least some the frames have detected landmarks, we continue processing
                if lmrks_present.shape[0] > 0.0: 
                    ann_lengths.append(num_lmrks)
                    # For annotations not matching exactly with the desired fixed length 
                    # We create a few random samples to make sure we have different examples
                    chosen_inds, s, p, l = ann_to_fixed_length(lmrks.shape[0], fixed_length, zero_pad)

                    smaller += s
                    larger += l
                    precise += p
                    glosses.append(key)
                    for i, chosen_ind in enumerate(chosen_inds):
                        if -1 in chosen_ind: # -1 indicates that we want to use padding
                            num_pads = len(chosen_ind)
                            lmrks_select = np.pad(lmrks, ((0, num_pads), (0,0)))
                            if train_element:
                                mirror_lmrks_select = np.pad(mirror_lmrks, ((0, num_pads), (0,0)))
                        # Get the annotation for the selected frames
                        else:
                            lmrks_select  = lmrks[chosen_ind] 
                            if train_element:
                                mirror_lmrks_select = mirror_lmrks[chosen_ind]

                        # Store the train or test example, for train we also add the mirrored example
                        if train_element:
                            X_train, y_train = add_lmrks(X_train, y_train, [lmrks_select, mirror_lmrks_select], [key, key])
                            X_train_var_len, y_train_var_len = add_lmrks(X_train_var_len, y_train_var_len, [lmrks], [key])
                            # Add every other train element to the non-augmented set because we take 2 samples for augmentation
                            if i % 1 == 0: 
                                X_train_no_mirr,y_train_no_mirr = add_lmrks(X_train_no_mirr,y_train_no_mirr,[lmrks_select],[key])
                        elif vid_id in test_vids:
                            X_test, y_test = add_lmrks(X_test, y_test, [lmrks_select], [key])
                        elif vid_id in val_vids:
                            X_val, y_val = add_lmrks(X_val, y_val, [lmrks_select], [key])
    # Group together the X data, y labels, the annotation length stats and exact lengths
    result = ([X_train, X_train_no_mirr, X_val, X_test], [y_train, y_train_no_mirr, y_val, y_test], 
              [empty, smaller, precise, larger], [X_train_var_len, y_train_var_len], ann_lengths, glosses)
    return result

# Custom normalisation method because scipy can cause issues with NaNs
def norm(arr, mean = None, std = None):
    shape = arr.shape
    arr = arr.reshape(-1, shape[-1])
    if mean is None or std is None:
        mean, std = np.mean(arr, axis = 0), np.std(arr, axis = 0)
        std[std == 0] = 1 # Avoid division by zero
    arr_norm = (arr-mean)/std
    arr_norm = arr_norm.reshape(shape)
    return arr_norm, mean, std