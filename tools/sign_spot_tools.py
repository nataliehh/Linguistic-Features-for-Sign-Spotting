import numpy as np
import os
import pympi
from scipy.spatial.distance import cdist
from tools.create_model import get_loss
from tools.make_dataset import norm
from tools.constants import PATHS, ANN_LENGTH # path constants
from tools.tools import is_overlap, get_gloss_vals, man_sim_and_hand_dist, load_dict
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.optimizers import Adam

np.random.seed(123) # Set random seed for consistency

# The set of features (by index) to remove if we're using linguistic features, because they are highly correlated with other features
to_remove_linguistic = [] # 186, 187, 188, 189
#list(range(47, 72))#list(range(45)) 

# https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
# Converts consecutive numbers to ranges, e.g. [1,2,3,5,6,8] -> [(1,3), (5,6), (8,8)]
# If we specify gap > 0, then we also allow there to be one or more values missing consecutively
def ranges(nums, gap = 0):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1+abs(gap) < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

# Given a range of (start,end) points, compute what ranges lie in between
# E.g. ranges = [(1,2), (5,6)] -> in-between = (2,5)
def between_ranges(ranges, length):
    between = []
    if len(ranges) == 0:
        return [(0,length)]
    # Append a range from the beginning of the video: (0, ...)
    if ranges[0][0] > 0:
        between.append((0, ranges[0][0]))
    for i in range(len(ranges)-1):
        start, end = ranges[i][-1], ranges[i+1][0]
        if start < end:
            between.append((start,end))
    # Append a range for the end of the video (..., end)
    if ranges[-1][-1] < length:
        between.append((ranges[-1][-1], length))
    return between

# Custom z-score method because scipy can cause issues with NaNs
def z_score(arr, mean = None, std = None):
    if mean is None or std is None:
        mean, std = np.mean(arr, axis = 0), np.std(arr, axis = 0)
        std[std == 0] = 1 # Avoid division by zero
    return (arr-mean)/std, mean, std

# You need to use a path with a formatting option here (e.g. containing {})
def load_data(format_path, data_split, mean = None, std = None, norm = False, rand_aug = False, to_remove = [], log = False):
    if log: print('Loading {} data...'.format(data_split))
    X_path = format_path.format('X', data_split)
    y_path = format_path.format('y', data_split)
    if log: print('X being loaded...')
    X = np.load(X_path)
    if log: print('original shape:', X.shape)
    
    # If the data isn't already normalised (we check the path name to verify this)
    # We do it here, and also get rid of any NaNs
    if log: print('NaN to num...')
    X = np.nan_to_num(X)
    
    # Adding slight randomness to the data to augment it
    if rand_aug:
        if log: print('Augmenting with randomness...')
        shift = 0.01
        start, end = 1-shift, 1+shift

        if 'only_lmrks' in X_path:
            rand = np.random.uniform(start, end, X.shape)
            X *= rand
        else:
            shape = X.shape
            # Get the location features (we don't want to augment them)
            loc_feats_l = [45,46] + list(range(61,71))
            loc_feats_r = [f + 71 for f in loc_feats_l]
            # Get indices of non-location features
            non_loc_feats = list(set(range(142)) - set(loc_feats_l) - set(loc_feats_r))
            non_loc_shape = (shape[0], shape[1], len(non_loc_feats))
            # Only augment non-location features
            rand = np.random.uniform(start, end, non_loc_shape)
            X[:, :, non_loc_feats] *= rand
        
    # We remove any columns that are highly correlated with each other
    # To reduce computations and make training easier
    cols = range(X.shape[-1])
    remain = set(cols) - set(to_remove)
    X = X[:, :, list(remain)]
        
    # Normalise if the data is not already normalised 
    # This should be set to false probably, because we already use some normalisation for the landmarks
    if norm:
        if log: print('Normalising...')
        X, mean, std = z_score(X, mean, std) 
        
    if log: print('Converting to smaller float representation...')
    # Train data has to be float16 to fit into memory later
    # Otherwise the data is huge and can't be worked with
    X = X.astype(np.float32)
    y = np.load(y_path)
    if log: print('Done.')
    return X, y, mean, std

def get_data(only_lmrks, norm = False, log = False, rand_aug = True):
    # Path to the data and labels
    root = PATHS['only_lmrks_root'] if only_lmrks else PATHS['cngt_data_root']
    features = '_only_lmrks' if only_lmrks else ''
    suffix = features + '__top.npy'
    path = root + 'CNGT_{}_{}' + suffix  
    top = '_top' if 'top' in path else ''

    # Indices of features to remove bc they're highly correlated with other features
    # If we are using only landmarks, we do not remove any features because this removes too many
    to_remove = [] if only_lmrks else to_remove_linguistic

    # Loading train, test data and printing their shapes
    X_train, y_train, train_mean, train_std = load_data(path, 'train', rand_aug = rand_aug, to_remove = to_remove, norm = norm, log = log)

    # We use only the non-mirrored, not augmented train data to make reference embeddings later
    X_train_no_mirr, y_train_no_mirr, _, _ = load_data(path, 'train_no_mirror', norm = norm, to_remove = to_remove, log = log)
   
    # We load in the validation and test data and normalise them using the train mean and std
    X_val, y_val, _, _ = load_data(path, 'val', train_mean, train_std, norm = norm, to_remove = to_remove, log = log)
    X_test, y_test, _, _ = load_data(path, 'test', train_mean, train_std, norm = norm, to_remove = to_remove, log = log)

    print('-'*100) # Printing data and label shapes
    print('Train shape:\t\t{}\tVal shape:\t{}\tTest shape:\t{}'.format(X_train.shape, X_val.shape, X_test.shape))
    print('Train label:\t\t{}\t\tVal label:\t{}\tTest label:\t{}'.format(y_train.shape, y_val.shape, y_test.shape))
    print('W/o augmentation:\t{}\tlabels:\t{}'.format(X_train_no_mirr.shape, y_train_no_mirr.shape))
    
    return X_train, y_train, X_train_no_mirr, y_train_no_mirr, X_val, y_val, X_test, y_test, top

# Training the model with data shuffling
def train_model(model, train_batch_gen, val_batch_gen, X_train, y_train, X_val, y_val, labels, target_num_label, batch_size, 
                learning_rate, temperature, num_epochs = 20, plot_freq = 1, patience = 5, decay_rate = 5, ws = 8, num_classes =1000):
    model.compile(optimizer=Adam(learning_rate), loss=get_loss(temperature, num_classes))
    # Initialize lists to store history of train, val losses
    val_losses, train_losses = [[] for _ in range(2)]
    increases = 0

    for i in range(num_epochs):
        # Decay the learning rate every X epochs
        if i % decay_rate == 0 and i > 0:
            learning_rate /= 2 # halve the learning rate

            # Recompile model with lower learning rate
            model.compile(optimizer=Adam(learning_rate), loss=get_loss(temperature, num_classes))
            
        print('Epoch {}'.format(i+1))
        history = model.fit(x=train_batch_gen, validation_data = val_batch_gen, epochs=1, verbose = 1)
        val_losses.append(history.history['val_loss'][0])
        train_losses.append(history.history['loss'][0])
        # If the loss increased for X consecutive epochs, we stop
        if len(val_losses) > 1:
            if val_losses[-1] > val_losses[-2]: 
                increases +=1
            else: # If it's not consecutively increasing, reset the counter
                increases = 0
            if increases >= patience: break # Increasing for more than X epochs
        # Can't directly shuffle the generator, so we delete it and make a new one
        del train_batch_gen
        # Shuffle the training data using indices
        train_shuffle = np.random.choice(np.arange(y_train.shape[0]), y_train.shape[0], replace = False)
        X_train, y_train = X_train[train_shuffle], y_train[train_shuffle]
        # Make new batch generator from the newly shuffled data
        train_batch_gen = positive_pairs_batch_gen(X_train, y_train, batch_size = batch_size, window_size = ws)
        # Plot about 1/4th of the validation data
        if i % plot_freq == 0:
            plot_sim_of_sampled_embds(model, X_val, y_val, target_num_label, labels, num_samples = 2000, 
                                      log = False, set_str = 'val')
    return val_losses, train_losses

# This makes one reference embedding per sign seen during training, because there are multiple instances of each sign
# Top_rate = the ratio of signs to use (e.g. 0.1 is 10%)
def ref_embds(model, X_train_no_mirr, y_train_no_mirr, y_train_labels, top_ratio = 0.1):
    # Make reference embeddings for each sign based on the train set embeddings
    reference_sign_embds = {}
    for i, label in enumerate(y_train_labels):
        print('embedding sign {}/{}'.format(i+1,len(y_train_labels)), end = '\r')
        # Get all indices where the num. label matches
        ref_inds = np.where(y_train_no_mirr == label)[0]
        train_ref = X_train_no_mirr[ref_inds]
        N = train_ref.shape[0] # The num. of instances with that label
        train_embds = model.predict(train_ref, verbose = 0) # Predict
        # Get cosine distance between all the embeddings
        train_ref_dist = cdist(train_embds, train_embds, metric = 'cosine')
        # Get averaged distance from all other reference embeddings
        train_total_dist = np.sum(train_ref_dist, axis = 1)/N
        # We then get the top X% of most representative (closest) embeddings for the sign
        top = max(1,round(N*top_ratio))
        ind = train_total_dist.argsort()[:top]
        top_similar_embds = train_embds[ind]
        # Average these top X% embeddings, store it as the reference embedding for the sign
        reference_res = np.mean(top_similar_embds, axis = 0)
        reference_sign_embds[label] = reference_res
    return reference_sign_embds

# For each target sign in the test set, we check if the right train embedding (of the same sign) ranks
# If the target's train embedding is in the top-k, we count it as an accurate ranking
def top_k_ranking_acc(reference_sign_embds, X_test_pred, y_test, y_train_labels, k_vals = [20]):
    acc= {}
    for k in k_vals:
        acc[k] = 0
    total = 0

    # Get the embeddings of the reference signs and their labels 
    train_embds = np.array(list(reference_sign_embds.values()))
    train_embds_labels = np.array(list(reference_sign_embds.keys()))

    # Go over all sign labels we saw during training
    for i, label in enumerate(y_train_labels):
        print('Getting ranking for retrieved sign {}/{}'.format(i+1, len(y_train_labels)), end = '\r')
        
        # Get the test set annotations that match with the sign label
        test_inds = np.where(y_test == label)[0] 
        test_samples = X_test_pred[test_inds] # Get their embeddings

        # For each test set embedding of the current target sign
        for sample in test_samples:
            total += 1
            sample = np.expand_dims(sample, 0)
            # Get the distance with the reference sign embeddings
            dist = cdist(sample, train_embds).flatten()
            for k in k_vals:
                # Get the top X closest reference embeddings to the test embedding
                ind = dist.argsort()[:k]
                # Get the labels corresponding to the top X reference embeddings
                similar_labels = train_embds_labels[ind]
                if label in similar_labels: acc[k]+=1 # If the real label is in the top X, it's accurately ranked

    for k in k_vals:
        print('Number of target signs ranked in top {}: {}/{}'.format(k, acc[k], total))
        print('accuracy@{}: {}%'.format(k, round(100*acc[k]/total,2))) # or is it recall?

# Make embeddings of each video in the test set, by windowing over it
def make_test_video_embds(model, only_lmrks, select_video = ''):
    if only_lmrks: # We don't remove any landmark features
        to_remove = []
    else: # We remove specific, highly correlated linguistic features
        to_remove = to_remove_linguistic

    print('Loading annotations and getting test video ids...')
    test_vid_ids = np.load(PATHS['test_vid_ids']) # Loading ids of the test videos
    dataset_root = PATHS['cngt_vids_and_eaf'] # Root where all annotated .eaf sign files are 
    anns_with_tiers = load_dict(PATHS['dataset_anns']) # Get video annotations
    if len(select_video) > 0 and select_video in anns_with_tiers:
        anns_with_tiers = {select_video: anns_with_tiers[select_video]}
    # Path where the data features are stored
    print('Loading test video features...')
    str_features = '_only_lmrks' if only_lmrks else '' 
    features_path = PATHS['features_data'].format(str_features)
    features_data = dict(np.load(features_path, allow_pickle = True))

    norm_vals = np.load(PATHS['normalisation'].format(str_features))
    mean, std = norm_vals[0], norm_vals[1]

    # Getting the annotations and embeddings for test set videos
    video_embd_dict, anns = {}, {}
    vid_i = 0
    for k in anns_with_tiers:
        if not k.replace('.eaf', '') in test_vid_ids: # Only do this for test videos
            continue
        vid_i += 1
        print('embedding test set video {}/{}'.format(vid_i, len(test_vid_ids)), end = '\r')
        # Getting the landmarks
        eaf_file = pympi.Elan.Eaf(os.path.join(dataset_root, k))
        # Get the glosses and mouthings of the file
        anns_dict, mouthings_dict = get_gloss_vals(eaf_file)
        anns_dict = man_sim_and_hand_dist(anns_dict, manual_sim = False, two_hand_suffix = False)
        features = get_features(features_data[k], not only_lmrks)

        # Normalize the features in the same way as the train data
        features, _ , _ = norm(features, mean, std)

        # Making sliding window
        batch = []
        target_shape = (ANN_LENGTH,) + features.shape[1:] # Sliding window size is determined by train data frame length
        batch = np.lib.stride_tricks.sliding_window_view(features, window_shape = target_shape).reshape((-1,)+target_shape)
        # for i in range(0, features.shape[0]-size+1):
        #     window = features[i:i+size]
        #     batch.append(window)
        # batch = np.array(batch)
        # Remove the same features as for the train, validation sets
        batch = batch[:,:, list(set(range(batch.shape[-1]))-set(to_remove))]
        # Get the embedding for each sliding window chunk and store them
        embd = model.predict(batch, verbose = 0)
        video_embd_dict[k] = embd
        anns[k] = anns_dict # Store also the corresponding annotations in a dictionary
    print('Done.' + ' ' * 100)
    return anns, video_embd_dict

# This ensures the batches contain positive pairs, based on: 
# https://stackoverflow.com/questions/74760839/how-to-generate-batches-in-keras-for-contrastive-learning-to-ensure-positive-pai
# Makes minibatches, batching same labels together. Then makes larger batches with the minibatches.
def positive_pairs_batch_gen(X, y, cat = False, batch_size = 64, window_size = 8):
    cat = len(y.shape) > 1 # Categorical data is 2d, integer labels are 1D
    # We make sure to convert y to be type int64, this is what group_by_window expects
    data = tf.data.Dataset.from_tensor_slices((X, y.astype(np.int64)))
    data = data.group_by_window( 
        # We use the labels to group into minibatches    
        key_func=lambda _, l: tf.where(l==1)[0][0] if cat else l, # Using label (l) as batching-key    
        reduce_func=lambda _, window: window.batch(window_size),     
        window_size=window_size)

    data = data.shuffle(y.shape[0]) # Shuffle the data
    
    # Unbatch the minibatches and batch again based on batch_size
    batch_generator = data.unbatch()
    batch_generator = batch_generator.batch(batch_size)
    return batch_generator

# Computing cosine values between embeddings for given indices of X
# We split the negative & positive pair distances
def compute_cosines(model, indices, X, y, log = True):
    X_ex, y_ex = X[indices], y[indices]
    res = model.predict(X_ex, verbose = 0)
    if log:
        print('Examples:', X_ex.shape)
        print('Prediction result shape:', res.shape)
    cosines_pos, cosines_neg = [], []
    # Get the cosine distances between the examples
    cosines = cdist(res, res, metric = 'cosine')
    for i in range(len(indices)-1):
        for j in range(i+1, len(indices)): 
            cos = cosines[i,j]
            # If it's a positive pair, we add it to the positive pairs list
            if y_ex[i] == y_ex[j]: # Aka same label
                cosines_pos.append(cos)
            else: # We add negative pairs to the negative pairs list
                cosines_neg.append(cos)
    return cosines_pos, cosines_neg

def plot_sim_of_sampled_embds(model, X, y, target_num_label, labels, num_samples = 500, log = False, set_str = 'test'):
    # We plot some sampled cosine similarities (closer to zero is more similar)
    # For random negative and positive pairs and one for a positive pairs of a common sign
    filtered = np.where(y == target_num_label)[0]  # Common sign
    num_entries = filtered.shape[0]
    indices = range(y.shape[0])

    # Random indices (positive and negative pairs)
    ex_indices = np.random.choice(indices, min(num_samples, indices[-1]), replace = False)
    pos_cosines, neg_cosines = compute_cosines(model, ex_indices, X, y, log)

    # Indices selected from specific, common sign (e.g. GEBAREN-A)
    ex_indices = np.random.choice(filtered, min(num_samples, num_entries), replace = False)
    filtered_pos_cosines, _ = compute_cosines(model, ex_indices, X, y, log)

    # Making lists of the histograms and their titles to make it easier to plot
    plots_cos = [neg_cosines, pos_cosines, filtered_pos_cosines]
    plots_titles = ['Randomly selected negative pairs', 'Randomly selected positive pairs', 
                   'Positive pairs for the sign {}'.format(find_target_label(target_num_label,labels)[0])]
    # Use subplots to plot them next to each other
    fig, ax = plt.subplots(1, 3, figsize=(20,4))
    fig.suptitle('Cosine distance between positive/negative pair embeddings ({} set)'.format(set_str), fontsize = 18)
    
    # Loop over the embedding cosine distances and plot them
    for i in range(len(plots_cos)):
        cos = plots_cos[i]
        title = plots_titles[i]
        negative = 'negative' in title # Whether the plot is of negative pairs
        ax[i].set_xlim(0,1)
        # We assume that each bin will not have a higher count than some fraction of the total number of instances
        # To make the y-axis a consistent length and so we don't make the plots way too tall to see the bins
        div = 3 if negative else 4
        ax[i].set_ylim(0, round(len(cos)/div)) 
        ax[i].hist(cos, bins = 30, color = 'crimson' if negative else 'mediumseagreen')
        ax[i].set_title(title) 
        ax[i].set_xlabel('Cosine distance')
        ax[i].set_ylabel('Counts')
        # We also print the ratio of examples that make the cut below/above 0.5 (for pos pairs: below)
        thresh = 0.5
        less_or_greater = '>' if negative else '<'
        num_dist_thresh = len([c for c in cos if c >= thresh]) if negative else len([c for c in cos if c <= thresh])
        print("Ratio of {} {}= {}: {}".format(title.lower(), less_or_greater, thresh, round(num_dist_thresh/len(cos), 3)))
    plt.tight_layout()
    plt.show()

# Prepare the features of a dictionary of features for a specific video
def get_features(features_dict, ling_features = False):
    # Get the landmarks of the left and right hand
    l_features, r_features = features_dict['l_hand'], features_dict['r_hand']
    shape = l_features.shape
    l_features = l_features.reshape(shape[0], -1) # flatten x,y coordinates
    r_features = r_features.reshape(shape[0], -1)
    
    # Put the hands together in one np array
    features = np.append(l_features, r_features, axis = 1)
    if ling_features:
        diff_wrists = r_features[:,[44,45]] - l_features[:,[44,45]]
        features = np.append(features, diff_wrists, axis = 1)
    # Remove NaNs (make them zero)
    features = np.nan_to_num(features)
    return features

# We find the string equivalent of a numerical label (e.g. 123 -> 'SIGN-A')
def find_target_label(target_num_label, labels):
    target_label = [x for x in labels.items() if x[1] == target_num_label]
    if len(target_label) != 0:
        target_label = target_label[0]
    else:
        target_label = ('', -1)
    return target_label

# Get the distractors for each video and each target sign, making sure that the distractors
# Do not overlap with each other or with target sign annotations, for a given tolerance window size
def get_distractors(anns, reference_sign_embds, labels, dist_df, tolerance = 75, balanced_num_distractors = False):
    distractor_times, distractor_glosses = {}, {}
    linguistic_distances = []
    video_ids = list(anns.keys())
    for i, video in enumerate(video_ids):
        print('video {}/{}'.format(i+1, len(video_ids)), end = '\r')
        distractor_times[video] = {}
        distractor_glosses[video] = {}
        ann = anns[video]
        for sign in reference_sign_embds:
            # The sign variable above is numerical, so convert to string 
            str_label = find_target_label(sign, labels)[0]

            distractor_times[video][str_label] = []
            distractor_glosses[video][str_label] = []
            
            
            if len(str_label) == 0 or str_label not in ann: # If sign is not in train labels, we skip it
                continue
                
            # Get all annotation timestamps of the sign - if they exist
            timestamps = sorted(list(set(ann[str_label])))
            # Skip signs for which we have no timestamps at all
            if len(timestamps) == 0:
                continue
                
            # We try to balance the number of distractors (should be equal to the number of target anns)
            num_distractors = len(timestamps)
            # Get tolerance windows for target annotations 
            target_starts = [t[0] for t in timestamps]
            
            # Go over all other signs in order of similarity
            similar_signs = dist_df[str_label].sort_values().keys()
            for sim_sign in similar_signs:
                # Make sure the distractor candidate sign is a known train label, occurs in the video
                # And also is not the same as the target sign
                if sim_sign in ann and sim_sign != str_label and sim_sign in labels:
                    for a in ann[sim_sign]:
                        # Stop if we have an equal number of distractors to annotations of the target sign
                        if len(distractor_times[video][str_label]) == num_distractors:
                            break
                        # Get the tolerance window of the candidate distractor
                        candidate_distractor = (a[0]-tolerance, a[0])
                        # Make sure the distractor does not overlap with any other distractor or a target
                        overlap = False
                        # Go over the annotations (distractors or targets) that are already selected
                        # And only keep the new candidate distractor if it doesn't overlap with them
                        for chosen_anns in distractor_times[video][str_label] + target_starts:
                            ann_with_tolerance = (chosen_anns-tolerance, chosen_anns)
                            if is_overlap(ann_with_tolerance, candidate_distractor, offset = 0):
                                overlap = True
                                break
                        if not overlap:
                            distractor_times[video][str_label].append(a[0])
                            distractor_glosses[video][str_label].append(sim_sign)
                            linguistic_distances.append(dist_df[str_label][sim_sign])
                    # If we want a balanced number of distractors (num. targets = num. distractors)
                    # And there's not enough distractors, we reset the distractor list to empty
                    if balanced_num_distractors and len(distractor_times[video][str_label]) < num_distractors:
                        distractor_times[video][str_label] = []
    return distractor_times, distractor_glosses, linguistic_distances 

# This computes the TP, FP, TN, FNs for accuracy, precision and recall
# Tolerance should be provided as frames, not seconds
def compute_spotting_with_tolerance_metric(anns, video_embd_dict, reference_sign_embds, labels, distractor_times,
                                             distractor_glosses, tolerance=25, spotting_thresh=0.5, random=False):
    tp, fp, tn, fn = [0]*4
    instances = {} # FP, TN, FN, TN instances with info such as which sign we mistook for the target sign
    total_positives = 0
    for eval in ['TP', 'FN', 'FP', 'TN']:
            instances[eval] = {}
    for i, video in enumerate(anns): # Loop over the video_ids
        print('video {}/{}'.format(i+1, len(list(anns.keys()))), end = '\r')
        video_embd = video_embd_dict[video]
        ann = anns[video]
        for eval in ['TP', 'FN', 'FP', 'TN']:
            instances[eval][video] = {}
        
        # Loop over the signs and their embeddings
        for sign, reference_embd in reference_sign_embds.items():
            # The sign variable above is numerical, so convert to string 
            str_label = find_target_label(sign, labels)[0]
            if len(str_label) == 0: # If sign is not in train labels, we skip it
                continue
            timestamps = [] # Start with an empty list of timestamps of annotations
            # Get all annotation timestamps of the sign - if they exist
            if str_label in ann: 
                timestamps = sorted(list(set(ann[str_label])))
            # Skip signs for which we have no target annotations at all, or no distractors
            if len(timestamps) == 0 or len(distractor_times[video][str_label]) == 0:
                continue
            total_positives += len(timestamps)
            # Get tolerance windows for target annotations 
            target_windows = [(t[0]-tolerance, t[0]) for t in timestamps]
            if random: # Random baseline uses randomly generated distances
                dist = np.random.rand(video_embd.shape[0])
            else:
                # Compute the distance for each result of using the sliding window
                dist = cdist(video_embd, reference_embd.reshape(1,-1), metric = 'cosine').flatten()
            # Get the frames below a threshold cosine distance and make them into timespans
            below_thresh = np.where((dist < spotting_thresh) & (~np.isnan(dist)))[0]
            range_below_thresh = ranges(below_thresh, gap = 2) # Predicted spottings
            # Convert to jump-in-points as the start of the spotting plus a small constant to allow for overlap
            jump_in_points = np.array([spot[0] for spot in range_below_thresh])
            correct_jump_in_points = [] # Correct jump-in-points (JIPs for short)
            spotted_targets = []
            
            for jip in jump_in_points:
                for t in target_windows:
                    # Only count each target window as correct once
                    # So skip it if it's already been spotted
                    if t in spotted_targets:
                        continue
                    t_start, t_end = t
                    # If the tolerance and JIP match, we are done with the JIP (so we break) 
                    if t_end>=jip and t_start<=jip:
                        correct_jump_in_points.append(jip)
                        spotted_targets.append(t)
                        break

            spotted_targets = list(set(spotted_targets))
            correct_jump_in_points = set(correct_jump_in_points)
            tp += len(spotted_targets)  
            # TPs are targets which are spotted, whereas FNs are not-spotted targets
            instances['TP'][video][str_label] = spotted_targets  
            instances['FN'][video][str_label] = list(set(target_windows)-set(spotted_targets))      

            # Get JIPs which have not been matched 
            wrong_jump_in_points = list(set(jump_in_points)-correct_jump_in_points) 
            
            # We then make fake annotations in-between the target sign and keep track of which of them are spotted
            spotted_distractors = []
            distractor_starts = [(d-tolerance, d) for d in distractor_times[video][str_label]]
            distractor_time_and_gloss = []
            # Check which jump-in points that didn't match a target, match with a distractor
            for jip in wrong_jump_in_points:
                for i in range(len(distractor_starts)):
                    t_d, gloss_d = distractor_starts[i], distractor_glosses[video][str_label][i] 
                    distractor_time_and_gloss.append((t_d, gloss_d))
                    if (t_d, gloss_d) in spotted_distractors:
                        continue
                    t_start, t_end = t_d
                    if t_end>=jip and t_start<=jip: 
                        spotted_distractors.append((t_d, gloss_d))
                        break
            # Compute the number distractors
            spotted_distractors = list(set(spotted_distractors))
            num_negatives = len(distractor_times[video][str_label])
            instances['FP'][video][str_label] = spotted_distractors
            instances['TN'][video][str_label] = list(set(distractor_time_and_gloss) - set(spotted_distractors))

            fp += len(spotted_distractors) # All non-targets which are spotted are FPs
            
            # Get the TNs for the sign, video: all negatives that are not FPs are TNs
            tn += num_negatives - len(spotted_distractors)

    fn = total_positives - tp # FNs: all positives - TPs

    print('\nTP: {:<12}FP: {:<12}FN: {:<12}TN: {:<12}'.format(tp, fp, fn, tn))
    
    # Compute metrics (accuracy, precision, recall) and get total classifications
    acc = round((tp+tn)/(tp+tn+fp+fn), 3)
    precision = round(tp/(tp+fp), 3)
    recall = round(tp/(tp+fn), 3)
    total = tp+tn+fp+fn 
    targets = tp+fn
    distractors = fp+tn
    
    print('Accuracy: {}\tPrecision: {}\tRecall: {}'.format(acc, precision, recall))
    print('Total judgments: {} ({} targets, {} distractors)'.format(total, targets, distractors))
    return instances, [tp, fn, fp, tn, acc, precision, recall]