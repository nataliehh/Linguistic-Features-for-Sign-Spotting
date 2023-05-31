# Imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
import re
from collections import Counter
import datetime
import time

np.random.seed(123) # Set random seed for consistency

# Getting the mean, std and median of a given (numpy) array, also the max and min if extra = True
def stats(arr, extra = False):
    if extra:
        return (np.mean(arr), np.std(arr), np.median(arr), np.max(arr), np.min(arr))
    return (np.mean(arr), np.std(arr), np.median(arr))

# Compute some statistics about the sign durations
def get_duration_stats(time_deltas):
    mean, std, median, max_, min_ = stats(time_deltas, extra = True)
    stats_ = [mean, median, max_, min_]
    stats_str = ['Avg', 'Median', 'Max', 'Min']
    for s in range(len(stats_)):
        stat = stats_[s]
        str_ = stats_str[s]
        print('{} timespan length:\t~{} frames \t({} ms)'.format(str_, ms_delta_to_frame(stat), round(stat,1)))

# Print the mean, std and median of a given (numpy) array
def print_stats(arr, round_val = 2):
    mean, std, median = stats(arr)
    print('mean:\t{}\tstd:\t{}\tmedian:\t{}'.format(round(mean,round_val), round(std,round_val), round(median,round_val)))

# Loading all numpy landmark arrays associated with a specific target video
def load_numpy(paths, target_vid):
    # Get which tuples in paths of format (root, name) have a specific target video title in their name
    target_paths = [p for p in paths if target_vid in p[1]]
    dictionary = {}
    for p in target_paths:
        root, name = p
        # Cuts to only the 'key' of the numpy array (i.e. which landmarks it is presenting)
        # E.g. the keys 'face', 'l_hand', 'r_hand', 'pose' and 'video_dim'
        key = name[14:].replace('.npy', '') 
        dictionary[key] = np.load(os.path.join(root, name))
    return dictionary

# Loading a pickled dictionary from a path
def load_dict(path):
    with (open(path, "rb")) as f:
        dictionary = pickle.load(f)
        f.close()
    return dictionary

# Storing a pickled dictionary for given path
def store_dict(path, contents):
    with (open(path, "wb")) as f:
        pickle.dump(contents, f)
        f.close()

# Append to dict_[key] (value = list) if it exists, if not, create a list
def append_to_list_in_dict(dict_, key, val):
    if key not in dict_: # Create list if it doesn't exist yet
        dict_[key] = []
    dict_[key] += [val]
    return dict_

# Add to the value of dict_[key] if it exists, if not, initialize the value
def add_to_val_in_dict(dict_, key, val, base_val = 0):
    if key not in dict_: # Initialize to base_val if it doesn't exist yet
        dict_[key] = base_val 
    dict_[key] += val
    return dict_

# For a given dict, either change the value for an existing key (appending or adding)
# Or instantiate the key
def manipulate_dict_entry(dict_, key, val, append = True):
    if append:
        return append_to_list_in_dict(dict_, key, val)
    return add_to_val_in_dict(dict_, key, val)

# Convert each element of the tuple to int
def to_int(tup):
    return tuple([int(el.replace('ts', '')) for el in tup])

# Overlap calculation
# Based on: https://stackoverflow.com/questions/9044084/efficient-date-range-overlap-calculation
def compute_overlap(t1, t2, offset = 0):
    max1 = max(t1[0], t2[0]) - offset
    min2 = min(t1[1], t2[1]) + offset
    delta = min2-max1
    return max(0, delta)

# Check if there's overlap between two timespans
# Which is the case if the overlap is not zero (> 0)
def is_overlap(t1, t2, offset = 0):
    return compute_overlap(t1, t2, offset) != 0

# Check how much overlap there is between the timestamps
def overlap_ratio(t1, t2, offset = 0):
    # Get total duration of the timespans by computing the diff. between the earliest and latest timestamp
    min_start, max_end = min(t1[0], t2[0]), max(t1[1], t2[1])
    total_duration = max_end - min_start
    
    # Get the overlap 
    overlap = compute_overlap(t1, t2, offset)
    
    # Return the ratio of the overlap w.r.t. the total duration
    return overlap/total_duration

# Fuse the annotation ann_values (they are currently stored per EAF file)
def fuse_ann_values(ann_vals):
    ann_vals_fused = {}
    for eaf_file in ann_vals:
        anns = ann_vals[eaf_file]
        for key_ann in anns:
            if key_ann in ann_vals_fused:
                ann_vals_fused[key_ann] += anns[key_ann]
            else:
                ann_vals_fused[key_ann] = anns[key_ann]
    return ann_vals_fused

# Separate the timestamps and their annotation values into two separate lists
def split_ann_values(ann_values):
    timeframes, repeat_annotations = [], []
    for key in ann_values:
        # Repeat the annotation based on its frequency (once for each occurrence)
        # E.g. for an annotation that occurs twice we repeat the annotation two times 
        if len(ann_values[key]) == 0 or len(key) == 0:
            continue
        repeat_ann = [key] * len(ann_values[key])
        timeframes += ann_values[key]
        repeat_annotations += repeat_ann
    return timeframes, repeat_annotations

def store_anns(ann_dict, ann_object, timeslots, tier, distinguish_tier = False):
    # Get the annotations (first element in the tuple)
    annotations = ann_object.tiers[tier][0]

    # Loop through the annotations
    for ann in annotations.values():
        # ann format: (start timestamp, end timestamp, annotation value)
        if len(ann[2]) == 0: # Skip empty annotations
            continue
        # Get the start and end timestamps
        new_t = (timeslots[ann[0]], timeslots[ann[1]])
        if distinguish_tier:
            new_t = (timeslots[ann[0]], timeslots[ann[1]], tier)

        # Try to append the timestamp tuple to the timestamps for that annotation value
        ann_dict = manipulate_dict_entry(ann_dict, ann[2], new_t)

    # return ann_dict

# Get the gloss values that are annotated in an .eaf file (annotation object)
# This specifically distinguishes between signs performed with one and two hands
def get_gloss_vals(ann_object, distinguish_tier = False):
    # List of all (hand) annotation values in the file, mouthings are stored separate
    ann_values, mouthings = {}, {}
    timeslots = ann_object.timeslots

    # Loop through the tiers of the different annotated parts (e.g. left hand, right hand)
    for tier in ann_object.tiers:
        if tier.startswith('Gloss'): # 'Gloss' tiers are tiers which contain hand annotations
            store_anns(ann_values, ann_object, timeslots, tier, distinguish_tier)
        if tier.startswith('Mouth '): # 'Mouth' tier contains mouthings
            store_anns(mouthings, ann_object, timeslots, tier, distinguish_tier)
    return ann_values, mouthings

# Getting the tiers given an annotation object
def get_tiers(ann_object):
    return list(ann_object.tiers)

# Get the manual simultaneity and hand distinction in the same dictionary
def man_sim_and_hand_dist(anns, manual_sim = True, hand_distinct = True, filtering = True, two_hand_suffix = True):
    ann_values = anns.copy() # Copy to not change the original dictionary
    overlap, timespans, man_sim_lst, hand_dist_lst = [[] for _ in range(4)]
    
    for key in ann_values:
        # Add the key as an element to each tuple of timespans
        tup_with_key = [(v)+(key,) for v in ann_values[key]]
        timespans += tup_with_key
    # Sort by the first element in the tuple (aka the start of the timespan)
    timespans = sorted(timespans, key = lambda x: x[0])
    timespan_len = len(timespans)

    for i in range(timespan_len-1): # Iterate over the timespan tuples
        t0 = timespans[i]
        for j in range(i+1, timespan_len): # Iterate over all timespans following the selected one
            t1 = timespans[j]
            # Manual sim. should have strict overlap, offset = 0
            if manual_sim and is_overlap(t0[:2], t1[:2]) and t0[-1] != t1[-1]:
                # If there's overlap and the annotation keys are not the same
                # We have manual sim., so we store which timespans to add as manually simultaneous
                overlap += [t0, t1]
                # We make the new timespan as large as the two other timespans combined
                ts_start, ts_end = min(t0[0], t1[0]), max(t0[1], t1[1])
                man_sim_lst += [(ts_start, ts_end, t0[-1], t1[-1])]
            if is_overlap(t0[:2], t1[:2]) and t0[-1] == t1[-1]:
                # Make sure the new timespan is as large as the other two are when combined
                ts_start, ts_end = min(t0[0], t1[0]), max(t0[1], t1[1])
                
                if hand_distinct:
                    overlap += [t0, t1] # Delete both overlapping signs
                    # Add back a two-handed instance that spans both of them
                    two_hand_str = '__2H' if two_hand_suffix else ''
                    hand_dist_lst += [(ts_start, ts_end, t1[-1]+two_hand_str)] 
                else: # If we don't want to distinguish hands
                    # We check if we want to drop one of the instances that's overlapping
                    if filtering: 
                        overlap += [t0, t1] # Delete both overlapping signs
                        # Add back one of the instances, but don't note that it's two-handed
                        hand_dist_lst += [(ts_start, ts_end, t1[-1])] 
                    else: # If we don't want to filter out overlapping signs at all, we do nothing here
                        pass
                        
    # Loop over the timespans found to overlap
    for o in overlap:
        key, val = o[-1], o[:-1]
        # Remove these manually simultaneous values from the dictionary
        if key in ann_values and val in ann_values[key]:
            ann_values[key].remove(val)
            if len(ann_values[key]) == 0: # Delete the entry for the key if it's now empty
                del ann_values[key]

    # Loop over the timespans with manual simultaneity
    # To add them to the annotation dictionary
    for a in man_sim_lst:
        # The two signs that overlap are fused using '&&' (a&&b -> a and b)
        # We sort the signs alphabetically to ensure consistent naming
        key = '&&'.join(sorted(a[-2:]))
        if '&&' in key: # if we somehow only find one sign, we skip it
            val = a[:-2]
            if key in ann_values:
                for val_old in ann_values[key]:
                    if is_overlap(val_old, val):
                        continue
                ann_values[key] += [val]
            else: # Just add the key if it doesn't exist yet
                ann_values[key] = [val]
    for h in hand_dist_lst:
        key, val = h[-1], h[:-1]
        ann_values = manipulate_dict_entry(ann_values, key, val)
    return ann_values
    
# Given a list of (key, value) counts, we plot them
def plot_counts(counts, clean = False):
    # Creates a list of glosses and a list of counts
    zipped = list(zip(*counts))
    keys, values = list(zipped[0]), list(zipped[1])

    # clean param removes labels and ticks from x-axis
    if clean:
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

    sns.barplot(x = keys, y = values)
    plt.xlabel('value')
    plt.ylabel('count')
    plt.show()

# Merge two dictionaries in a specific way
# We store a triple: (dict_1_value, dict_2_value, dict_1_value + dict_2_value)
def merge_dicts(dict1, dict2):
    for key, val in dict2.items():
        if key in dict1:
            dict1[key] = (dict1[key], val, dict1[key] + val)
        else:
            dict1[key] = (0, val, val)
    return dict1

# Converting time deltas (time diffs) from milliseconds to frames
# WARNING: should not be used to convert CNGT timestamps to frames
# Because there are alignment issues with the first 3 seconds of the video
def ms_delta_to_frame(ms):
    seconds = ms/1000
    return round(25*seconds)

# Computing the counts for a two handed sign:
# Once how many weak hand drops it has
# And once how many times it is signed normally
def get_weak_hand_drop_counts(weak_hand_drop, one_handed, two_handed):
    # Getting the glosses for which there is weak hand drop
    weak_hand_drop_names = weak_hand_drop['Annotation ID Gloss (Dutch)'].tolist()
    # Getting the one-handed instances of these glosses
    weak_hand_filter = [h for h in one_handed if h in weak_hand_drop_names]
    # Count the one-handed instances
    weak_hand_counts = Counter(weak_hand_filter)
    # Sort the one-handed instances by count
    sorted_one_hand = sorted(weak_hand_counts.items(), key=lambda item: item[1], reverse = True)

    # Getting the counts of the signs that have weak hand drop
    # But now, only the counts where they *don't* have weak hand drop
    two_handed_no_drop = [n+'__2H' for n in weak_hand_drop_names]
    two_hand_filter = [h for h in two_handed if h in two_handed_no_drop]
    two_hand_counts = Counter(two_hand_filter)
    sorted_two_hand = sorted(two_hand_counts.items(), key=lambda item: item[1], reverse = True)
    return sorted_one_hand, sorted_two_hand

# Remove a substring from the annotation string if the substring != the whole string
def remove_substring(ann, substrs: list):
    for s in substrs:
        if s in ann and ann != s: 
            ann = ann.replace(s, '')
    return ann

# CNGT annotations don't always fully match with Signbank glosses because of added markers
# (e.g. ~ for negation), so we process the annotations to better match with the glosses
def process_ann_values(ann_values):
    lst = []
    for ann in ann_values:
        # Remove markers for two-handedness, negations if they are not the whole annotation
        ann = remove_substring(ann, ['__2H', '~'])
        # Remove markers of uncertainty and being out of frame completely
        if '!' in ann: ann = ann.replace('!', '')
        if '?' in ann: ann = ann.replace('?', '')
        # For the combined signs, we split them into their parts (A+B -> A and B)
        # We do the same for manual simultaneity (A&&B -> A and B)
        if '+' in ann or '&&' in ann:
            ann_split = re.split('[+]|[&]{2}', ann)
            for a in ann_split:
                if len(a) > 0:
                    lst.append(a)
            continue
        # Fingerspellings (#) are split and we add the # to each letter (e.g. #BTW -> #B #T #W)
        if ann.startswith('#'):
            ann_split = ann.split('#')[1]
            for a in ann_split:
                lst.append('#'+a)
            continue
        lst.append(ann)
        
    lst = list(set(lst)) # Make sure we remove duplicates
    return lst

# Counting the annotations of specific types
def count_substrings(sub_str, lst):
    return sum(sub_str in s for s in lst)

# Get which annotation values intersect between two lists
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return list(set(lst3))

# Find all files accessible from a given path, filtering for an (optional) extension
def find_files(path, extension = '', name_filter = ''): # Leave extension emtpy to find all files on path
    path_tuples = [(root,name) for root, dirs, files in os.walk(path) for name in files if name.endswith(extension)]
    if len(name_filter) != 0: # If a filter for the filename is specified, filter out matches
        path_tuples = [tup for tup in path_tuples if not name_filter in tup[1]]
    return path_tuples

### TIME-BASED CONVERSIONS
# Returns mm:ss timestamp and also the milliseconds, given the frame number
def frame_to_timestamp(frame, cngt = True):
    seconds = (frame/25) + 3 * cngt
    # Should we round the seconds or convert to int (will round down)?
    return time.strftime("%M:%S", time.gmtime(round(seconds))), seconds * 1000

# Given the milliseconds, it returns which frame it's equivalent to
def ms_to_frame(ms, cngt = True):
    seconds = ms/1000
    frame = 25*(seconds-3 * cngt) # -3 if it's cngt footage (we ignore the 3 second disclaimer at the start)
    return round(frame)

def start_end_ts_to_frames(ts1, ts2):
    return ms_to_frame(ts1), ms_to_frame(ts2)

def print_frames_and_ts(f1, f2):
    print(f1, f2, frame_to_timestamp(f1), frame_to_timestamp(f2))

def sec_to_timestamp(n):
    return str(datetime.timedelta(seconds = n))
