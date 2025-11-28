import trackpy as tp
import pandas as pd
import os

def detect(images, diameter = 21, minmass = 500_000, subsample_factor=1, reprocess = False, kwargs={}):
    image_names = list(images.keys())
    all_features = []

    # Ensure checkpoint directory exists
    os.makedirs('../checkpoint_data', exist_ok=True)

    for i, name in enumerate(image_names):
        base_name = name.replace('.tif', '')  # Strip .tif extension
        checkpoint_path = f'../checkpoint_data/features_{base_name}.csv'
        if os.path.exists(checkpoint_path) and not reprocess:
            print(f"Loading features for {name} from checkpoint")
            feats = pd.read_csv(checkpoint_path)
            feats['frame'] = i  # Ensure frame is set
        else:
            print(f"Processing image: {name}")
            img = images[name]
            img_subsampled = img[::subsample_factor, ::subsample_factor, ::subsample_factor]
            feats = tp.locate(img_subsampled, diameter=diameter, minmass=minmass, **kwargs)
            feats['frame'] = i  # Assign frame number
            # Save to checkpoint
            feats.to_csv(checkpoint_path, index=False)

        all_features.append(feats)

    # Concatenate into one DataFrame
    features_df = pd.concat(all_features, ignore_index=True)
    return features_df

def track(features, search_range = 35):
    # Link trajectories across frames
    linked = tp.link_df(features, search_range=search_range, memory=0)  # Adjust search_range as needed

    # Filter for trajectories present in both frames
    trajectories = linked.groupby('particle').filter(lambda x: len(x) == 2)

    return trajectories

def detect_and_track(images, diameter = 21, minmass = 500_000, subsample_factor=2, search_range = 35, reprocess = False, kwargs={}):
    features = detect(images,diameter = diameter, minmass = minmass, subsample_factor = subsample_factor, reprocess = reprocess, kwargs = kwargs)
    return track(features, search_range=search_range)