import numpy as np
import pandas as pd
from collections import defaultdict


# pure copilot code
def analyze_particle_statistics(tracks):
    """
    Analyze particle creation and disappearance statistics from tracking data.
    
    Parameters:
    tracks: DataFrame or dict with columns/keys: 'particle', 'frame', 'x', 'y', 'z', etc.
    """
    if isinstance(tracks, pd.DataFrame):
        df = tracks
    else:
        df = pd.DataFrame(tracks)
    
    # Group by particle ID
    particle_groups = df.groupby('particle')
    
    # Track lifetimes
    lifetimes = {}
    first_frames = {}
    last_frames = {}
    displacements = []
    speeds = []
    
    for particle_id, group in particle_groups:
        group = group.sort_values('frame')
        frames = group['frame'].values
        first_frames[particle_id] = frames[0]
        last_frames[particle_id] = frames[-1]
        lifetimes[particle_id] = len(frames)
        
        # Calculate displacements and speeds
        if len(group) > 1:
            for i in range(len(group) - 1):
                pos1 = group.iloc[i][['x', 'y', 'z']].values
                pos2 = group.iloc[i+1][['x', 'y', 'z']].values
                disp = np.linalg.norm(pos2 - pos1)
                displacements.append(disp)
                # Speed: displacement per frame (assuming dt=1)
                speeds.append(disp)
    
    # Find overall frame range
    all_frames = sorted(df['frame'].unique())
    min_frame = min(all_frames)
    max_frame = max(all_frames)
    
    # Count appearances and disappearances per frame
    appearances = defaultdict(int)
    disappearances = defaultdict(int)
    
    for particle_id in first_frames:
        if first_frames[particle_id] > min_frame:
            appearances[first_frames[particle_id]] += 1
    
    for particle_id in last_frames:
        if last_frames[particle_id] < max_frame:
            disappearances[last_frames[particle_id]] += 1
    
    # Print statistics
    print("=" * 60)
    print("PARTICLE TRACKING STATISTICS")
    print("=" * 60)
    print(f"\nTotal particles tracked: {len(particle_groups)}")
    print(f"Frame range: {min_frame} to {max_frame} ({max_frame - min_frame + 1} frames)")
    print(f"\nTotal appearances (new particles): {sum(appearances.values())}")
    print(f"Total disappearances (lost particles): {sum(disappearances.values())}")
    print(f"Particles present throughout: {len([p for p in first_frames if first_frames[p] == min_frame and last_frames[p] == max_frame])}")
    
    print(f"\nAverage particle lifetime: {np.mean(list(lifetimes.values())):.2f} frames")
    print(f"Median particle lifetime: {np.median(list(lifetimes.values())):.2f} frames")
    print(f"Min/Max lifetime: {min(lifetimes.values())} / {max(lifetimes.values())} frames")
    
    print(f"\nAverage appearances per frame: {np.mean(list(appearances.values())) if appearances else 0:.2f}")
    print(f"Average disappearances per frame: {np.mean(list(disappearances.values())) if disappearances else 0:.2f}")
    
    # Show frames with most changes
    if appearances:
        max_appear_frame = max(appearances.items(), key=lambda x: x[1])
        print(f"\nMost appearances in single frame: {max_appear_frame[1]} (frame {max_appear_frame[0]})")
    
    if disappearances:
        max_disappear_frame = max(disappearances.items(), key=lambda x: x[1])
        print(f"Most disappearances in single frame: {max_disappear_frame[1]} (frame {max_disappear_frame[0]})")
    
    # Displacement and speed statistics
    if displacements:
        print(f"\nDisplacement statistics:")
        print(f"  Mean displacement: {np.mean(displacements):.2f} pixels")
        print(f"  Median displacement: {np.median(displacements):.2f} pixels")
        print(f"  Min/Max displacement: {min(displacements):.2f} / {max(displacements):.2f} pixels")
        
        print(f"\nSpeed statistics (displacement per frame):")
        print(f"  Mean speed: {np.mean(speeds):.2f} pixels/frame")
        print(f"  Median speed: {np.median(speeds):.2f} pixels/frame")
        print(f"  Min/Max speed: {min(speeds):.2f} / {max(speeds):.2f} pixels/frame")
    else:
        print("\nNo displacements calculated (particles not tracked across frames).")
    
    print("=" * 60)
    
    return {
        'total_particles': len(particle_groups),
        'total_appearances': sum(appearances.values()),
        'total_disappearances': sum(disappearances.values()),
        'lifetimes': lifetimes,
        'appearances_by_frame': dict(appearances),
        'disappearances_by_frame': dict(disappearances),
        'displacements': displacements,
        'speeds': speeds
    }

# Example usage:
# stats = analyze_particle_statistics(tracked_data)