


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_single_frame(features_df, frame_number=0, save_path='../docs/plots/single_frame.png'):

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    features = features_df[features_df["frame"] == frame_number]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the features
    ax.scatter(features['x'], features['y'], features['z'], c='blue', marker='o', alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Plot of Detected Particles in Frame {frame_number}')

    # Calculate the range of each axis to set equal aspect ratio
    x_range = features['x'].max() - features['x'].min()
    y_range = features['y'].max() - features['y'].min()
    z_range = features['z'].max() - features['z'].min()
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

    plt.savefig(save_path)
    plt.close()  # Close to free memory

def visualize_trajectories(trajectories, save_path='../docs/plots/trajectories.png'):
    """
    Visualize trajectories in 3D and save the plot.

    Parameters:
    trajectories: DataFrame with 'particle', 'x', 'y', 'z' columns
    save_path: Path to save the plot image
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Plot the trajectories in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each trajectory
    for particle_id in trajectories['particle'].unique():
        traj = trajectories[trajectories['particle'] == particle_id]
        ax.scatter(traj['x'], traj['y'], traj['z'], marker='o')
        ax.plot(traj['x'], traj['y'], traj['z'], label=f'Particle {particle_id}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectories Across Time-Steps')
    # ax.legend()

    plt.savefig(save_path)
    plt.close()  # Close to free memory