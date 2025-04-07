"""
Neural manifold visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

def plot_3d_manifold(components, times, subject_id, band_name, num_channels, title=None, output_dir=None, output_suffix=None):
    """
    Plot 3D neural manifold trajectory.
    
    Parameters:
    -----------
    components : array, shape (n_times, n_components)
        PCA components representing the neural manifold
    times : array
        Array of time points
    subject_id : int
        Subject ID
    band_name : str
        Name of the frequency band
    num_channels : int
        Number of channels used
    title : str, optional
        Custom title prefix
    output_dir : str, optional
        Directory to save plot. If None, plot is displayed but not saved.
    output_suffix : str, optional
        Additional suffix for output filename
    """
    # Create a default title if none provided
    if title is None:
        title = f"Subject {subject_id}: {band_name} Neural Manifold\n({num_channels} channels)"
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap for time
    norm = plt.Normalize(times.min(), times.max())
    cmap = sns.color_palette("crest", as_cmap=True)
    colors = cmap(norm(times))
    
    # Plot 3D trajectory
    ax.scatter(
        components[:, 0], 
        components[:, 1], 
        components[:, 2], 
        c=colors, 
        s=15, 
        alpha=0.8,
        marker='o'
    )
    
    # Add a colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Time (s)')
    
    # Mark specific time points with markers and annotations
    # Find indices of evenly spaced time points for annotation
    time_markers = np.linspace(0, len(times)-1, 5).astype(int)
    
    for idx in time_markers:
        t = times[idx]
        x, y, z = components[idx, 0], components[idx, 1], components[idx, 2]
        ax.scatter([x], [y], [z], c='red', s=50, edgecolors='black', linewidths=1)
        ax.text(x, y, z, f"{t:.2f}s", fontsize=8)
    
    # Set labels with explained variance if available
    var_explained = np.ones(3) * 100 / 3  # Default if no explained variance available
    if hasattr(components, 'explained_variance_ratio_'):
        var_explained = components.explained_variance_ratio_ * 100
    
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({var_explained[2]:.1f}%)")
    
    plt.title(title)
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/manifold_all_bands_subject_{subject_id}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.show()

def plot_manifold_comparison(manifold_dict, band_name, times, region_labels, output_dir=None):
    """
    Create a comparison plot of manifolds for multiple subjects.
    
    Parameters:
    -----------
    manifold_dict : dict
        Dictionary mapping subject IDs to manifold data
    band_name : str
        Name of the frequency band
    times : array
        Array of time points
    region_labels : list
        List of brain region names
    output_dir : str, optional
        Directory to save plot. If None, plot is displayed but not saved.
    """
    # Number of subjects
    num_subjects = len(manifold_dict)
    
    if num_subjects == 0:
        print(f"No manifold data available for {band_name} band.")
        return
    
    # Calculate grid layout
    n_rows = int(np.ceil(np.sqrt(num_subjects)))
    n_cols = int(np.ceil(num_subjects / n_rows))
    
    # Create figure
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    plt.suptitle(f"{band_name.capitalize()} Band Neural Manifolds - Regions: {', '.join(region_labels)}", 
                fontsize=16)
    
    # Create a shared colormap for time
    norm = plt.Normalize(times.min(), times.max())
    cmap = sns.color_palette("crest", as_cmap=True)
    
    # Plot each subject's manifold
    for i, (subject_id, data) in enumerate(manifold_dict.items()):
        # Create 3D subplot
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        
        # Get manifold and explained variance
        manifold = data['manifold']
        var_explained = data['explained_variance'] * 100
        
        # Plot trajectory colored by time
        colors = cmap(norm(times))
        
        # Plot 3D trajectory
        scatter = ax.scatter(
            manifold[:, 0], 
            manifold[:, 1], 
            manifold[:, 2], 
            c=colors, 
            s=10, 
            alpha=0.8,
            marker='o'
        )
        
        # Mark specific time points (start, middle, end)
        time_markers = [0, len(times)//2, len(times)-1]
        for idx in time_markers:
            t = times[idx]
            x, y, z = manifold[idx, 0], manifold[idx, 1], manifold[idx, 2]
            ax.scatter([x], [y], [z], c='red', s=30, edgecolors='black')
            ax.text(x, y, z, f"{t:.2f}s", fontsize=6)
        
        # Set labels with explained variance
        ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
        ax.set_zlabel(f"PC3 ({var_explained[2]:.1f}%)")
        
        # Set title for this subplot
        ax.set_title(f"Subject {subject_id}")
        
        # Optimize viewing angle
        ax.view_init(elev=30, azim=45)
    
    # Add a shared colorbar for time
    plt.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Time (s)')
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/manifold_comparison_{band_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.show()

def plot_gesture_manifolds(subject_id, gesture_data, times, band_name, output_dir=None):
    """
    Plot gesture-specific manifolds for a single subject.
    
    Parameters:
    -----------
    subject_id : int
        Subject ID
    gesture_data : dict
        Dictionary mapping gesture names to manifold data
    times : array
        Array of time points
    band_name : str
        Name of the frequency band
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for each gesture (using distinct colors)
    colors = {
        "elbow": "red",
        "scissor": "blue",
        "rock": "green",
        "rotation": "purple",
        "thumb": "orange"
    }
    
    # Plot each gesture's manifold
    for gesture, data in gesture_data.items():
        manifold = data['manifold']
        var_explained = data['explained_variance'] * 100
        
        # Plot 3D trajectory
        ax.plot(manifold[:, 0], manifold[:, 1], manifold[:, 2], 
               color=colors.get(gesture, "gray"), linewidth=2, label=gesture)
        
        # Mark specific time points (start, middle, end)
        time_markers = [0, len(times)//2, len(times)-1]
        for idx in time_markers:
            t = times[idx]
            x, y, z = manifold[idx, 0], manifold[idx, 1], manifold[idx, 2]
            ax.scatter([x], [y], [z], color=colors.get(gesture, "gray"), s=50, edgecolors='black')
            ax.text(x, y, z, f"{t:.2f}s", fontsize=8)
    
    # Set labels with explained variance
    first_gesture = list(gesture_data.keys())[0]
    var_explained = gesture_data[first_gesture]['explained_variance'] * 100
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({var_explained[2]:.1f}%)")
    
    # Add title and legend
    ax.set_title(f"Subject {subject_id}: {band_name} Neural Manifolds by Gesture", fontsize=14)
    ax.legend(title="Gestures", loc="upper right")
    
    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/gesture_manifolds_{band_name}_subject_{subject_id}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

