"""
Functions for visualizing high-dimensional neural data using dimensionality reduction techniques.
Includes t-SNE, UMAP and PCA visualizations for gesture trials in channel space.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import warnings

# Define a set of visually distinct colors for gestures
GESTURE_COLORS = {
    'elbow': '#FF5733',      # Red-orange
    'scissor': '#33FF57',    # Green
    'rock': '#3357FF',       # Blue
    'rotation': '#FF33F5',   # Pink
    'thumb': '#33FFF5',      # Cyan
}

# Define markers for different gestures
GESTURE_MARKERS = {
    'elbow': 'o',       # Circle
    'scissor': '^',     # Triangle up
    'rock': 's',        # Square
    'rotation': 'D',    # Diamond
    'thumb': 'P',       # Plus filled
}

def prepare_trial_data(epochs_dict, gestures=None, band_name='delta'):
    """
    Prepare trial data for visualization by extracting time-averaged activity
    for each trial and organizing them by gesture.
    
    Parameters:
    -----------
    epochs_dict : dict or mne.Epochs
        Either a dictionary mapping band names to mne.Epochs objects,
        or a single mne.Epochs object
    gestures : list, optional
        List of gesture names to analyze. If None, uses all available gestures
    band_name : str, optional
        Name of the frequency band to use if epochs_dict is a dictionary
    
    Returns:
    --------
    trial_data : dict
        Dictionary mapping gesture names to arrays of trial data (trials x channels)
    all_trials : np.ndarray
        Array containing all trials (all_trials x channels)
    all_labels : list
        List of gesture labels corresponding to each trial in all_trials
    """
    # Handle either band dictionary or direct epochs object
    if isinstance(epochs_dict, dict) and band_name in epochs_dict:
        epochs = epochs_dict[band_name]
    else:
        epochs = epochs_dict
    
    # Get list of gestures if not provided
    if gestures is None:
        gestures = list(epochs.event_id.keys())
    
    # Initialize dictionaries to store results
    trial_data = {}
    
    # Lists to store all trials and their labels
    all_trials = []
    all_labels = []
    
    # Process each gesture
    for gesture in gestures:
        try:
            # Extract epochs for this gesture
            gesture_epochs = epochs[gesture]
            
            # If there are no epochs for this gesture, skip it
            if len(gesture_epochs) == 0:
                print(f"No epochs found for gesture: {gesture}, skipping...")
                continue
            
            # Get the data: shape is (trials, channels, time)
            data = gesture_epochs.get_data()
            
            # Compute the time-average for each trial: shape will be (trials, channels)
            time_averaged_trials = data.mean(axis=2)
            
            # Store the trial data
            trial_data[gesture] = time_averaged_trials
            
            # Add to all_trials and all_labels
            all_trials.append(time_averaged_trials)
            all_labels.extend([gesture] * len(time_averaged_trials))
            
        except KeyError:
            print(f"Gesture '{gesture}' not found in epochs, skipping...")
    
    # Convert all_trials to numpy array
    if all_trials:
        all_trials = np.vstack(all_trials)
    else:
        all_trials = np.array([])
    
    return trial_data, all_trials, all_labels

def apply_tsne(data, perplexity=None, n_components=3, random_state=42):
    """
    Apply t-SNE dimensionality reduction to the data.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array with shape (n_samples, n_features)
    perplexity : float, optional
        Perplexity parameter for t-SNE. If None, uses sqrt(n_samples)
    n_components : int, optional
        Number of dimensions in the embedded space (2 or 3)
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    embedded : np.ndarray
        Embedded data with shape (n_samples, n_components)
    """
    # Check if we have enough data
    if data.shape[0] < 10:
        warnings.warn("Very few samples for t-SNE. Results may be unreliable.")
    
    # Set default perplexity if not provided
    if perplexity is None:
        # Rule of thumb: perplexity should be around sqrt(n_samples)
        perplexity = min(30, max(5, int(np.sqrt(data.shape[0]))))
    
    print(f"Running t-SNE with perplexity={perplexity}, n_components={n_components}")
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=1000,
        random_state=random_state,
        init='pca'  # Using PCA initialization for more stable results
    )
    
    # Handle potential memory issues with large datasets
    try:
        embedded = tsne.fit_transform(data)
    except MemoryError:
        print("Memory error during t-SNE. Applying PCA first to reduce dimensions...")
        # Use PCA to reduce to 50 dimensions first
        pca = PCA(n_components=min(50, data.shape[1]))
        reduced_data = pca.fit_transform(data)
        print(f"Reduced data from {data.shape[1]} to {reduced_data.shape[1]} dimensions")
        embedded = tsne.fit_transform(reduced_data)
    
    return embedded

def plot_tsne_3d(embedded, labels, title=None, subject_id=None, region_label=None,
                output_dir=None, show_hull=True, alpha=0.7, figsize=(12, 10),
                gesture_colors=None, gesture_markers=None):
    """
    Create a 3D visualization of t-SNE results with each gesture colored differently.
    
    Parameters:
    -----------
    embedded : np.ndarray
        Embedded data with shape (n_samples, 3)
    labels : list
        List of gesture labels for each sample
    title : str, optional
        Title for the plot
    subject_id : str or int, optional
        Subject ID for the plot title
    region_label : str, optional
        Brain region label for the plot title
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    show_hull : bool, optional
        Whether to show convex hulls around each gesture cluster
    alpha : float, optional
        Transparency level for points
    figsize : tuple, optional
        Figure size
    gesture_colors : dict, optional
        Dictionary mapping gesture names to colors
    gesture_markers : dict, optional
        Dictionary mapping gesture names to marker shapes
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the visualization
    """
    if embedded.shape[1] != 3:
        raise ValueError(f"Expected 3D data but got {embedded.shape[1]} dimensions")
    
    # Use default colors and markers if not provided
    if gesture_colors is None:
        gesture_colors = GESTURE_COLORS
    
    if gesture_markers is None:
        gesture_markers = GESTURE_MARKERS
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique gestures
    unique_gestures = sorted(set(labels))
    
    # Plot each gesture cluster
    for gesture in unique_gestures:
        # Get indices for this gesture
        indices = [i for i, label in enumerate(labels) if label == gesture]
        
        # Skip if no trials for this gesture
        if not indices:
            continue
        
        # Get color and marker for this gesture
        color = gesture_colors.get(gesture, 'gray')  # Default to gray if gesture not in dict
        marker = gesture_markers.get(gesture, 'o')   # Default to circle if gesture not in dict
        
        # Extract data for this gesture
        x = embedded[indices, 0]
        y = embedded[indices, 1]
        z = embedded[indices, 2]
        
        # Plot the points
        ax.scatter(x, y, z, c=color, marker=marker, s=80, alpha=alpha, label=gesture)
        
        # Add convex hull if requested and we have enough points
        if show_hull and len(indices) >= 4:
            try:
                # Compute convex hull
                hull = ConvexHull(embedded[indices, :])
                
                # Get hull vertices
                for simplex in hull.simplices:
                    # Get vertices coordinates
                    v_x = x[simplex]
                    v_y = y[simplex]
                    v_z = z[simplex]
                    
                    # Plot hull face
                    ax.plot_trisurf(v_x, v_y, v_z, color=color, alpha=0.1)
            except Exception as e:
                print(f"Could not compute convex hull for {gesture}: {e}")
    
    # Set title
    if title is None:
        title = "t-SNE Visualization of Gesture Trials in Channel Space"
        if subject_id is not None and region_label is not None:
            title += f"\nSubject {subject_id}, Region: {region_label}"
        elif subject_id is not None:
            title += f"\nSubject {subject_id}"
        elif region_label is not None:
            title += f"\nRegion: {region_label}"
    
    ax.set_title(title, fontsize=14)
    
    # Set labels
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_zlabel('t-SNE Component 3', fontsize=12)
    
    # Add legend
    ax.legend(title="Gestures", loc='best', fontsize=10)
    
    # Set background color to white for better visibility
    ax.set_facecolor('white')
    
    # Enhance grid for better depth perception
    ax.xaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.5)
    ax.yaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.5)
    ax.zaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.5)
    
    # Optimize view angle
    ax.view_init(elev=30, azim=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir is not None:
        output_file = "tsne_3d_visualization"
        if subject_id is not None:
            output_file += f"_sub-{subject_id}"
        if region_label is not None:
            # Check if this is a brain-wide analysis with multiple regions
            if ',' in region_label and len(region_label) > 30:
                # We have multiple regions, use a shorter identifier
                region_count = len(region_label.split(','))
                output_file += f"_brain-wide_{region_count}_regions"
            else:
                # Single region or short list, use the original approach
                region_str = region_label.replace(', ', '_').replace(' ', '_')
                output_file += f"_{region_str}"
        output_file += ".png"
        
        plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches='tight')
    
    return fig

def plot_tsne_2d(embedded, labels, title=None, subject_id=None, region_label=None,
               output_dir=None, show_hull=True, alpha=0.7, figsize=(12, 10),
               gesture_colors=None, gesture_markers=None):
    """
    Create a 2D visualization of t-SNE results with each gesture colored differently.
    
    Parameters:
    -----------
    embedded : np.ndarray
        Embedded data with shape (n_samples, 2) or (n_samples, 3)
    labels : list
        List of gesture labels for each sample
    title : str, optional
        Title for the plot
    subject_id : str or int, optional
        Subject ID for the plot title
    region_label : str, optional
        Brain region label for the plot title
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    show_hull : bool, optional
        Whether to show convex hulls around each gesture cluster
    alpha : float, optional
        Transparency level for points
    figsize : tuple, optional
        Figure size
    gesture_colors : dict, optional
        Dictionary mapping gesture names to colors
    gesture_markers : dict, optional
        Dictionary mapping gesture names to marker shapes
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the visualization
    """
    # If we have 3D data, use only the first two dimensions
    if embedded.shape[1] > 2:
        embedded = embedded[:, :2]
        print("Warning: Using only the first 2 dimensions for 2D visualization")
    
    # Use default colors and markers if not provided
    if gesture_colors is None:
        gesture_colors = GESTURE_COLORS
    
    if gesture_markers is None:
        gesture_markers = GESTURE_MARKERS
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique gestures
    unique_gestures = sorted(set(labels))
    
    # Plot each gesture cluster
    for gesture in unique_gestures:
        # Get indices for this gesture
        indices = [i for i, label in enumerate(labels) if label == gesture]
        
        # Skip if no trials for this gesture
        if not indices:
            continue
        
        # Get color and marker for this gesture
        color = gesture_colors.get(gesture, 'gray')  # Default to gray if gesture not in dict
        marker = gesture_markers.get(gesture, 'o')   # Default to circle if gesture not in dict
        
        # Extract data for this gesture
        x = embedded[indices, 0]
        y = embedded[indices, 1]
        
        # Plot the points
        ax.scatter(x, y, c=color, marker=marker, s=100, alpha=alpha, label=gesture)
        
        # Add convex hull if requested and we have enough points
        if show_hull and len(indices) >= 3:
            try:
                from scipy.spatial import ConvexHull
                points = embedded[indices, :2]
                hull = ConvexHull(points)
                
                # Get hull vertices in order
                hull_indices = hull.vertices
                hull_x = points[hull_indices, 0]
                hull_y = points[hull_indices, 1]
                
                # Close the polygon
                hull_x = np.append(hull_x, hull_x[0])
                hull_y = np.append(hull_y, hull_y[0])
                
                # Plot the hull
                ax.fill(hull_x, hull_y, color=color, alpha=0.1)
                ax.plot(hull_x, hull_y, color=color, alpha=0.5)
            except Exception as e:
                print(f"Could not compute convex hull for {gesture}: {e}")
    
    # Set title
    if title is None:
        title = "t-SNE Visualization of Gesture Trials in Channel Space"
        if subject_id is not None and region_label is not None:
            title += f"\nSubject {subject_id}, Region: {region_label}"
        elif subject_id is not None:
            title += f"\nSubject {subject_id}"
        elif region_label is not None:
            title += f"\nRegion: {region_label}"
    
    ax.set_title(title, fontsize=14)
    
    # Set labels
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    
    # Add legend
    ax.legend(title="Gestures", loc='best', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Set background color
    ax.set_facecolor('white')
    
    # Add a subtle border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir is not None:
        output_file = "tsne_2d_visualization"
        if subject_id is not None:
            output_file += f"_sub-{subject_id}"
        if region_label is not None:
            # Check if this is a brain-wide analysis with multiple regions
            if ',' in region_label and len(region_label) > 30:
                # We have multiple regions, use a shorter identifier
                region_count = len(region_label.split(','))
                output_file += f"_brain-wide_{region_count}_regions"
            else:
                # Single region or short list, use the original approach
                region_str = region_label.replace(', ', '_').replace(' ', '_')
                output_file += f"_{region_str}"
        output_file += ".png"
        
        plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches='tight')
    
    return fig


