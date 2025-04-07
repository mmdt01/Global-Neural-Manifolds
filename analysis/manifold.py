"""
Neural manifold analysis functions for neural data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from .band_power import compute_band_power, get_frequency_bands

def compute_neural_manifold(region_epochs, region_channels_dict, band_name, n_components=3, 
                          downsample_factor=1, plot=True, plot_title=None, output_dir=None):
    """
    This function computes the neural manifold for each subject's region by applying PCA to the band power data.
    It also generates a 3D plot of the manifold if requested and if the number of components is 3.
    The manifold is computed for each subject separately, and the explained variance ratios are also returned.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    band_name : str
        Name of the frequency band to analyze (delta, theta, alpha, beta, low_gamma, high_gamma, broad)
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    plot : bool, optional
        Whether to plot the low-dimensional representation
    plot_title : str, optional
        Title prefix for the plots
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    manifold_dict : dict
        Dictionary mapping subject IDs to dictionaries containing:
            'manifold': array of shape (n_times, n_components) - the neural manifold
            'explained_variance': array of explained variance ratios
            'pca': fitted PCA object
    """
    # Get frequency bands dictionary
    bands = get_frequency_bands()
    
    # Check if the requested band exists
    if band_name not in bands:
        raise ValueError(f"Unknown band name: {band_name}. Available bands: {list(bands.keys())}")
    
    # Initialize results dictionary
    manifold_dict = {}
    
    # Process each subject
    for subject_id, epochs in region_epochs.items():
        print(f"\nComputing neural manifold for Subject {subject_id}, {band_name} band...")
        
        try:
            # Number of channels for this subject
            num_channels = len(region_channels_dict[subject_id])
            
            # Compute band power
            band_power = compute_band_power(epochs, band_name, downsample_factor)
            
            # Get dimensions
            n_epochs, n_channels, n_times = band_power.shape
            
            # Reshape to 2D for PCA: (n_epochs*n_channels, n_times)
            X = band_power.reshape(n_epochs * n_channels, n_times)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X.T)  # Transpose to have time points as samples
            
            # Save results
            manifold_dict[subject_id] = {
                'manifold': components,
                'explained_variance': pca.explained_variance_ratio_,
                'pca': pca
            }
            
            # Print explained variance
            print(f"Explained variance: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
            
            # Plot if requested and if 3 components
            if plot and n_components == 3:
                # Create a title
                if plot_title is None:
                    title = f"Subject {subject_id}: {band_name} Neural Manifold\n({num_channels} channels)"
                else:
                    title = f"{plot_title} - Subject {subject_id}: {band_name} band"
                
                # Create 3D plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Get downsampled time points
                times = epochs.times[::downsample_factor]
                
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
                
                # Set labels and title
                var_explained = pca.explained_variance_ratio_ * 100
                ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
                ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
                ax.set_zlabel(f"PC3 ({var_explained[2]:.1f}%)")
                
                plt.title(title)
                plt.tight_layout()
                
                # Save or show the figure
                if output_dir is not None:
                    plt.savefig(f"{output_dir}/manifold_{band_name}_subject_{subject_id}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
            
            print(f"Neural manifold computation completed for Subject {subject_id}")
            
        except Exception as e:
            print(f"Error computing neural manifold for Subject {subject_id}: {e}")
    
    # Return the manifold dictionary
    return manifold_dict

def analyze_neural_manifolds(region_epochs, region_channels_dict, region_labels, 
                           bands=None, n_components=3, downsample_factor=1, output_dir=None):
    """
    Analyze neural manifolds for region-specific epoch data across multiple frequency bands.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    bands : list, optional
        List of frequency bands to analyze. If None, uses ['delta', 'beta', 'high_gamma']
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    manifold_results : dict
        Dictionary mapping band names to manifold dictionaries
    """
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'beta', 'high_gamma']
    
    # Initialize results dictionary
    manifold_results = {}
    
    # Analyze each frequency band
    for band_name in bands:
        print(f"\n\n===== Analyzing {band_name.upper()} band neural manifolds =====")
        
        # Compute neural manifolds
        manifold_dict = compute_neural_manifold(
            region_epochs,
            region_channels_dict,
            band_name,
            n_components=n_components,
            downsample_factor=downsample_factor,
            plot=True,
            plot_title=f"Regions: {', '.join(region_labels)}",
            output_dir=output_dir
        )
        
        # Save results
        manifold_results[band_name] = manifold_dict
    
    return manifold_results

# functions for analysing neural manifolds of different gestures independently

def compute_gesture_manifolds(region_epochs, region_channels_dict, band_name, gestures, 
                            n_components=3, downsample_factor=1, plot=True, output_dir=None):
    """
    Compute low-dimensional neural manifold representations for each gesture type.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    band_name : str
        Name of the frequency band to analyze
    gestures : list
        List of gesture names to analyze
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data
    plot : bool, optional
        Whether to plot the manifolds
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    gesture_manifolds : dict
        Dictionary mapping subject IDs to dictionaries of gesture-specific manifolds
    """
    # Get frequency bands dictionary
    bands = get_frequency_bands()
    
    # Check if the requested band exists
    if band_name not in bands:
        raise ValueError(f"Unknown band name: {band_name}. Available bands: {list(bands.keys())}")
    
    # Initialize results dictionary
    gesture_manifolds = {}
    
    # Process each subject
    for subject_id, epochs in region_epochs.items():
        print(f"\nComputing gesture-specific manifolds for Subject {subject_id}, {band_name} band...")
        
        try:
            # Number of channels for this subject
            num_channels = len(region_channels_dict[subject_id])
            
            # Initialize dictionary for this subject's gesture manifolds
            gesture_manifolds[subject_id] = {}
            
            # Process each gesture
            for gesture in gestures:
                print(f"  Processing gesture: {gesture}")
                
                # Get epochs for this gesture
                gesture_epochs = epochs[gesture]
                
                # If no epochs for this gesture, skip
                if len(gesture_epochs) == 0:
                    print(f"  No epochs found for gesture {gesture}, skipping...")
                    continue
                
                # Compute band power for this gesture's epochs
                band_power = compute_band_power(gesture_epochs, band_name, downsample_factor)
                
                # Get dimensions
                n_epochs, n_channels, n_times = band_power.shape
                
                # Reshape to 2D for PCA: (n_epochs*n_channels, n_times)
                X = band_power.reshape(n_epochs * n_channels, n_times)
                
                # Apply PCA
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(X.T)  # Transpose to have time points as samples
                
                # Store the results
                gesture_manifolds[subject_id][gesture] = {
                    'manifold': components,
                    'explained_variance': pca.explained_variance_ratio_,
                    'pca': pca
                }
                
                print(f"  Explained variance for {gesture}: {pca.explained_variance_ratio_}")
                print(f"  Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
            
            # Plot all gestures for this subject if requested
            if plot and n_components == 3 and len(gesture_manifolds[subject_id]) > 0:
                plot_gesture_manifolds(
                    subject_id, 
                    gesture_manifolds[subject_id], 
                    epochs.times[::downsample_factor], 
                    band_name, 
                    output_dir
                )
            
            print(f"Gesture-specific manifold computation completed for Subject {subject_id}")
            
        except Exception as e:
            print(f"Error computing gesture manifolds for Subject {subject_id}: {e}")
    
    return gesture_manifolds

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

def analyze_gesture_manifolds(region_epochs, region_channels_dict, region_labels, 
                            bands=None, gestures=None, n_components=3, 
                            downsample_factor=1, output_dir=None):
    """
    Analyze neural manifolds for each gesture across multiple frequency bands.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    bands : list, optional
        List of frequency bands to analyze. If None, uses ['delta', 'beta', 'high_gamma']
    gestures : list, optional
        List of gesture names to analyze. If None, uses all available gestures
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    gesture_manifold_results : dict
        Dictionary mapping band names to dictionaries of gesture manifolds
    """
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'beta', 'high_gamma']
    
    # Get list of gestures if not provided
    if gestures is None:
        # Get gestures from the first subject's epochs
        first_subject_id = list(region_epochs.keys())[0]
        gestures = list(region_epochs[first_subject_id].event_id.keys())
    
    # Initialize results dictionary
    gesture_manifold_results = {}
    
    # Analyze each frequency band
    for band_name in bands:
        print(f"\n\n===== Analyzing {band_name.upper()} band gesture-specific neural manifolds =====")
        
        # Compute neural manifolds for each gesture
        gesture_manifolds = compute_gesture_manifolds(
            region_epochs,
            region_channels_dict,
            band_name,
            gestures,
            n_components=n_components,
            downsample_factor=downsample_factor,
            plot=True,
            output_dir=output_dir
        )
        
        # Save results
        gesture_manifold_results[band_name] = gesture_manifolds
    
    return gesture_manifold_results