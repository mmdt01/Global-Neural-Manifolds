"""
Neural manifold analysis functions for neural data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import logging
from scipy.linalg import qr, svd, inv

from .band_power import compute_band_power, get_frequency_bands

def compute_neural_manifold(region_epochs, region_channels_dict, band_name, n_components=3,
                            plot=True, plot_title=None, output_dir=None):
    """
    This function computes the neural manifold for each subject's region by applying PCA to the band power data.
    It uses the pre-computed band power data directly from the epochs objects and mean-centers each channel
    to focus on covariations rather than absolute activity levels.
    
    Parameters:
    -----------
    region_epochs : dict
        Nested dictionary mapping subject IDs to frequency bands to epochs objects with band power
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    band_name : str
        Name of the frequency band to analyze
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA (usually 1 as no downsampling needed)
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
    # Initialize results dictionary
    manifold_dict = {}
    
    # Process each subject
    for subject_id, band_epochs in region_epochs.items():
        # Skip if this subject doesn't have data for the specified band
        if band_name not in band_epochs:
            print(f"No {band_name} band data for Subject {subject_id}, skipping...")
            continue
        
        print(f"\nComputing neural manifold for Subject {subject_id}, {band_name} band...")
        
        try:
            # Number of channels for this subject
            num_channels = len(region_channels_dict[subject_id])
            print(f"Total number of channels: {num_channels}")
            
            # Get the specific band's epochs object
            epochs = band_epochs[band_name]
            
            # Extract the data directly - it's already band power
            band_power = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
            
            # Get dimensions
            n_epochs, n_channels, n_times = band_power.shape

            # Mean-center each channel across epochs and time: this removes baseline differences between channels and focuses on covariations
            print(f"Mean-centering {n_channels} channels...")
            band_power_centered = np.zeros_like(band_power)
            
            for ch_idx in range(n_channels):
                # Compute mean activity for this channel across all epochs and time points
                channel_mean = np.mean(band_power[:, ch_idx, :])
                
                # Subtract the mean from all data points for this channel
                band_power_centered[:, ch_idx, :] = band_power[:, ch_idx, :] - channel_mean
                
                print(f"  Channel {ch_idx}: mean activity = {channel_mean:.4f}")
            
            # Reshape to 2D for PCA: (n_epochs*n_channels, n_times)
            X = band_power_centered.reshape(n_epochs * n_channels, n_times)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X.T)  # Transpose to have time points as samples
            
            # Save results
            manifold_dict[subject_id] = {
                'manifold': components,
                'explained_variance': pca.explained_variance_ratio_,
                'pca': pca,
                'mean_centered_data': band_power_centered 
            }
            
            # Print explained variance
            print(f"Explained variance: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
            
            # Plot if requested and if 3 components
            if plot and n_components == 3:
                # Create a title
                if plot_title is None:
                    title = f"Subject {subject_id}: {band_name} Neural Manifold\n({num_channels} channels, mean-centered)"
                else:
                    title = f"{plot_title} - Subject {subject_id}: {band_name} band (mean-centered)"
                
                # Create 3D plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Get time points
                times = epochs.times
                
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

# functions for analysing manifolds of different gestures independently

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

# functions for aligning manifolds using CCA

def canoncorr(X:np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)
    
    returns: A,B,r,U,V 
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations
    
    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Center the variables
    X = X - np.mean(X,0)
    Y = Y - np.mean(Y,0)

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1,T11,perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)

    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0,0])))).eps*max([n,p1]))

    if rankX == 0:
        logging.error(f'stats:canoncorr:BadData = X')
    elif rankX < p1:
        logging.warning('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:,:rankX]
        T11 = T11[:rankX,:rankX]

    Q2,T22,perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0,0])))).eps*max([n,p2]))

    if rankY == 0:
        logging.error(f'stats:canoncorr:BadData = Y')
    elif rankY < p2:
        logging.warning('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:,:rankY]
        T22 = T22[:rankY,:rankY]

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX,rankY)
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:,:d] * np.sqrt(n-1)
    B = inv(T22) @ M[:,:d] * np.sqrt(n-1)
    r = D[:d]
    # remove roundoff errs
    r[r>=1] = 1
    r[r<=0] = 0

    if not fullReturn:
        return r

    # Put coefficients back to their full size and their correct order
    A[perm1,:] = np.vstack((A, np.zeros((p1-rankX,d))))
    B[perm2,:] = np.vstack((B, np.zeros((p2-rankY,d))))
    
    # Compute the canonical variates
    U = X @ A
    V = Y @ B

    return A, B, r, U, V

def align_subject_manifolds_with_cca(manifold_dict, subject_id1, subject_id2, band_name, downsample_factor=1, plot=True, output_dir=None):
    """
    Align manifold representations of two subjects using CCA.
    
    Parameters:
    -----------
    manifold_dict : dict
        Dictionary containing manifold data for each subject (from compute_neural_manifold)
    subject_id1, subject_id2 : int
        IDs of the subjects to align
    band_name : str
        Name of the frequency band being analyzed
    plot : bool, optional
        Whether to plot the aligned manifolds
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    dict
        Dictionary containing CCA results and aligned manifolds
    """
    # Get the manifold data for both subjects
    X = manifold_dict[subject_id1]['manifold']  # Shape: (n_times, n_components)
    Y = manifold_dict[subject_id2]['manifold']  # Shape: (n_times, n_components)
    
    # Ensure both manifolds have the same number of time points
    min_times = min(X.shape[0], Y.shape[0])
    X = X[:min_times, :]
    Y = Y[:min_times, :]
    
    # Perform CCA
    A, B, r, U, V = canoncorr(X, Y, fullReturn=True)
    
    print(f"Canonical correlations between Subject {subject_id1} and Subject {subject_id2}:")
    for i, corr in enumerate(r):
        print(f"  Component {i+1}: {corr:.4f}")
    
    # Create a result dictionary
    result = {
        'A': A,  # Canonical coefficients for subject 1
        'B': B,  # Canonical coefficients for subject 2
        'r': r,  # Canonical correlations
        'U': U,  # Aligned manifold for subject 1 (canonical variates)
        'V': V,  # Aligned manifold for subject 2 (canonical variates)
        'X': X,  # Original manifold for subject 1
        'Y': Y   # Original manifold for subject 2
    }
    
    # Plot if requested
    if plot and X.shape[1] >= 3 and Y.shape[1] >= 3:
        # Plot original manifolds
        fig = plt.figure(figsize=(16, 12))
        
        # Get time points
        times = np.arange(min_times) / (1000 / downsample_factor)  # Convert to seconds based on sample rate
        
        # Create a colormap for time - using "crest" for original and "flare" for aligned
        norm = plt.Normalize(times.min(), times.max())
        cmap_orig = sns.color_palette("crest", as_cmap=True)
        cmap_aligned = sns.color_palette("flare", as_cmap=True)
        colors_orig = cmap_orig(norm(times))
        colors_aligned = cmap_aligned(norm(times))
        
        # Plot original manifolds
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors_orig, s=15, alpha=0.8)
        ax1.set_title(f"Subject {subject_id1} Original Manifold", fontsize=12)
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_zlabel("PC3")
        
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=colors_orig, s=15, alpha=0.8)
        ax2.set_title(f"Subject {subject_id2} Original Manifold", fontsize=12)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_zlabel("PC3")
        
        # Determine the common axis limits for aligned manifolds
        # Combine U and V data to find overall min and max for each dimension
        combined_aligned = np.vstack([U[:, :3], V[:, :3]])
        min_vals = np.min(combined_aligned, axis=0)
        max_vals = np.max(combined_aligned, axis=0)
        
        # Add a small margin (10%) to the ranges
        ranges = max_vals - min_vals
        min_vals = min_vals - 0.1 * ranges
        max_vals = max_vals + 0.1 * ranges
        
        # Plot aligned manifolds (using first 3 canonical components)
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.scatter(U[:, 0], U[:, 1], U[:, 2], c=colors_aligned, s=15, alpha=0.8)
        ax3.set_title(f"Subject {subject_id1} Aligned Manifold", fontsize=12)
        ax3.set_xlabel(f"CC1 (r={r[0]:.3f})")
        ax3.set_ylabel(f"CC2 (r={r[1]:.3f})")
        ax3.set_zlabel(f"CC3 (r={r[2]:.3f})")
        
        # Set common limits for the aligned manifolds
        ax3.set_xlim(min_vals[0], max_vals[0])
        ax3.set_ylim(min_vals[1], max_vals[1])
        ax3.set_zlim(min_vals[2], max_vals[2])
        
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.scatter(V[:, 0], V[:, 1], V[:, 2], c=colors_aligned, s=15, alpha=0.8)
        ax4.set_title(f"Subject {subject_id2} Aligned Manifold", fontsize=12)
        ax4.set_xlabel(f"CC1 (r={r[0]:.3f})")
        ax4.set_ylabel(f"CC2 (r={r[1]:.3f})")
        ax4.set_zlabel(f"CC3 (r={r[2]:.3f})")
        
        # Set common limits for the aligned manifolds
        ax4.set_xlim(min_vals[0], max_vals[0])
        ax4.set_ylim(min_vals[1], max_vals[1])
        ax4.set_zlim(min_vals[2], max_vals[2])
        
        # Add a colorbar for time - original manifolds
        fig.subplots_adjust(right=0.85)  # Make room for colorbar
        cbar_ax1 = fig.add_axes([0.88, 0.55, 0.02, 0.3])  # Position for top colorbar
        sm1 = plt.cm.ScalarMappable(cmap=cmap_orig, norm=norm)
        sm1.set_array([])
        cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
        cbar1.set_label('Time (s) - Original')
        
        # Add a colorbar for time - aligned manifolds
        cbar_ax2 = fig.add_axes([0.88, 0.15, 0.02, 0.3])  # Position for bottom colorbar
        sm2 = plt.cm.ScalarMappable(cmap=cmap_aligned, norm=norm)
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, cax=cbar_ax2)
        cbar2.set_label('Time (s) - Aligned')
        
        # Adjust the layout
        plt.suptitle(f"Band Neural Manifold Alignment ({band_name})\nSubject {subject_id1} vs Subject {subject_id2}", 
                    fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])  # Adjust for colorbar space
        
        # Save or show the figure
        if output_dir is not None:
            plt.savefig(f"{output_dir}/cca_aligned_{band_name}_sub{subject_id1}_sub{subject_id2}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    return result

