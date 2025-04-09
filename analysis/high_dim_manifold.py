"""
High-dimensional neural manifold analysis module
"""

def analyze_high_dim_neural_manifolds(region_epochs, region_channels_dict, region_labels, 
                                bands=None, n_components=10, downsample_factor=1, output_dir=None):
    """
    Analyze neural manifolds with higher dimensionality (more than 3 components) without plotting.
    This is a modified version of analyze_neural_manifolds that skips the 3D visualization steps.
    
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
        Number of PCA components to compute (default=10)
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    manifold_results : dict
        Dictionary mapping band names to manifold dictionaries
    """
    from .manifold import compute_band_power, get_frequency_bands
    from sklearn.decomposition import PCA
    import numpy as np
    
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'beta', 'high_gamma']
    
    # Initialize results dictionary
    manifold_results = {}
    
    # Get frequency bands dictionary
    freq_bands = get_frequency_bands()
    
    # Analyze each frequency band
    for band_name in bands:
        print(f"\n\n===== Analyzing {band_name.upper()} band neural manifolds with {n_components} components =====")
        
        # Check if the requested band exists
        if band_name not in freq_bands:
            print(f"Unknown band name: {band_name}. Available bands: {list(freq_bands.keys())}")
            continue
        
        # Initialize band-specific dictionary
        manifold_results[band_name] = {}
        
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
                manifold_results[band_name][subject_id] = {
                    'manifold': components,
                    'explained_variance': pca.explained_variance_ratio_,
                    'pca': pca
                }
                
                # Print explained variance
                print(f"Explained variance: {pca.explained_variance_ratio_}")
                print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
                
                print(f"Neural manifold computation completed for Subject {subject_id}")
                
            except Exception as e:
                print(f"Error computing neural manifold for Subject {subject_id}: {e}")
    
    # Return the manifold dictionary
    return manifold_results

def align_high_dim_manifolds(manifold_results, subject_ids, bands=None, output_dir=None):
    """
    Compare high-dimensional manifold representations between pairs of subjects using CCA.
    This is a modified version of compare_subject_manifolds that handles higher dimensions.
    
    Parameters:
    -----------
    manifold_results : dict
        Dictionary containing manifold data for each band and subject
    subject_ids : list
        List of subject IDs to compare
    bands : list, optional
        List of frequency bands to analyze. If None, uses all available bands.
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    dict
        Dictionary containing CCA results for each pair of subjects and each band
    """
    from .manifold import canoncorr
    
    # If bands not specified, use all available bands
    if bands is None:
        bands = list(manifold_results.keys())
    
    # Initialize results dictionary
    cca_results = {}
    
    # Compare each pair of subjects for each band
    for band_name in bands:
        print(f"\n===== Aligning {band_name.upper()} band manifolds between subjects =====")
        
        # Skip if this band is not in the results
        if band_name not in manifold_results:
            print(f"No results found for {band_name} band, skipping...")
            continue
        
        # Initialize band-specific results
        cca_results[band_name] = {}
        
        # Compare each pair of subjects
        for i, subject_id1 in enumerate(subject_ids):
            for subject_id2 in subject_ids[i+1:]:
                # Skip if either subject is not in the results
                if subject_id1 not in manifold_results[band_name] or subject_id2 not in manifold_results[band_name]:
                    print(f"Missing data for Subject {subject_id1} or {subject_id2}, skipping...")
                    continue
                
                print(f"\nComparing Subject {subject_id1} vs Subject {subject_id2}...")
                
                # Get manifold data for both subjects
                X = manifold_results[band_name][subject_id1]['manifold']
                Y = manifold_results[band_name][subject_id2]['manifold']
                
                # Ensure manifolds have same number of time points
                min_times = min(X.shape[0], Y.shape[0])
                X = X[:min_times, :]
                Y = Y[:min_times, :]
                
                # Perform CCA
                try:
                    A, B, r, U, V = canoncorr(X, Y, fullReturn=True)
                    
                    # Print correlations
                    for i, corr in enumerate(r):
                        print(f"  Mode {i+1}: r = {corr:.4f}")
                    
                    # Store results
                    pair_key = f"{subject_id1}_vs_{subject_id2}"
                    cca_results[band_name][pair_key] = {
                        'A': A,      # Canonical coefficients for subject 1
                        'B': B,      # Canonical coefficients for subject 2
                        'r': r,      # Canonical correlations
                        'U': U,      # Aligned manifold for subject 1
                        'V': V       # Aligned manifold for subject 2
                    }
                except Exception as e:
                    print(f"  Error computing CCA: {e}")
    
    return cca_results