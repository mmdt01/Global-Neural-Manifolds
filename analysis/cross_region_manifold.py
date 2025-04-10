"""
Cross-region neural manifold analysis module for analyzing relationships between brain regions.
"""

import os
import numpy as np
from scipy import stats
from .high_dim_manifold import analyze_high_dim_neural_manifolds
from .manifold import canoncorr

def align_cross_region_manifolds(all_region_epochs, all_region_channels_dict, region_lists, 
                               bands=None, n_components=10, downsample_factor=1, output_dir=None):
    """
    Align neural manifolds across different brain regions and different subjects using CCA.
    Compares all possible subject combinations between regions.
    
    Parameters:
    -----------
    all_region_epochs : dict
        Dictionary mapping region names to subject-specific epochs
    all_region_channels_dict : dict
        Dictionary mapping region names to subject-specific channel information
    region_lists : dict
        Dictionary mapping region group names to lists of region labels
    bands : list, optional
        List of frequency bands to analyze. If None, uses ['delta', 'beta', 'high_gamma']
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    output_dir : str, optional
        Directory to save results. If None, results are not saved.
        
    Returns:
    --------
    dict
        Dictionary containing cross-region manifold results
    """
    
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'beta', 'high_gamma']
    
    # First, compute manifolds for each region separately
    print("\n===== Computing neural manifolds for each region separately =====")
    region_manifold_results = {}
    
    for region_name, region_epochs in all_region_epochs.items():
        region_channels_dict = all_region_channels_dict[region_name]
        region_labels = region_lists[region_name]
        
        # Create region-specific output directory if needed
        if output_dir is not None:
            region_output_dir = os.path.join(output_dir, region_name)
            os.makedirs(region_output_dir, exist_ok=True)
        else:
            region_output_dir = None
        
        print(f"\nComputing manifolds for region: {region_name}")
        # Compute high-dimensional manifolds for this region
        manifold_results = analyze_high_dim_neural_manifolds(
            region_epochs,
            region_channels_dict,
            region_labels,
            bands=bands,
            n_components=n_components,
            downsample_factor=downsample_factor,
            output_dir=region_output_dir
        )
        
        # Store results for this region
        region_manifold_results[region_name] = manifold_results
    
    # Initialize cross-region CCA results dictionary
    cross_region_results = {}
    
    # Now, align manifolds across different regions and different subjects
    print("\n===== Aligning manifolds across regions and subjects =====")
    
    # Get list of regions for comparison
    regions = list(all_region_epochs.keys())
    
    # Compare each pair of regions (including same region for within-region analysis)
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            # Create a key for this region pair
            region_pair_key = f"{region1}_vs_{region2}"
            print(f"\nAnalyzing region pair: {region_pair_key}")
            
            # Get manifolds for both regions
            manifolds1 = region_manifold_results[region1]
            manifolds2 = region_manifold_results[region2]
            
            # Initialize results for this region pair
            cross_region_results[region_pair_key] = {}
            
            # Process each frequency band
            for band_name in bands:
                if band_name not in manifolds1 or band_name not in manifolds2:
                    print(f"  Missing data for band {band_name} in one or both regions, skipping...")
                    continue
                
                print(f"  Processing {band_name} band...")
                
                # Initialize band-specific results
                cross_region_results[region_pair_key][band_name] = {}
                
                # Get available subjects for each region
                subjects1 = sorted(list(manifolds1[band_name].keys()))
                subjects2 = sorted(list(manifolds2[band_name].keys()))
                
                print(f"  Found {len(subjects1)} subjects in {region1}: {subjects1}")
                print(f"  Found {len(subjects2)} subjects in {region2}: {subjects2}")
                
                if not subjects1 or not subjects2:
                    print(f"  No subjects available for one or both regions, skipping...")
                    continue
                
                # Process ALL subject combinations
                for subject_id1 in subjects1:
                    for subject_id2 in subjects2:
                        # For within-region analysis, only compare different subjects
                        # to avoid perfect correlations when comparing a subject with itself
                        if region1 == region2 and subject_id1 == subject_id2:
                            continue
                            
                        # Get subject-region key
                        subject_pair_key = f"{subject_id1}_vs_{subject_id2}"
                        
                        # Get manifold data for both subjects
                        X = manifolds1[band_name][subject_id1]['manifold']
                        Y = manifolds2[band_name][subject_id2]['manifold']
                        
                        # Ensure manifolds have same number of time points
                        min_times = min(X.shape[0], Y.shape[0])
                        X = X[:min_times, :]
                        Y = Y[:min_times, :]
                        
                        # Perform CCA between different subjects and regions
                        try:
                            print(f"    Comparing Subject {subject_id1} ({region1}) with Subject {subject_id2} ({region2})")
                            A, B, r, U, V = canoncorr(X, Y, fullReturn=True)
                            
                            # Print correlations
                            for k, corr in enumerate(r):
                                print(f"      Mode {k+1}: r = {corr:.4f}")
                            
                            # Store results
                            cross_region_results[region_pair_key][band_name][subject_pair_key] = {
                                'A': A,  # Canonical coefficients for subject 1
                                'B': B,  # Canonical coefficients for subject 2
                                'r': r,  # Canonical correlations
                                'U': U,  # Aligned manifold for subject 1
                                'V': V,  # Aligned manifold for subject 2
                                'X': X,  # Original manifold for subject 1
                                'Y': Y   # Original manifold for subject 2
                            }
                        except Exception as e:
                            print(f"    Error computing CCA: {e}")
    
    # Return the cross-region results and the individual region manifold results
    return cross_region_results, region_manifold_results

def compute_region_similarity_matrix(cross_region_results, bands=None, method='mean', 
                                   comparison_type='all'):
    """
    Compute a similarity matrix between brain regions based on CCA results.
    
    Parameters:
    -----------
    cross_region_results : dict
        Dictionary containing cross-region CCA results
    bands : list, optional
        List of frequency bands to include. If None, uses all available bands.
    method : str, optional
        Method to aggregate correlations. Options: 'mean', 'max', 'median'
    comparison_type : str, optional
        Type of comparison to include. Options: 'all', 'within', 'cross'
        
    Returns:
    --------
    dict
        Dictionary mapping bands to region similarity matrices
    """
    
    # If bands not specified, determine from results
    if bands is None:
        # Find all unique bands across all region pairs
        bands = set()
        for region_pair in cross_region_results:
            bands.update(cross_region_results[region_pair].keys())
        bands = sorted(list(bands))
    
    # Get unique regions
    regions = set()
    for region_pair in cross_region_results:
        region1, region2 = region_pair.split('_vs_')
        regions.add(region1)
        regions.add(region2)
    regions = sorted(list(regions))
    n_regions = len(regions)
    
    # Initialize similarity matrices for each band
    similarity_matrices = {}
    
    for band_name in bands:
        # Create matrix for this band
        similarity_matrix = np.zeros((n_regions, n_regions))
        count_matrix = np.zeros((n_regions, n_regions))  # To track number of comparisons
        
        # Process each region pair
        for region_pair, band_data in cross_region_results.items():
            if band_name not in band_data:
                continue
                
            # Get regions from the key
            region1, region2 = region_pair.split('_vs_')
            
            # Get indices in the matrix
            i = regions.index(region1)
            j = regions.index(region2)
            
            # Skip if comparison type doesn't match
            if comparison_type == 'within' and region1 != region2:
                continue
            if comparison_type == 'cross' and region1 == region2:
                continue
            
            # Get all subject pair correlations
            all_correlations = []
            
            for subject_pair, result in band_data[band_name].items():
                # Add correlations for this subject pair
                all_correlations.extend(result['r'])
            
            if not all_correlations:
                continue
            
            # Compute aggregate correlation based on method
            if method == 'mean':
                agg_correlation = np.mean(all_correlations)
            elif method == 'max':
                agg_correlation = np.max(all_correlations)
            elif method == 'median':
                agg_correlation = np.median(all_correlations)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Store in matrix
            similarity_matrix[i, j] = agg_correlation
            count_matrix[i, j] = len(all_correlations)
            
            # For within-region, also store symmetrically (if comparison type allows)
            if region1 == region2 or comparison_type == 'all':
                similarity_matrix[j, i] = agg_correlation
                count_matrix[j, i] = len(all_correlations)
        
        # Set diagonal to 1.0 for comparison_type 'all' or 'within'
        if comparison_type != 'cross':
            np.fill_diagonal(similarity_matrix, 1.0)
        
        # Store matrix for this band
        similarity_matrices[band_name] = {
            'matrix': similarity_matrix,
            'count_matrix': count_matrix,
            'regions': regions
        }
    
    return similarity_matrices

def analyze_mode_specific_correlations(cross_region_results, bands=None, n_modes=None):
    """
    Analyze correlations for each specific neural mode across regions.
    
    Parameters:
    -----------
    cross_region_results : dict
        Dictionary containing cross-region CCA results
    bands : list, optional
        List of frequency bands to include. If None, uses all available bands.
    n_modes : int, optional
        Number of modes to analyze. If None, uses all available modes.
        
    Returns:
    --------
    dict
        Dictionary of mode-specific correlation results
    """
    
    # If bands not specified, determine from results
    if bands is None:
        # Find all unique bands across all region pairs
        bands = set()
        for region_pair in cross_region_results:
            bands.update(cross_region_results[region_pair].keys())
        bands = sorted(list(bands))
    
    # Initialize results
    mode_correlations = {}
    
    # Process each band
    for band_name in bands:
        # Initialize band-specific results
        mode_correlations[band_name] = {
            'within_region': {},
            'cross_region': {},
            'all': {}
        }
        
        # Determine the maximum number of modes available
        max_modes = 0
        for region_pair in cross_region_results:
            if band_name not in cross_region_results[region_pair]:
                continue
                
            for subject_pair, result in cross_region_results[region_pair][band_name].items():
                max_modes = max(max_modes, len(result['r']))
        
        # If n_modes not specified, use all available
        if n_modes is None:
            n_modes = max_modes
        else:
            n_modes = min(n_modes, max_modes)
        
        # Initialize mode-specific results for each comparison type
        for comp_type in ['within_region', 'cross_region', 'all']:
            for mode in range(1, n_modes + 1):
                mode_correlations[band_name][comp_type][mode] = {
                    'by_region_pair': {},
                    'overall_mean': 0.0,
                    'overall_std': 0.0,
                    'overall_correlations': []
                }
        
        # Process each region pair
        for region_pair in cross_region_results:
            if band_name not in cross_region_results[region_pair]:
                continue
            
            # Determine if this is within-region or cross-region
            region1, region2 = region_pair.split('_vs_')
            comp_type = 'within_region' if region1 == region2 else 'cross_region'
            
            # Get subject-specific results for this region pair
            subject_results = cross_region_results[region_pair][band_name]
            
            # Process each mode
            for mode in range(1, n_modes + 1):
                mode_idx = mode - 1  # Convert 1-based to 0-based indexing
                
                # Collect correlations for this mode across subject pairs
                mode_corrs = []
                
                for subject_pair, result in subject_results.items():
                    if mode_idx < len(result['r']):
                        mode_corrs.append(result['r'][mode_idx])
                
                if not mode_corrs:
                    continue
                
                # Compute statistics for this mode and region pair
                mean_corr = np.mean(mode_corrs)
                std_corr = np.std(mode_corrs)
                
                # Store results for both specific type and 'all'
                for current_type in [comp_type, 'all']:
                    # Store region pair results
                    mode_correlations[band_name][current_type][mode]['by_region_pair'][region_pair] = {
                        'mean': mean_corr,
                        'std': std_corr,
                        'correlations': mode_corrs
                    }
                    
                    # Add to overall correlations
                    mode_correlations[band_name][current_type][mode]['overall_correlations'].extend(mode_corrs)
        
        # Compute overall statistics for each mode and comparison type
        for comp_type in ['within_region', 'cross_region', 'all']:
            for mode in range(1, n_modes + 1):
                all_corrs = mode_correlations[band_name][comp_type][mode]['overall_correlations']
                
                if all_corrs:
                    mode_correlations[band_name][comp_type][mode]['overall_mean'] = np.mean(all_corrs)
                    mode_correlations[band_name][comp_type][mode]['overall_std'] = np.std(all_corrs)
                    mode_correlations[band_name][comp_type][mode]['overall_count'] = len(all_corrs)
    
    return mode_correlations

def compare_within_vs_cross_region_correlations(mode_correlations, bands=None, n_modes=None):
    """
    Compare correlation strength between within-region and cross-region alignments.
    
    Parameters:
    -----------
    mode_correlations : dict
        Dictionary containing mode-specific correlation results
    bands : list, optional
        List of frequency bands to include. If None, uses all available bands.
    n_modes : int, optional
        Number of modes to analyze. If None, uses all available modes.
        
    Returns:
    --------
    dict
        Dictionary of comparative statistics between within and cross-region correlations
    """
    
    # If bands not specified, use all available
    if bands is None:
        bands = list(mode_correlations.keys())
    
    # Initialize results
    comparison_results = {}
    
    # Process each band
    for band_name in bands:
        if band_name not in mode_correlations:
            continue
            
        # Initialize band-specific results
        comparison_results[band_name] = {}
        
        # Determine maximum number of modes
        max_mode = 0
        for comp_type in ['within_region', 'cross_region']:
            if comp_type in mode_correlations[band_name]:
                max_mode = max(max_mode, max(mode_correlations[band_name][comp_type].keys()))
        
        # If n_modes specified, limit analysis
        if n_modes is not None:
            max_mode = min(max_mode, n_modes)
        
        # Process each mode
        for mode in range(1, max_mode + 1):
            # Check if this mode exists for both comparison types
            if (mode not in mode_correlations[band_name]['within_region'] or
                mode not in mode_correlations[band_name]['cross_region']):
                continue
            
            # Get correlations for each type
            within_corrs = mode_correlations[band_name]['within_region'][mode]['overall_correlations']
            cross_corrs = mode_correlations[band_name]['cross_region'][mode]['overall_correlations']
            
            # Skip if no correlations
            if not within_corrs or not cross_corrs:
                continue
            
            # Compute basic statistics
            within_mean = np.mean(within_corrs)
            within_std = np.std(within_corrs)
            cross_mean = np.mean(cross_corrs)
            cross_std = np.std(cross_corrs)
            
            # Compute difference
            mean_diff = within_mean - cross_mean
            
            # Perform statistical test (t-test)
            t_stat, p_value = stats.ttest_ind(within_corrs, cross_corrs, equal_var=False)
            
            # Store results
            comparison_results[band_name][mode] = {
                'within': {
                    'mean': within_mean,
                    'std': within_std,
                    'count': len(within_corrs)
                },
                'cross': {
                    'mean': cross_mean,
                    'std': cross_std,
                    'count': len(cross_corrs)
                },
                'difference': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return comparison_results