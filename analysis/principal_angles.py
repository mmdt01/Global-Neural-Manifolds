"""
Principal angles computation for gesture manifold comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from itertools import combinations
import os

def compute_principal_angles(manifold_A, manifold_B):
    """
    Compute principal angles between two manifolds using SVD.
    
    Parameters:
    -----------
    manifold_A : array, shape (n_channels, n_components)
        Spatial patterns (basis vectors) for first manifold
    manifold_B : array, shape (n_channels, n_components) 
        Spatial patterns (basis vectors) for second manifold
        
    Returns:
    --------
    angles : array, shape (n_components,)
        Principal angles in radians
    singular_values : array, shape (n_components,)
        Singular values (cosines of principal angles)
    """
    # Ensure manifolds have same number of components
    n_comp = min(manifold_A.shape[1], manifold_B.shape[1])
    A = manifold_A[:, :n_comp]
    B = manifold_B[:, :n_comp]
    
    # Compute dot product matrix between the two manifold bases
    # This captures how much each basis vector in A aligns with each in B
    dot_product_matrix = A.T @ B  # Shape: (n_comp, n_comp)
    
    # Perform SVD on the dot product matrix
    U, s, Vt = svd(dot_product_matrix, full_matrices=False)
    
    # Singular values are cosines of principal angles
    # Clamp to [0, 1] to handle numerical precision issues
    s = np.clip(s, 0, 1)
    
    # Convert to angles (in radians)
    angles = np.arccos(s)
    
    return angles, s

def compute_pairwise_angles_subject(manifold_data, n_components=20):
    """
    Compute principal angles between all pairs of gestures for a single subject.
    Returns both mean angles and component-wise angles.
    """
    gestures = list(manifold_data.keys())
    gesture_pairs = [f"{g1}_vs_{g2}" for g1, g2 in combinations(gestures, 2)]
    
    pairwise_angles = {}
    pairwise_angles_componentwise = {}  # NEW: Store all component angles
    
    for g1, g2 in combinations(gestures, 2):
        pair_key = f"{g1}_vs_{g2}"
        
        # Get spatial patterns (PCA components) for both gestures
        patterns_A = manifold_data[g1]['spatial_patterns'][:, :n_components]
        patterns_B = manifold_data[g2]['spatial_patterns'][:, :n_components]
        
        # Compute principal angles
        angles, _ = compute_principal_angles(patterns_A, patterns_B)
        
        pairwise_angles[pair_key] = angles  # Store ALL component angles
        pairwise_angles_componentwise[pair_key] = angles  # For clarity
        
        print(f"  {pair_key}: Mean angle = {np.mean(angles):.3f} rad ({np.degrees(np.mean(angles)):.1f}°)")
    
    return pairwise_angles, gesture_pairs

def aggregate_across_subjects(all_subject_angles, gesture_pairs):
    """
    Aggregate principal angles across all subjects.
    """
    from scipy.stats import sem
    
    # Initialize storage for aggregated results
    aggregated = {
        'mean_angles': {},
        'sem_angles': {},
        'all_angles': {},
        'mean_angle_degrees': {},
        'summary_stats': {}
    }
    
    # Aggregate for each gesture pair
    for pair in gesture_pairs:
        # Collect angles from all subjects for this pair
        pair_angles_all_subjects = []
        
        for subject_id, subject_angles in all_subject_angles.items():
            if pair in subject_angles:
                pair_angles_all_subjects.append(subject_angles[pair])
        
        if len(pair_angles_all_subjects) == 0:
            continue
        
        # Convert to array: (n_subjects, n_components)
        pair_angles_array = np.array(pair_angles_all_subjects)
        
        # Compute statistics
        mean_angles = np.mean(pair_angles_array, axis=0)
        sem_angles = sem(pair_angles_array, axis=0)
        
        aggregated['mean_angles'][pair] = mean_angles
        aggregated['sem_angles'][pair] = sem_angles
        aggregated['all_angles'][pair] = pair_angles_array
        aggregated['mean_angle_degrees'][pair] = np.degrees(mean_angles)
        
        # Summary statistics
        overall_mean = np.mean(mean_angles)
        overall_std = np.std(mean_angles)
        
        aggregated['summary_stats'][pair] = {
            'mean_angle_rad': overall_mean,
            'mean_angle_deg': np.degrees(overall_mean),
            'std_angle_rad': overall_std,
            'std_angle_deg': np.degrees(overall_std),
            'n_subjects': len(pair_angles_all_subjects)
        }
        
        print(f"{pair}: {np.degrees(overall_mean):.1f}° ± {np.degrees(overall_std):.1f}° (n={len(pair_angles_all_subjects)})")
    
    return aggregated

def create_similarity_visualizations(aggregated_results, band_name, n_components, output_dir=None):
    """
    Create visualizations of manifold similarity results.
    """
    # 1. Heatmap of mean angles between all gesture pairs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract gesture names and create symmetric matrix
    pairs = list(aggregated_results['summary_stats'].keys())
    gestures = set()
    for pair in pairs:
        g1, g2 = pair.split('_vs_')
        gestures.add(g1)
        gestures.add(g2)
    gestures = sorted(list(gestures))
    
    # Create symmetric matrix of mean angles
    n_gestures = len(gestures)
    angle_matrix = np.zeros((n_gestures, n_gestures))
    
    for i, g1 in enumerate(gestures):
        for j, g2 in enumerate(gestures):
            if i == j:
                angle_matrix[i, j] = 0  # Self-comparison
            elif i < j:
                pair_key = f"{g1}_vs_{g2}"
                if pair_key in aggregated_results['summary_stats']:
                    angle_deg = aggregated_results['summary_stats'][pair_key]['mean_angle_deg']
                    angle_matrix[i, j] = angle_deg
                    angle_matrix[j, i] = angle_deg  # Make symmetric
    
    # Plot heatmap
    sns.heatmap(angle_matrix, 
                xticklabels=[g.capitalize() for g in gestures],
                yticklabels=[g.capitalize() for g in gestures],
                annot=True, fmt='.1f', cmap='viridis_r',
                cbar_kws={'label': 'Mean Principal Angle (degrees)'},
                ax=ax1)
    ax1.set_title(f'{band_name.upper()} Band: Mean Principal Angles\nBetween Gesture Manifolds')
    
    # 2. Distribution of angles across all pairs
    all_mean_angles = []
    pair_labels = []
    
    for pair, stats in aggregated_results['summary_stats'].items():
        all_mean_angles.append(stats['mean_angle_deg'])
        pair_labels.append(pair.replace('_vs_', ' vs ').replace('_', ' ').title())
    
    bars = ax2.bar(range(len(all_mean_angles)), all_mean_angles, 
                   color='skyblue', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, angle in zip(bars, all_mean_angles):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{angle:.1f}°', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Gesture Pairs')
    ax2.set_ylabel('Mean Principal Angle (degrees)')
    ax2.set_title(f'{band_name.upper()} Band: Principal Angles\nAcross All Gesture Pairs')
    ax2.set_xticks(range(len(pair_labels)))
    ax2.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir is not None:
        plt.savefig(f"{output_dir}/manifold_similarity_{band_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 3. Component-wise angle analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(pairs)))
    
    for i, pair in enumerate(pairs):
        if pair in aggregated_results['mean_angles']:
            mean_angles = aggregated_results['mean_angles'][pair]
            sem_angles = aggregated_results['sem_angles'][pair]
            
            component_nums = np.arange(1, len(mean_angles) + 1)
            
            ax.errorbar(component_nums, np.degrees(mean_angles), 
                       yerr=np.degrees(sem_angles),
                       label=pair.replace('_vs_', ' vs ').replace('_', ' ').title(),
                       color=colors[i], marker='o', capsize=3)
    
    ax.set_xlabel('Neural Mode (Manifold Dimension)')
    ax.set_ylabel('Principal Angle (degrees)')
    ax.set_title(f'{band_name.upper()} Band: Principal Angles by Component')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, n_components + 0.5)
    
    plt.tight_layout()
    
    if output_dir is not None:
        plt.savefig(f"{output_dir}/component_wise_angles_{band_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Print summary
    print(f"\n===== MANIFOLD SIMILARITY SUMMARY ({band_name.upper()} BAND) =====")
    print(f"Overall Statistics:")
    all_angles = [stats['mean_angle_deg'] for stats in aggregated_results['summary_stats'].values()]
    print(f"  Mean angle across all pairs: {np.mean(all_angles):.1f}° ± {np.std(all_angles):.1f}°")
    print(f"  Range: {np.min(all_angles):.1f}° to {np.max(all_angles):.1f}°")
    print(f"  Number of gesture pairs: {len(all_angles)}")

def analyze_gesture_manifold_similarity(manifold_results, band_name, n_components=20, output_dir=None):
    """
    Analyze manifold similarity across all subjects for a specific frequency band.
    
    Parameters:
    -----------
    manifold_results : dict
        Results from gesture comparison analysis: {gesture_name: {subject_id: manifold_data}}
    band_name : str
        Name of the frequency band being analyzed
    n_components : int
        Number of components to use for angle computation
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    print(f"\n===== Analyzing Gesture Manifold Similarity ({band_name} band) =====")
    
    # Get all gestures and subjects
    gestures = list(manifold_results.keys())
    all_subjects = set()
    for gesture_data in manifold_results.values():
        all_subjects.update(gesture_data.keys())
    all_subjects = sorted(list(all_subjects))
    
    print(f"Gestures: {gestures}")
    print(f"Subjects: {all_subjects}")
    print(f"Using {n_components} components for analysis")
    
    # Initialize results storage
    all_subject_angles = {}
    
    # Analyze each subject
    for subject_id in all_subjects:
        print(f"\nSubject {subject_id}:")
        
        # Check if this subject has data for all gestures
        subject_gestures = {}
        for gesture in gestures:
            if subject_id in manifold_results[gesture]:
                subject_gestures[gesture] = manifold_results[gesture][subject_id]
        
        if len(subject_gestures) < 2:
            print(f"  Insufficient gestures ({len(subject_gestures)}) for comparison, skipping...")
            continue
        
        # Compute pairwise angles for this subject
        pairwise_angles, gesture_pairs = compute_pairwise_angles_subject(
            subject_gestures, n_components
        )
        
        all_subject_angles[subject_id] = pairwise_angles
    
    # Aggregate results across subjects
    print(f"\n===== Aggregating Results Across {len(all_subject_angles)} Subjects =====")
    
    aggregated_results = aggregate_across_subjects(all_subject_angles, gesture_pairs)
    
    # Create visualizations
    if output_dir is not None:
        similarity_dir = os.path.join(output_dir, "manifold_similarity")
        os.makedirs(similarity_dir, exist_ok=True)
    else:
        similarity_dir = None
    
    create_similarity_visualizations(
        aggregated_results, 
        band_name, 
        n_components,
        similarity_dir
    )
    
    # Compile final results
    results = {
        'band_name': band_name,
        'n_components': n_components,
        'gestures': gestures,
        'subjects': all_subjects,
        'subject_angles': all_subject_angles,
        'aggregated': aggregated_results
    }
    
    return results

# Main function to add to your analysis pipeline
def run_manifold_similarity_analysis(manifold_results, frequency_bands, n_components=20, output_dir=None):
    """
    Run manifold similarity analysis for all frequency bands.
    
    Parameters:
    -----------
    manifold_results : dict
        Results from gesture comparison analysis
    frequency_bands : list
        List of frequency bands to analyze
    n_components : int
        Number of components to use
    output_dir : str, optional
        Output directory
        
    Returns:
    --------
    all_results : dict
        Results for all bands
    """
    all_results = {}
    
    for band_name in frequency_bands:
        if band_name in manifold_results:
            print(f"\n{'='*60}")
            print(f"ANALYZING BAND: {band_name.upper()}")
            print(f"{'='*60}")
            
            results = analyze_gesture_manifold_similarity(
                manifold_results[band_name],
                band_name,
                n_components=n_components,
                output_dir=output_dir
            )
            
            all_results[band_name] = results
    
    return all_results


