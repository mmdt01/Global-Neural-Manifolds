"""
Neural manifold visualization functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CCA analysis results plots

def plot_correlation_heatmap(cca_results, band_name, output_dir=None):
    """
    Create a heatmap of canonical correlations between all subjects.
    """
    # Get list of unique subjects
    subjects = set()
    for pair_key in cca_results[band_name].keys():
        s1, s2 = pair_key.split('_vs_')
        subjects.add(s1)
        subjects.add(s2)
    subjects = sorted(list(subjects))
    n_subjects = len(subjects)
    
    # Create matrices to store the correlations for each component
    corr_matrices = [np.zeros((n_subjects, n_subjects)) for _ in range(3)]
    
    # Fill the matrices with correlations
    for pair_key, result in cca_results[band_name].items():
        s1, s2 = pair_key.split('_vs_')
        i = subjects.index(s1)
        j = subjects.index(s2)
        
        # Store correlations for each component
        for k in range(3):
            corr_matrices[k][i, j] = result['r'][k]
            corr_matrices[k][j, i] = result['r'][k]  # Mirror across diagonal
    
    # Set diagonal to 1 (correlation with self)
    for k in range(3):
        np.fill_diagonal(corr_matrices[k], 1)
    
    # Create figure with subplots for each component
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    component_names = ['CC1', 'CC2', 'CC3']
    
    # Plot each component as a separate heatmap
    for k, (ax, matrix, name) in enumerate(zip(axes, corr_matrices, component_names)):
        sns.heatmap(matrix, ax=ax, annot=True, cmap='YlOrRd', vmin=0, vmax=1,
                   xticklabels=subjects, yticklabels=subjects)
        ax.set_title(f'{name} Correlations - {band_name} Band')
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        plt.savefig(f"{output_dir}/correlation_heatmap_{band_name}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_correlation_radar(cca_results, band_name, output_dir=None):
    """
    Create a radar/spider plot of canonical correlations.
    """
    # Extract subject pairs and their correlations
    subject_pairs = []
    correlations = []
    
    for pair_key, result in cca_results[band_name].items():
        subject_pairs.append(pair_key)
        correlations.append(result['r'][:3])  # Get first 3 components
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables (subject pairs)
    N = len(subject_pairs)
    
    # Angles for each variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Component colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot each component
    for i, comp_name in enumerate(['CC1', 'CC2', 'CC3']):
        values = [corr[i] for corr in correlations]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=comp_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Fix axis to go from 0 to 1
    ax.set_ylim(0, 1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subject_pairs, size=8)
    
    # Add legend and title
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Canonical Correlations for {band_name} Band', size=15, y=1.1)
    
    # Save or show
    if output_dir:
        plt.savefig(f"{output_dir}/correlation_radar_{band_name}.png", dpi=300)
        plt.close()
    else:
        plt.show()
