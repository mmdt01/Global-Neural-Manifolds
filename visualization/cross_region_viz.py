"""
Visualization functions for cross-region neural manifold analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_region_similarity_matrices(similarity_matrices, bands=None, output_dir=None):
    """
    Visualize similarity matrices between brain regions.
    
    Parameters:
    -----------
    similarity_matrices : dict
        Dictionary mapping bands to region similarity matrices
    bands : list, optional
        List of frequency bands to visualize. If None, visualizes all available bands.
    output_dir : str, optional
        Directory to save visualizations. If None, plots are displayed but not saved.
    """
    # If bands not specified, use all available
    if bands is None:
        bands = list(similarity_matrices.keys())
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each band
    for band_name in bands:
        if band_name not in similarity_matrices:
            print(f"No similarity matrix available for {band_name} band, skipping...")
            continue
        
        # Get similarity matrix and region labels
        matrix_data = similarity_matrices[band_name]
        similarity_matrix = matrix_data['matrix']
        regions = matrix_data['regions']
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=regions,
            yticklabels=regions,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Mean Correlation'}
        )
        
        plt.title(f'Region Similarity Matrix - {band_name.capitalize()} Band', fontsize=14)
        plt.tight_layout()
        
        # Save or show the figure
        if output_dir is not None:
            plt.savefig(f"{output_dir}/region_similarity_matrix_{band_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # Create a combined figure showing all bands
    if len(bands) > 1:
        # Determine grid layout
        n_bands = len(bands)
        ncols = min(3, n_bands)
        nrows = int(np.ceil(n_bands / ncols))
        
        # Create figure
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
        if nrows * ncols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot each band
        for i, band_name in enumerate(bands):
            if i >= len(axes) or band_name not in similarity_matrices:
                continue
                
            # Get matrix data
            matrix_data = similarity_matrices[band_name]
            similarity_matrix = matrix_data['matrix']
            regions = matrix_data['regions']
            
            # Create heatmap
            sns.heatmap(
                similarity_matrix,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                xticklabels=regions,
                yticklabels=regions,
                vmin=0,
                vmax=1,
                ax=axes[i],
                cbar_kws={'label': 'Correlation'}
            )
            
            axes[i].set_title(f'{band_name.capitalize()} Band')
        
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Region Similarity Matrices Across Frequency Bands', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save or show the figure
        if output_dir is not None:
            plt.savefig(f"{output_dir}/region_similarity_matrices_all_bands.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def visualize_mode_correlations_by_region(mode_correlations, bands=None, comparison_type='all', output_dir=None):
    """
    Visualize correlations for each neural mode by region pair.
    
    Parameters:
    -----------
    mode_correlations : dict
        Dictionary containing mode-specific correlation results
    bands : list, optional
        List of frequency bands to visualize. If None, visualizes all available bands.
    comparison_type : str, optional
        Type of comparison to visualize. Options: 'all', 'within_region', 'cross_region'
    output_dir : str, optional
        Directory to save visualizations. If None, plots are displayed but not saved.
    """
    # If bands not specified, use all available
    if bands is None:
        bands = list(mode_correlations.keys())
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each band
    for band_name in bands:
        if band_name not in mode_correlations:
            print(f"No mode correlation data available for {band_name} band, skipping...")
            continue
        
        # Get mode data for this band and comparison type
        if comparison_type not in mode_correlations[band_name]:
            print(f"No {comparison_type} data for {band_name} band, skipping...")
            continue
            
        band_data = mode_correlations[band_name][comparison_type]
        
        # Collect data for bar chart
        all_data = []
        
        for mode, mode_data in band_data.items():
            for region_pair, pair_data in mode_data['by_region_pair'].items():
                all_data.append({
                    'Mode': f'Mode {mode}',
                    'Mode_num': mode,
                    'Region Pair': region_pair,
                    'Mean Correlation': pair_data['mean'],
                    'Std': pair_data['std']
                })
        
        if not all_data:
            print(f"No data to plot for {band_name} band, {comparison_type}, skipping...")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Sort by mode number
        df = df.sort_values('Mode_num')
        
        # Get unique modes
        modes = df['Mode'].unique()
        
        # 1. Create a bar chart showing mean correlation by mode for each region pair
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        barplot = sns.barplot(
            data=df,
            x='Region Pair',
            y='Mean Correlation',
            hue='Mode',
            palette='viridis',
            errorbar=('ci', 95),
            capsize=0.1
        )
        
        # Customize the plot
        plt.title(f'{band_name.capitalize()} Band: Mean Correlation by Neural Mode and Region Pair ({comparison_type.replace("_", " ").title()})',
                 fontsize=14)
        plt.xlabel('Region Pair', fontsize=12)
        plt.ylabel('Mean Canonical Correlation', fontsize=12)
        plt.ylim([0, 1])
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(title='Neural Mode')
        
        # Rotate x-axis labels if many region pairs
        if len(df['Region Pair'].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save or show the figure
        if output_dir is not None:
            plt.savefig(f"{output_dir}/mode_correlations_by_region_{band_name}_{comparison_type}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # 2. Create a heatmap showing correlation by mode and region pair
        # Pivot the DataFrame for heatmap
        pivot_df = df.pivot(index='Region Pair', columns='Mode', values='Mean Correlation')
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Mean Correlation'}
        )
        
        plt.title(f'{band_name.capitalize()} Band: Correlation Heatmap by Mode and Region Pair ({comparison_type.replace("_", " ").title()})',
                 fontsize=14)
        plt.tight_layout()
        
        # Save or show the figure
        if output_dir is not None:
            plt.savefig(f"{output_dir}/mode_correlation_heatmap_{band_name}_{comparison_type}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def visualize_overall_mode_correlations(mode_correlations, bands=None, comparison_type='all', output_dir=None):
    """
    Visualize overall correlations for each neural mode across all region pairs.
    
    Parameters:
    -----------
    mode_correlations : dict
        Dictionary containing mode-specific correlation results
    bands : list, optional
        List of frequency bands to visualize. If None, visualizes all available bands.
    comparison_type : str, optional
        Type of comparison to visualize. Options: 'all', 'within_region', 'cross_region'
    output_dir : str, optional
        Directory to save visualizations. If None, plots are displayed but not saved.
    """
    # If bands not specified, use all available
    if bands is None:
        bands = list(mode_correlations.keys())
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Collect data for all bands
    all_data = []
    
    for band_name in bands:
        if band_name not in mode_correlations:
            continue
            
        if comparison_type not in mode_correlations[band_name]:
            continue
            
        band_data = mode_correlations[band_name][comparison_type]
        
        for mode, mode_data in band_data.items():
            # Skip if no overall statistics
            if mode_data['overall_mean'] == 0 and mode_data['overall_std'] == 0:
                continue
                
            all_data.append({
                'Band': band_name.capitalize(),
                'Mode': f'Mode {mode}',
                'Mode_num': mode,
                'Mean Correlation': mode_data['overall_mean'],
                'Std': mode_data['overall_std']
            })
    
    if not all_data:
        print(f"No overall mode correlation data available for {comparison_type}, skipping visualization...")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by band and mode
    df = df.sort_values(['Band', 'Mode_num'])
    
    # 1. Create line plot showing mean correlation by mode for each band
    plt.figure(figsize=(10, 6))
    
    # Plot mean correlation by mode for each band
    for band_name in df['Band'].unique():
        band_df = df[df['Band'] == band_name]
        
        plt.errorbar(
            band_df['Mode_num'],
            band_df['Mean Correlation'],
            yerr=band_df['Std'],
            marker='o',
            markersize=8,
            linewidth=2,
            capsize=5,
            label=band_name
        )
    
    plt.title(f'Overall Mean Correlation by Neural Mode ({comparison_type.replace("_", " ").title()})', fontsize=14)
    plt.xlabel('Neural Mode', fontsize=12)
    plt.ylabel('Mean Canonical Correlation', fontsize=12)
    plt.xticks(df['Mode_num'].unique())
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend(title='Frequency Band')
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/overall_mode_correlations_{comparison_type}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 2. Create bar chart showing mean correlation by mode and band
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar chart
    sns.barplot(
        data=df,
        x='Mode',
        y='Mean Correlation',
        hue='Band',
        palette='viridis',
        errorbar=('ci', 95),
        capsize=0.1
    )
    
    plt.title(f'Overall Mean Correlation by Neural Mode and Frequency Band ({comparison_type.replace("_", " ").title()})', fontsize=14)
    plt.xlabel('Neural Mode', fontsize=12)
    plt.ylabel('Mean Canonical Correlation', fontsize=12)
    plt.ylim([0, 1])
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title='Frequency Band')
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/overall_mode_correlations_bar_{comparison_type}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_within_vs_cross_comparison(comparison_results, bands=None, output_dir=None):
    """
    Visualize comparison between within-region and cross-region correlations.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary containing comparative statistics between within and cross-region correlations
    bands : list, optional
        List of frequency bands to visualize. If None, visualizes all available bands.
    output_dir : str, optional
        Directory to save visualizations. If None, plots are displayed but not saved.
    """
    # If bands not specified, use all available
    if bands is None:
        bands = list(comparison_results.keys())
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Collect data for all visualizations
    all_data = []
    sig_data = []  # For statistically significant differences
    
    for band_name in bands:
        if band_name not in comparison_results:
            continue
            
        band_data = comparison_results[band_name]
        
        for mode, mode_data in band_data.items():
            all_data.append({
                'Band': band_name.capitalize(),
                'Mode': f'Mode {mode}',
                'Mode_num': mode,
                'Type': 'Within-Region',
                'Mean Correlation': mode_data['within']['mean'],
                'Std': mode_data['within']['std'],
                'Sample Size': mode_data['within']['count']
            })
            
            all_data.append({
                'Band': band_name.capitalize(),
                'Mode': f'Mode {mode}',
                'Mode_num': mode,
                'Type': 'Cross-Region',
                'Mean Correlation': mode_data['cross']['mean'],
                'Std': mode_data['cross']['std'],
                'Sample Size': mode_data['cross']['count']
            })
            
            # Add difference to significant data if p < 0.05
            if mode_data['significant']:
                sig_data.append({
                    'Band': band_name.capitalize(),
                    'Mode': f'Mode {mode}',
                    'Mode_num': mode,
                    'Difference': mode_data['difference'],
                    'p-value': mode_data['p_value'],
                    't-statistic': mode_data['t_statistic']
                })
    
    if not all_data:
        print("No comparison data available, skipping visualization...")
        return
    
    # Convert to DataFrames
    df = pd.DataFrame(all_data)
    sig_df = pd.DataFrame(sig_data) if sig_data else None
    
    # Sort by band and mode
    df = df.sort_values(['Band', 'Mode_num', 'Type'])
    
    # 1. Create grouped bar chart comparing within vs. cross correlations
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar chart
    sns.barplot(
        data=df,
        x='Mode',
        y='Mean Correlation',
        hue='Type',
        palette=['#1f77b4', '#ff7f0e'],  # Blue for within, orange for cross
        errorbar=('ci', 95),
        capsize=0.1
    )
    
    plt.title('Within-Region vs. Cross-Region Mean Correlations by Neural Mode', fontsize=14)
    plt.xlabel('Neural Mode', fontsize=12)
    plt.ylabel('Mean Canonical Correlation', fontsize=12)
    plt.ylim([0, 1])
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title='Comparison Type')
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/within_vs_cross_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 2. Create faceted bar chart by band
    plt.figure(figsize=(14, 8))
    
    # Create facet grid
    g = sns.catplot(
        data=df,
        x='Mode',
        y='Mean Correlation',
        hue='Type',
        col='Band',
        kind='bar',
        palette=['#1f77b4', '#ff7f0e'],
        errorbar=('ci', 95),
        capsize=0.1,
        legend=True,
        height=4,
        aspect=1.2,
        sharey=True
    )
    
    g.set_titles("{col_name} Band")
    g.set_axis_labels("Neural Mode", "Mean Canonical Correlation")
    g.set_ylabels("Mean Canonical Correlation")
    g.set(ylim=(0, 1))
    
    # Add grid lines
    for ax in g.axes.flat:
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/within_vs_cross_by_band.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 3. Plot statistically significant differences if any
    if sig_df is not None and len(sig_df) > 0:
        plt.figure(figsize=(10, 6))
        
        # Sort by band and mode
        sig_df = sig_df.sort_values(['Band', 'Mode_num'])
        
        # Create bar chart of differences
        bars = sns.barplot(
            data=sig_df,
            x='Mode',
            y='Difference',
            hue='Band',
            palette='viridis',
            errorbar=None
        )
        
        # Add stars for significance
        for i, p in enumerate(sig_df['p-value']):
            stars = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
            bars.text(i, sig_df['Difference'].iloc[i] + 0.02, stars, 
                     ha='center', fontweight='bold')
        
        plt.title('Significant Differences Between Within-Region and Cross-Region Correlations', fontsize=14)
        plt.xlabel('Neural Mode', fontsize=12)
        plt.ylabel('Difference (Within - Cross)', fontsize=12)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(title='Frequency Band')
        
        plt.tight_layout()
        
        # Save or show the figure
        if output_dir is not None:
            plt.savefig(f"{output_dir}/significant_differences.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def visualize_cross_region_correlations(cross_region_results, similarity_matrices, 
                                      mode_correlations, comparison_results,
                                      bands=None, output_dir=None):
    """
    Generate comprehensive visualizations for cross-region correlation analysis.
    
    Parameters:
    -----------
    cross_region_results : dict
        Dictionary containing cross-region CCA results
    similarity_matrices : dict
        Dictionary mapping bands to region similarity matrices
    mode_correlations : dict
        Dictionary containing mode-specific correlation results
    comparison_results : dict
        Dictionary containing comparative statistics between within and cross-region correlations
    bands : list, optional
        List of frequency bands to visualize. If None, visualizes all available bands.
    output_dir : str, optional
        Directory to save visualizations. If None, plots are displayed but not saved.
    """
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create summarized report visualizations
    
    # 1. Visualize region similarity matrices
    visualize_region_similarity_matrices(
        similarity_matrices,
        bands=bands,
        output_dir=output_dir
    )
    
    # # 2. Visualize mode correlations by region for each comparison type
    # for comp_type in ['all', 'within_region', 'cross_region']:
    #     visualize_mode_correlations_by_region(
    #         mode_correlations,
    #         bands=bands,
    #         comparison_type=comp_type,
    #         output_dir=output_dir
    #     )
    
    # 3. Visualize overall mode correlations for each comparison type
    for comp_type in ['all', 'within_region', 'cross_region']:
        visualize_overall_mode_correlations(
            mode_correlations,
            bands=bands,
            comparison_type=comp_type,
            output_dir=output_dir
        )
    
    # 4. Visualize comparison between within-region and cross-region correlations
    visualize_within_vs_cross_comparison(
        comparison_results,
        bands=bands,
        output_dir=output_dir
    )
    
    # 5. Generate region-pair specific visualizations
    for region_pair in cross_region_results:
        # Create separate directory for this region pair
        if output_dir is not None:
            pair_dir = os.path.join(output_dir, region_pair)
            os.makedirs(pair_dir, exist_ok=True)
        else:
            pair_dir = None
        
        # Process each band
        for band_name in cross_region_results[region_pair]:
            if bands is not None and band_name not in bands:
                continue
                
            # Collect all correlation data for this region pair and band
            all_data = []
            
            for subject_pair, result in cross_region_results[region_pair][band_name].items():
                for i, corr in enumerate(result['r']):
                    all_data.append({
                        'Subject Pair': subject_pair,
                        'Mode': f'Mode {i+1}',
                        'Mode_num': i+1,
                        'Correlation': corr
                    })
            
            if not all_data:
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Create heatmap of correlations by subject pair and mode
            if len(df) > 0:
                plt.figure(figsize=(10, 8))
                
                # Pivot data for heatmap
                pivot_df = df.pivot(index='Subject Pair', columns='Mode', values='Correlation')
                
                # Create heatmap
                sns.heatmap(
                    pivot_df,
                    annot=True,
                    fmt=".2f",
                    cmap="YlGnBu",
                    vmin=0,
                    vmax=1,
                    cbar_kws={'label': 'Correlation'}
                )
                
                plt.title(f'{region_pair} - {band_name.capitalize()} Band: Correlations by Subject Pair and Mode', fontsize=14)
                plt.tight_layout()
                
                # Save or show the figure
                if pair_dir is not None:
                    plt.savefig(f"{pair_dir}/subject_pair_correlations_{band_name}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()