#!/usr/bin/env python
"""
Spatial Loadings Regional Analysis

This script analyzes spatial loadings (PCA weights) by brain region to understand
which regions are most engaged in neural manifolds for different gestures.

Author: Neural Manifold Analysis Project
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict
import argparse

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress minor warnings

class SpatialLoadingsAnalyzer:
    """
    Analyzes spatial loadings by brain region for gesture manifolds.
    """
    
    def __init__(self, data_dir="output/spatial_loadings/brain_wide/delta_band/", 
             n_modes=3, gestures=None, subjects=None):
        """
        Initialize the analyzer with enhanced data structures.
        """
        self.data_dir = Path(data_dir)
        self.n_modes = n_modes
        self.gestures = gestures
        self.subjects = subjects
        
        # EXISTING storage
        self.spatial_data = {}  # {subject_id: {gesture: {band: DataFrame}}}
        
        # ENHANCED storage structures
        self.regional_stats = {}  # Enhanced with mode-specific stats
        self.mode_specialization = {}  # New: store specialization metrics
        self.cross_gesture_consistency = {}  # New: store consistency metrics
        self.regional_rankings = {}  # New: store various ranking schemes
        
        # Configuration for enhanced analysis
        self.specialization_threshold = 1.5  # Threshold for "specialized" regions
        self.analysis_config = {
            'compute_percentiles': True,
            'compute_effect_sizes': True,
            'compute_consistency_metrics': True,
            'bootstrap_confidence': True
        }
        
        print(f"Initialized ENHANCED Spatial Loadings Analyzer")
        print(f"Data directory: {self.data_dir}")
        print(f"Analyzing first {self.n_modes} neural modes")
        print(f"Enhanced features: Mode specialization, Cross-gesture consistency, Advanced statistics")
    
    def initialize_enhanced_stats_structure(self, band_name):
        """Initialize the enhanced statistics storage structure."""
        self.regional_stats[band_name] = {
            'by_gesture': {},
            'combined': {},
            
            # NEW: Mode-specific storage
            'mode_specific': {
                'by_gesture': {},  # Mode stats for each gesture
                'combined': {},    # Mode stats across all gestures
                'specialization': {},  # Specialization metrics
                'consistency': {}  # Cross-gesture consistency
            },
            
            # NEW: Advanced metrics
            'effect_sizes': {},      # Cohen's d between gestures
            'confidence_intervals': {},  # Bootstrap CIs
            'statistical_tests': {}  # Significance tests
        }
    
    def extract_region_from_channel(self, channel_name):
        """
        Extract brain region from channel name.
        
        Examples: 'seeg-117_caudalmiddlefrontal_rh' → 'caudalmiddlefrontal'
        """
        try:
            # Split by underscores and get the region part
            parts = channel_name.split('_')
            if len(parts) >= 2:
                # Remove hemisphere suffix if present
                region = parts[1].replace('_rh', '').replace('_lh', '')
                return region
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    def parse_spatial_loadings_file(self, filepath):
        """
        Parse a single spatial loadings text file.
        
        Parameters:
        -----------
        filepath : Path
            Path to spatial loadings text file
            
        Returns:
        --------
        df : DataFrame
            DataFrame with columns: channel_name, region, mode_1_weight, mode_2_weight, mode_3_weight
        metadata : dict
            Metadata extracted from file header
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Find the data start (after header)
            data_start_idx = None
            metadata = {}
            
            for i, line in enumerate(lines):
                if line.startswith('Channel_Name\t'):
                    data_start_idx = i
                    break
                elif 'Subject' in line and 'Band' in line:
                    # Extract subject and band info
                    match = re.search(r'Subject (\w+).*?(\w+) Band', line)
                    if match:
                        metadata['subject_id'] = match.group(1)
                        metadata['band_name'] = match.group(2)
                elif 'Component' in line and '%' in line:
                    # Extract explained variance
                    if 'explained_variance' not in metadata:
                        metadata['explained_variance'] = []
                    match = re.search(r'(\d+\.\d+)%', line)
                    if match:
                        metadata['explained_variance'].append(float(match.group(1)))
            
            if data_start_idx is None:
                raise ValueError("Could not find data section in file")
            
            # Read the data section
            data_lines = []
            for i in range(data_start_idx + 1, len(lines)):
                line = lines[i].strip()
                if line and not line.startswith('=') and 'SUMMARY' not in line:
                    data_lines.append(line)
                elif line.startswith('=') or 'SUMMARY' in line:
                    break  # Stop at summary section
            
            # Parse data into DataFrame
            data_rows = []
            for line in data_lines:
                parts = line.split('\t')
                if len(parts) >= 2 + self.n_modes:  # channel, region, + mode weights
                    row = {
                        'channel_name': parts[0],
                        'region': parts[1]
                    }
                    # Add mode weights
                    for mode_idx in range(self.n_modes):
                        try:
                            weight = float(parts[2 + mode_idx])
                            row[f'mode_{mode_idx + 1}_weight'] = weight
                        except (ValueError, IndexError):
                            row[f'mode_{mode_idx + 1}_weight'] = np.nan
                    
                    data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            
            if len(df) == 0:
                print(f"Warning: No data rows parsed from {filepath.name}")
                return None, None
            
            # Add magnitude columns
            for mode_idx in range(self.n_modes):
                weight_col = f'mode_{mode_idx + 1}_weight'
                mag_col = f'mode_{mode_idx + 1}_magnitude'
                if weight_col in df.columns:
                    df[mag_col] = np.abs(df[weight_col])
                else:
                    print(f"Warning: Column {weight_col} not found in {filepath.name}")
                    df[mag_col] = np.nan
            
            # Add average magnitude across modes
            magnitude_cols = [f'mode_{i+1}_magnitude' for i in range(self.n_modes)]
            available_mag_cols = [col for col in magnitude_cols if col in df.columns]
            if available_mag_cols:
                df['avg_magnitude'] = df[available_mag_cols].mean(axis=1)
            else:
                print(f"Warning: No magnitude columns found in {filepath.name}")
                df['avg_magnitude'] = np.nan
            
            print(f"Parsed {len(df)} channels from {filepath.name}")
            print(f"  Columns: {list(df.columns)}")
            return df, metadata
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None, None
    
    def load_all_spatial_loadings(self):
        """
        Load all spatial loadings files from the data directory.
        Expects directory structure: gesture_*/spatial_loadings_subject_*.txt
        """
        print(f"\nLoading spatial loadings from {self.data_dir}")
        
        # Find all spatial loadings files in subdirectories
        pattern = "**/spatial_loadings_subject_*.txt"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            print(f"No files matching pattern '{pattern}' found in {self.data_dir}")
            print("Expected structure: gesture_*/spatial_loadings_subject_*.txt")
            return
        
        print(f"Found {len(files)} spatial loadings files")
        
        # Parse each file
        for filepath in files:
            df, metadata = self.parse_spatial_loadings_file(filepath)
            
            if df is not None and metadata is not None:
                # Extract subject, gesture, band from filename and metadata
                gesture = self.extract_gesture_from_path(filepath)
                subject_id = metadata.get('subject_id', 'unknown')
                band_name = metadata.get('band_name', 'unknown')
                
                print(f"  Loading: Subject {subject_id}, Gesture {gesture}, Band {band_name}")
                
                # Store in nested dictionary
                if subject_id not in self.spatial_data:
                    self.spatial_data[subject_id] = {}
                if gesture not in self.spatial_data[subject_id]:
                    self.spatial_data[subject_id][gesture] = {}
                
                self.spatial_data[subject_id][gesture][band_name] = {
                    'data': df,
                    'metadata': metadata,
                    'filepath': filepath
                }
        
        # Update gestures and subjects lists if not provided
        if self.gestures is None:
            self.gestures = self.get_available_gestures()
        if self.subjects is None:
            self.subjects = self.get_available_subjects()
        
        print(f"Loaded data for {len(self.subjects)} subjects and {len(self.gestures)} gestures")
        print(f"Subjects: {self.subjects}")
        print(f"Gestures: {self.gestures}")
    
    def extract_gesture_from_path(self, filepath):
        """Extract gesture name from file path."""
        # Try to extract from directory structure or filename
        path_parts = filepath.parts
        
        # Look for gesture in directory names (e.g., "gesture_elbow" → "elbow")
        for part in path_parts:
            if part.startswith('gesture_'):
                gesture_name = part.replace('gesture_', '')
                return gesture_name
        
        # Look for gesture in filename pattern (e.g., "delta_elbow" → "elbow")
        filename = filepath.name.lower()
        gesture_keywords = ['elbow', 'scissor', 'rock', 'rotation', 'thumb']
        for gesture in gesture_keywords:
            if gesture in filename:
                return gesture
        
        # Default fallback
        return 'unknown'
    
    def get_available_subjects(self):
        """Get list of available subjects."""
        return sorted(list(self.spatial_data.keys()))
    
    def get_available_gestures(self):
        """Get list of available gestures."""
        gestures = set()
        for subject_data in self.spatial_data.values():
            gestures.update(subject_data.keys())
        return sorted(list(gestures))
        
    # STEP 1.2: Replace core computation method
    def compute_regional_statistics_enhanced(self, band_name='delta'):
        """
        ENHANCED regional statistics computation using region-first approach.
        Computes mode-specific regional statistics before overall averaging.
        """
        print(f"\n{'='*60}")
        print(f"ENHANCED REGIONAL STATISTICS COMPUTATION - {band_name.upper()} BAND")
        print(f"{'='*60}")
        
        # Initialize enhanced storage structure
        self.initialize_enhanced_stats_structure(band_name)
        
        # Find matching bands (keep existing logic)
        available_bands = set()
        for subject_data in self.spatial_data.values():
            for gesture_data in subject_data.values():
                available_bands.update(gesture_data.keys())
        
        matching_bands = [band for band in available_bands 
                        if band_name.lower() in band.lower()]
        
        if not matching_bands:
            print(f"No bands matching '{band_name}' found. Available: {list(available_bands)}")
            return
        
        print(f"Found matching bands: {matching_bands}")
        
        # PHASE 1: Process each gesture with REGION-FIRST approach
        all_gesture_data = {}  # Store for combined analysis
        
        for gesture in self.gestures:
            print(f"\nProcessing gesture: {gesture}")
            
            # Collect raw data for this gesture
            gesture_data = []
            for subject_id in self.subjects:
                if (subject_id in self.spatial_data and 
                    gesture in self.spatial_data[subject_id]):
                    
                    # Find matching band
                    for band in matching_bands:
                        if band in self.spatial_data[subject_id][gesture]:
                            df = self.spatial_data[subject_id][gesture][band]['data']
                            df_copy = df.copy()
                            df_copy['subject_id'] = subject_id
                            df_copy['gesture'] = gesture
                            gesture_data.append(df_copy)
                            break
            
            if not gesture_data:
                print(f"  No data found for gesture: {gesture}")
                continue
                
            # Combine all subjects for this gesture
            combined_df = pd.concat(gesture_data, ignore_index=True)
            all_gesture_data[gesture] = combined_df
            
            # REGION-FIRST COMPUTATION: Mode-specific regional statistics
            mode_specific_regional = self._compute_mode_specific_regional_stats(combined_df)
            
            # Compute overall regional stats from mode-specific means
            overall_regional = self._compute_overall_from_mode_stats(mode_specific_regional)
            
            # Compute specialization metrics
            specialization_stats = self._compute_specialization_metrics(mode_specific_regional)
            
            # Store results
            self.regional_stats[band_name]['by_gesture'][gesture] = {
                'mode_specific': mode_specific_regional,
                'overall': overall_regional,
                'specialization': specialization_stats,
                'raw_data': combined_df
            }
            
            print(f"  Processed: {len(combined_df)} channels, {len(mode_specific_regional)} regions")
        
        # PHASE 2: Combined analysis across all gestures
        if all_gesture_data:
            print(f"\nComputing combined statistics across {len(all_gesture_data)} gestures...")
            
            # Combine all gesture data
            all_combined_df = pd.concat(list(all_gesture_data.values()), ignore_index=True)
            
            # Region-first computation for combined data
            combined_mode_specific = self._compute_mode_specific_regional_stats(all_combined_df)
            combined_overall = self._compute_overall_from_mode_stats(combined_mode_specific)
            combined_specialization = self._compute_specialization_metrics(combined_mode_specific)
            
            # Store combined results
            self.regional_stats[band_name]['combined'] = {
                'mode_specific': combined_mode_specific,
                'overall': combined_overall,
                'specialization': combined_specialization,
                'raw_data': all_combined_df
            }
            
            # PHASE 3: Cross-gesture consistency analysis
            if len(all_gesture_data) > 1:
                print("Computing cross-gesture consistency metrics...")
                consistency_stats = self._compute_cross_gesture_consistency(all_gesture_data)
                self.regional_stats[band_name]['mode_specific']['consistency'] = consistency_stats
            
            # PHASE 4: Advanced statistical analysis (if enabled)
            if self.analysis_config['compute_effect_sizes']:
                print("Computing effect sizes between gestures...")
                effect_sizes = self._compute_effect_sizes_between_gestures(all_gesture_data)
                self.regional_stats[band_name]['effect_sizes'] = effect_sizes
            
            print(f"Enhanced computation completed: {len(all_combined_df)} total channels")
        
        print(f"{'='*60}")

    def _compute_mode_specific_regional_stats(self, combined_df):
        """
        Core method: Compute regional statistics for each mode separately.
        This is the KEY change from channel-first to region-first approach.
        """
        mode_specific_stats = {}
        
        # For each neural mode, compute regional statistics
        for mode_idx in range(1, self.n_modes + 1):
            mode_col = f'mode_{mode_idx}_magnitude'
            
            # Group by region and compute comprehensive statistics
            regional_stats = combined_df.groupby('region')[mode_col].agg([
                'mean', 'std', 'count', 'median', 'min', 'max',
                lambda x: np.percentile(x, 25),  # Q1
                lambda x: np.percentile(x, 75),  # Q3
                lambda x: np.percentile(x, 95),  # 95th percentile
                'skew'  # Distribution shape
            ]).round(6)
            
            # Rename columns for clarity
            regional_stats.columns = [
                f'mode_{mode_idx}_mean', f'mode_{mode_idx}_std', f'mode_{mode_idx}_count',
                f'mode_{mode_idx}_median', f'mode_{mode_idx}_min', f'mode_{mode_idx}_max',
                f'mode_{mode_idx}_q25', f'mode_{mode_idx}_q75', f'mode_{mode_idx}_p95',
                f'mode_{mode_idx}_skew'
            ]
            
            mode_specific_stats[f'mode_{mode_idx}'] = regional_stats
        
        # Combine all mode statistics into single DataFrame
        all_mode_stats = pd.concat(list(mode_specific_stats.values()), axis=1)
        
        # Fill any missing values
        all_mode_stats = all_mode_stats.fillna(0)
        
        return all_mode_stats

    def _compute_overall_from_mode_stats(self, mode_specific_regional):
        """
        Compute overall regional engagement from mode-specific regional means.
        This is the region-first approach: average regional means, not channel means.
        """
        # Extract mode means for each region
        mode_mean_cols = [f'mode_{i}_mean' for i in range(1, self.n_modes + 1)]
        
        overall_stats = pd.DataFrame(index=mode_specific_regional.index)
        
        # Overall mean: average of regional mode means
        overall_stats['overall_mean'] = mode_specific_regional[mode_mean_cols].mean(axis=1)
        
        # Overall std: weighted by mode counts (more robust)
        mode_count_cols = [f'mode_{i}_count' for i in range(1, self.n_modes + 1)]
        mode_std_cols = [f'mode_{i}_std' for i in range(1, self.n_modes + 1)]
        
        # Weighted standard deviation across modes
        weights = mode_specific_regional[mode_count_cols].values
        stds = mode_specific_regional[mode_std_cols].values
        
        # Compute pooled standard deviation
        overall_stats['overall_std'] = np.sqrt(
            np.average(stds**2, weights=weights, axis=1)
        )
        
        # Overall count: should be same across modes (use mode 1)
        overall_stats['overall_count'] = mode_specific_regional[f'mode_1_count']
        
        # Overall median: median of mode medians
        mode_median_cols = [f'mode_{i}_median' for i in range(1, self.n_modes + 1)]
        overall_stats['overall_median'] = mode_specific_regional[mode_median_cols].median(axis=1)
        
        # Overall range metrics
        overall_stats['overall_min'] = mode_specific_regional[
            [f'mode_{i}_min' for i in range(1, self.n_modes + 1)]
        ].min(axis=1)
        
        overall_stats['overall_max'] = mode_specific_regional[
            [f'mode_{i}_max' for i in range(1, self.n_modes + 1)]
        ].max(axis=1)
        
        return overall_stats

    def _compute_specialization_metrics(self, mode_specific_regional):
        """
        Compute mode specialization metrics for each region.
        """
        mode_mean_cols = [f'mode_{i}_mean' for i in range(1, self.n_modes + 1)]
        mode_means = mode_specific_regional[mode_mean_cols]
        
        specialization_stats = pd.DataFrame(index=mode_specific_regional.index)
        
        # Specialization ratio: max mode / mean mode
        specialization_stats['specialization_ratio'] = (
            mode_means.max(axis=1) / mode_means.mean(axis=1)
        )
        
        # Dominant mode identification
        specialization_stats['dominant_mode'] = (
            mode_means.idxmax(axis=1).str.replace('_mean', '').str.replace('mode_', 'Mode ')
        )
        
        # Specialization strength: max - second max (absolute difference)
        sorted_modes = np.sort(mode_means.values, axis=1)
        specialization_stats['specialization_strength'] = sorted_modes[:, -1] - sorted_modes[:, -2]
        
        # Mode entropy: measure of how evenly distributed across modes
        # Lower entropy = more specialized
        mode_probs = mode_means.div(mode_means.sum(axis=1), axis=0)
        specialization_stats['mode_entropy'] = -np.sum(
            mode_probs * np.log(mode_probs + 1e-10), axis=1
        )
        
        # Coefficient of variation across modes
        specialization_stats['mode_cv'] = mode_means.std(axis=1) / mode_means.mean(axis=1)
        
        # Binary specialization flag
        specialization_stats['is_specialized'] = (
            specialization_stats['specialization_ratio'] > self.specialization_threshold
        )
        
        return specialization_stats
    
    # STEP 2.1 Cross-gesture consistency analysis
    def _compute_cross_gesture_consistency(self, all_gesture_data):
        """
        Compute cross-gesture consistency metrics for regional engagement patterns.
        Identifies regions with stable vs. dynamic engagement across gestures.
        """
        consistency_results = {}
        
        # Get all unique regions across all gestures
        all_regions = set()
        for gesture_df in all_gesture_data.values():
            all_regions.update(gesture_df['region'].unique())
        
        all_regions = sorted(list(all_regions))
        
        # For each region, compute consistency metrics
        for region in all_regions:
            region_consistency = {}
            
            # Extract mode patterns for this region across gestures
            gesture_patterns = {}
            for gesture, df in all_gesture_data.items():
                region_data = df[df['region'] == region]
                if len(region_data) > 0:
                    # Compute mode means for this region in this gesture
                    mode_means = []
                    for mode_idx in range(1, self.n_modes + 1):
                        mode_col = f'mode_{mode_idx}_magnitude'
                        mode_mean = region_data[mode_col].mean()
                        mode_means.append(mode_mean)
                    gesture_patterns[gesture] = np.array(mode_means)
            
            if len(gesture_patterns) < 2:
                continue  # Need at least 2 gestures for consistency
            
            # Compute pairwise correlations between gesture patterns
            gestures = list(gesture_patterns.keys())
            correlations = []
            pattern_differences = []
            
            for i, gesture1 in enumerate(gestures):
                for j, gesture2 in enumerate(gestures[i+1:], i+1):
                    # Correlation between mode patterns
                    corr = np.corrcoef(gesture_patterns[gesture1], gesture_patterns[gesture2])[0, 1]
                    correlations.append(corr)
                    
                    # Euclidean distance between patterns
                    diff = np.linalg.norm(gesture_patterns[gesture1] - gesture_patterns[gesture2])
                    pattern_differences.append(diff)
            
            # Summary statistics
            region_consistency = {
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'min_correlation': np.min(correlations),
                'max_correlation': np.max(correlations),
                'mean_pattern_difference': np.mean(pattern_differences),
                'std_pattern_difference': np.std(pattern_differences),
                'n_gesture_pairs': len(correlations),
                'consistency_score': np.mean(correlations),  # Higher = more consistent
                'gesture_patterns': gesture_patterns  # Store for detailed analysis
            }
            
            # Classify consistency level
            mean_corr = region_consistency['mean_correlation']
            if mean_corr > 0.8:
                consistency_level = 'Highly Consistent'
            elif mean_corr > 0.5:
                consistency_level = 'Moderately Consistent'
            elif mean_corr > 0.2:
                consistency_level = 'Low Consistency'
            else:
                consistency_level = 'Highly Variable'
            
            region_consistency['consistency_level'] = consistency_level
            consistency_results[region] = region_consistency
        
        return consistency_results

    def _compute_effect_sizes_between_gestures(self, all_gesture_data):
        """
        Compute Cohen's d effect sizes between gestures for each region and mode.
        Identifies regions with significant differences between gestures.
        """
        from scipy import stats
        
        effect_sizes = {}
        gesture_pairs = []
        
        # Generate all gesture pairs
        gestures = list(all_gesture_data.keys())
        for i, gesture1 in enumerate(gestures):
            for gesture2 in gestures[i+1:]:
                gesture_pairs.append((gesture1, gesture2))
        
        # For each gesture pair
        for gesture1, gesture2 in gesture_pairs:
            pair_name = f"{gesture1}_vs_{gesture2}"
            effect_sizes[pair_name] = {}
            
            df1 = all_gesture_data[gesture1]
            df2 = all_gesture_data[gesture2]
            
            # Get common regions
            regions1 = set(df1['region'].unique())
            regions2 = set(df2['region'].unique())
            common_regions = regions1.intersection(regions2)
            
            for region in common_regions:
                region_df1 = df1[df1['region'] == region]
                region_df2 = df2[df2['region'] == region]
                
                region_effects = {}
                
                # For each mode
                for mode_idx in range(1, self.n_modes + 1):
                    mode_col = f'mode_{mode_idx}_magnitude'
                    
                    values1 = region_df1[mode_col].values
                    values2 = region_df2[mode_col].values
                    
                    if len(values1) > 1 and len(values2) > 1:
                        # Cohen's d effect size
                        mean1, mean2 = np.mean(values1), np.mean(values2)
                        std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
                        
                        # Pooled standard deviation
                        pooled_std = np.sqrt(((len(values1)-1)*std1**2 + (len(values2)-1)*std2**2) / 
                                        (len(values1) + len(values2) - 2))
                        
                        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                        
                        # T-test for significance
                        t_stat, p_value = stats.ttest_ind(values1, values2)
                        
                        region_effects[f'mode_{mode_idx}'] = {
                            'cohens_d': cohens_d,
                            'mean_diff': mean1 - mean2,
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'effect_size_magnitude': self._interpret_effect_size(abs(cohens_d))
                        }
                
                # Overall effect size (average across modes)
                mode_effects = [region_effects[f'mode_{i}']['cohens_d'] 
                            for i in range(1, self.n_modes + 1) 
                            if f'mode_{i}' in region_effects]
                
                if mode_effects:
                    region_effects['overall'] = {
                        'mean_cohens_d': np.mean(mode_effects),
                        'max_cohens_d': np.max(np.abs(mode_effects)),
                        'n_significant_modes': sum(region_effects[f'mode_{i}']['significant'] 
                                                for i in range(1, self.n_modes + 1) 
                                                if f'mode_{i}' in region_effects)
                    }
                
                effect_sizes[pair_name][region] = region_effects
        
        return effect_sizes

    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size magnitude."""
        if cohens_d < 0.2:
            return 'Negligible'
        elif cohens_d < 0.5:
            return 'Small'
        elif cohens_d < 0.8:
            return 'Medium'
        else:
            return 'Large'

    def _compute_bootstrap_confidence_intervals(self, data, n_bootstrap=1000, confidence_level=0.95):
        """
        Compute bootstrap confidence intervals for regional statistics.
        Provides robust uncertainty estimates.
        """
        np.random.seed(42)  # For reproducibility
        
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ci_lower, ci_upper, bootstrap_means

    # STEP 3.1 Mode-Specific Regional Engagement Plots
    def plot_mode_specific_regional_engagement(self, band_name='delta', gesture=None, 
                                         output_dir=None, figsize=(16, 10), top_regions=12):
        """
        ENHANCED: Plot mode-specific engagement for each brain region.
        Shows how different neural modes contribute to regional engagement.
        """
        if gesture is None:
            gesture = self.gestures[0]  # Use first available gesture
        
        if (band_name not in self.regional_stats or 
            gesture not in self.regional_stats[band_name]['by_gesture']):
            print(f"No data for {gesture} in {band_name} band")
            return
        
        # Get data
        mode_data = self.regional_stats[band_name]['by_gesture'][gesture]['mode_specific']
        specialization_data = self.regional_stats[band_name]['by_gesture'][gesture]['specialization']
        
        # Get top regions by overall engagement
        overall_data = self.regional_stats[band_name]['by_gesture'][gesture]['overall']
        top_regions_list = overall_data.nlargest(top_regions, 'overall_mean').index.tolist()
        
        # Create complex subplot layout
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[4, 1, 1])
        
        # Main plot: Mode-specific engagement by region
        ax_main = fig.add_subplot(gs[0, :2])
        
        # Prepare data for grouped bar plot
        mode_cols = [f'mode_{i}_mean' for i in range(1, self.n_modes + 1)]
        plot_data = mode_data.loc[top_regions_list]
        
        x = np.arange(len(top_regions_list))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        mode_labels = [f'Mode {i}' for i in range(1, self.n_modes + 1)]
        
        # Plot each mode with error bars
        for i, (mode_col, color, label) in enumerate(zip(mode_cols, colors, mode_labels)):
            values = plot_data[mode_col].values
            errors = plot_data[f'mode_{i+1}_std'].values
            
            bars = ax_main.bar(x + i*width, values, width, yerr=errors, 
                            label=label, color=color, alpha=0.8, capsize=3)
            
            # Add value labels on bars for significant values
            for j, (bar, val, err) in enumerate(zip(bars, values, errors)):
                if val > 0.02:  # Only label substantial values
                    ax_main.text(bar.get_x() + bar.get_width()/2, 
                            bar.get_height() + err + 0.005, 
                            f'{val:.3f}', ha='center', va='bottom', 
                            fontsize=8, rotation=0)
        
        # Format main plot
        ax_main.set_xlabel('Brain Region', fontsize=14, fontweight='bold')
        ax_main.set_ylabel('Mean Magnitude (|PCA Weight|)', fontsize=14, fontweight='bold')
        ax_main.set_title(f'Mode-Specific Regional Engagement - {gesture.capitalize()}\n'
                        f'{band_name.upper()} Band | Top {top_regions} Most Engaged Regions', 
                        fontsize=16, fontweight='bold')
        ax_main.set_xticks(x + width)
        ax_main.set_xticklabels(top_regions_list, rotation=45, ha='right', fontsize=11)
        ax_main.legend(fontsize=12, loc='upper right')
        ax_main.grid(True, alpha=0.3, axis='y')
        ax_main.set_ylim(0, ax_main.get_ylim()[1] * 1.1)  # Add space for labels
        
        # Side plot 1: Mode specialization
        ax_spec = fig.add_subplot(gs[0, 2])
        
        plot_spec_data = specialization_data.loc[top_regions_list]
        specialization = plot_spec_data['specialization_ratio'].values
        dominant_modes = plot_spec_data['dominant_mode'].values
        
        # Color bars by dominant mode
        mode_colors = {'Mode 1': colors[0], 'Mode 2': colors[1], 'Mode 3': colors[2]}
        bar_colors = [mode_colors.get(mode, 'gray') for mode in dominant_modes]
        
        bars_spec = ax_spec.barh(range(len(top_regions_list)), specialization, 
                            color=bar_colors, alpha=0.7)
        
        ax_spec.set_yticks(range(len(top_regions_list)))
        ax_spec.set_yticklabels(top_regions_list, fontsize=10)
        ax_spec.set_xlabel('Specialization\nRatio', fontsize=12, fontweight='bold')
        ax_spec.set_title('Mode\nSpecialization', fontsize=14, fontweight='bold')
        ax_spec.grid(True, alpha=0.3, axis='x')
        
        # Add specialization threshold line
        ax_spec.axvline(x=self.specialization_threshold, color='red', 
                    linestyle='--', alpha=0.7, linewidth=2)
        ax_spec.text(self.specialization_threshold + 0.1, len(top_regions_list)*0.95, 
                    f'Specialized\n(>{self.specialization_threshold})', 
                    ha='left', va='top', fontsize=9, color='red', fontweight='bold')
        
        # Bottom plot: Mode pattern radar for top 3 regions
        ax_radar = fig.add_subplot(gs[1, :])
        
        # Get top 3 regions for detailed view
        top_3_regions = top_regions_list[:3]
        
        # Create side-by-side radar-like plots
        for i, region in enumerate(top_3_regions):
            region_modes = [plot_data.loc[region, f'mode_{j}_mean'] 
                        for j in range(1, self.n_modes + 1)]
            
            # Normalize for radar plot
            max_val = max(region_modes) if max(region_modes) > 0 else 1
            normalized_modes = [val/max_val for val in region_modes]
            
            # Plot as bar chart (simpler than radar)
            x_pos = np.arange(self.n_modes) + i * (self.n_modes + 1)
            bars = ax_radar.bar(x_pos, normalized_modes, alpha=0.7, 
                            color=colors, width=0.8)
            
            # Add region label
            ax_radar.text(x_pos[1], 1.1, region, ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', rotation=0)
            
            # Add actual values as text
            for j, (bar, val) in enumerate(zip(bars, region_modes)):
                ax_radar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Format radar plot
        ax_radar.set_ylim(0, 1.3)
        ax_radar.set_ylabel('Normalized Magnitude', fontsize=12)
        ax_radar.set_title('Mode Patterns for Top 3 Regions (Normalized)', 
                        fontsize=14, fontweight='bold')
        ax_radar.set_xticks([])
        ax_radar.grid(True, alpha=0.3, axis='y')
        
        # Add mode labels at bottom
        mode_positions = []
        for i in range(len(top_3_regions)):
            for j in range(self.n_modes):
                mode_positions.append(j + i * (self.n_modes + 1))
        
        mode_labels_extended = (mode_labels * len(top_3_regions))
        ax_radar.set_xticks(mode_positions)
        ax_radar.set_xticklabels(mode_labels_extended, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'mode_specific_engagement_{gesture}_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()

    def plot_specialization_landscape(self, band_name='delta', output_dir=None, figsize=(14, 10)):
        """
        NEW: Create a comprehensive specialization landscape across all gestures.
        Shows which regions are specialized vs. generalist across different gestures.
        """
        if band_name not in self.regional_stats:
            print(f"No statistics computed for {band_name} band")
            return
        
        # Collect specialization data across all gestures
        all_specialization = {}
        all_dominant_modes = {}
        
        for gesture in self.gestures:
            if gesture in self.regional_stats[band_name]['by_gesture']:
                spec_data = self.regional_stats[band_name]['by_gesture'][gesture]['specialization']
                all_specialization[gesture] = spec_data['specialization_ratio']
                all_dominant_modes[gesture] = spec_data['dominant_mode']
        
        if not all_specialization:
            print("No specialization data available")
            return
        
        # Create DataFrame for plotting
        spec_df = pd.DataFrame(all_specialization)
        dominant_df = pd.DataFrame(all_dominant_modes)
        
        # Get common regions across all gestures
        common_regions = spec_df.dropna().index.tolist()
        spec_df = spec_df.loc[common_regions]
        dominant_df = dominant_df.loc[common_regions]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Specialization heatmap
        sns.heatmap(spec_df.T, annot=True, fmt='.2f', cmap='viridis', 
                    ax=ax1, cbar_kws={'label': 'Specialization Ratio'})
        ax1.set_title('Specialization Ratios Across Gestures', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Brain Region')
        ax1.set_ylabel('Gesture')
        
        # Plot 2: Dominant mode heatmap
        # Convert dominant modes to numeric for heatmap
        mode_map = {'Mode 1': 1, 'Mode 2': 2, 'Mode 3': 3}
        dominant_numeric = dominant_df.applymap(lambda x: mode_map.get(x, 0))
        
        sns.heatmap(dominant_numeric.T, annot=dominant_df.T.values, fmt='', 
                    cmap='Set1', ax=ax2, cbar_kws={'label': 'Dominant Mode'})
        ax2.set_title('Dominant Modes Across Gestures', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Brain Region')
        ax2.set_ylabel('Gesture')
        
        # Plot 3: Specialization consistency
        spec_consistency = spec_df.std(axis=1)  # Low std = consistent specialization
        spec_mean = spec_df.mean(axis=1)
        
        scatter = ax3.scatter(spec_mean, spec_consistency, 
                            c=spec_mean, s=60, alpha=0.7, cmap='coolwarm')
        
        for region in common_regions[:10]:  # Label top 10
            ax3.annotate(region, (spec_mean[region], spec_consistency[region]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Mean Specialization Ratio', fontsize=12)
        ax3.set_ylabel('Specialization Consistency (Std)', fontsize=12)
        ax3.set_title('Specialization Consistency vs. Level', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Mean Specialization')
        
        # Plot 4: Mode stability across gestures
        mode_stability = {}
        for region in common_regions:
            modes = dominant_df.loc[region].tolist()
            # Calculate how often the same mode dominates
            mode_counts = pd.Series(modes).value_counts()
            stability = mode_counts.iloc[0] / len(modes) if len(mode_counts) > 0 else 0
            mode_stability[region] = stability
        
        stability_series = pd.Series(mode_stability)
        stability_series.plot(kind='barh', ax=ax4, color='skyblue', alpha=0.7)
        ax4.set_xlabel('Mode Stability (Fraction Consistent)', fontsize=12)
        ax4.set_title('Dominant Mode Stability Across Gestures', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'specialization_landscape_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()

    # STEP 3.2 Cross-Gesture Comparison and Consistency Visualizations
    def plot_cross_gesture_consistency(self, band_name='delta', output_dir=None, figsize=(16, 12)):
        """
        NEW: Plot cross-gesture consistency analysis.
        Shows which regions have stable vs. variable engagement patterns across gestures.
        """
        if (band_name not in self.regional_stats or 
            'consistency' not in self.regional_stats[band_name]['mode_specific']):
            print(f"No consistency data for {band_name} band. Run enhanced analysis first.")
            return
        
        consistency_data = self.regional_stats[band_name]['mode_specific']['consistency']
        
        if not consistency_data:
            print("No consistency data available")
            return
        
        # Convert to DataFrame for easier plotting
        consistency_df = pd.DataFrame(consistency_data).T
        
        # Create comprehensive consistency plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Consistency score distribution
        consistency_scores = consistency_df['consistency_score'].dropna()
        
        ax1.hist(consistency_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(consistency_scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {consistency_scores.mean():.3f}')
        ax1.axvline(consistency_scores.median(), color='green', linestyle='--', 
                label=f'Median: {consistency_scores.median():.3f}')
        
        ax1.set_xlabel('Consistency Score (Mean Correlation)', fontsize=12)
        ax1.set_ylabel('Number of Regions', fontsize=12)
        ax1.set_title('Distribution of Cross-Gesture Consistency Scores', 
                    fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Consistency vs. Mean Engagement
        # Get overall engagement for comparison
        if 'combined' in self.regional_stats[band_name]:
            overall_engagement = self.regional_stats[band_name]['combined']['overall']
            common_regions = set(consistency_df.index) & set(overall_engagement.index)
            
            x_vals = [overall_engagement.loc[region, 'overall_mean'] for region in common_regions]
            y_vals = [consistency_df.loc[region, 'consistency_score'] for region in common_regions]
            
            scatter = ax2.scatter(x_vals, y_vals, alpha=0.7, s=60, c=y_vals, cmap='coolwarm')
            
            # Add labels for most/least consistent regions
            sorted_regions = sorted(common_regions, 
                                key=lambda r: consistency_df.loc[r, 'consistency_score'])
            
            # Label most consistent (top 3) and least consistent (bottom 3)
            for region in sorted_regions[-3:] + sorted_regions[:3]:
                x = overall_engagement.loc[region, 'overall_mean']
                y = consistency_df.loc[region, 'consistency_score']
                ax2.annotate(region, (x, y), xytext=(5, 5), 
                            textcoords='offset points', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            ax2.set_xlabel('Overall Regional Engagement', fontsize=12)
            ax2.set_ylabel('Consistency Score', fontsize=12)
            ax2.set_title('Consistency vs. Overall Engagement', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='Consistency Score')
        
        # Plot 3: Pattern difference analysis
        pattern_diffs = consistency_df['mean_pattern_difference'].dropna()
        consistency_levels = consistency_df['consistency_level'].dropna()
        
        # Create boxplot of pattern differences by consistency level
        level_order = ['Highly Consistent', 'Moderately Consistent', 'Low Consistency', 'Highly Variable']
        level_data = []
        level_labels = []
        
        for level in level_order:
            level_regions = consistency_levels[consistency_levels == level].index
            if len(level_regions) > 0:
                level_diffs = pattern_diffs[level_regions]
                level_data.append(level_diffs.values)
                level_labels.append(f'{level}\n(n={len(level_diffs)})')
        
        if level_data:
            ax3.boxplot(level_data, labels=level_labels)
            ax3.set_ylabel('Mean Pattern Difference', fontsize=12)
            ax3.set_title('Pattern Variability by Consistency Level', 
                        fontsize=14, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Gesture pattern similarity matrix
        # Create a similarity matrix between gestures based on regional patterns
        gestures = self.gestures
        n_gestures = len(gestures)
        
        similarity_matrix = np.zeros((n_gestures, n_gestures))
        
        for i, gesture1 in enumerate(gestures):
            for j, gesture2 in enumerate(gestures):
                if i <= j:
                    # Calculate average consistency between these gestures
                    gesture_similarities = []
                    
                    for region, region_data in consistency_data.items():
                        if 'gesture_patterns' in region_data:
                            patterns = region_data['gesture_patterns']
                            if gesture1 in patterns and gesture2 in patterns:
                                corr = np.corrcoef(patterns[gesture1], patterns[gesture2])[0, 1]
                                if not np.isnan(corr):
                                    gesture_similarities.append(corr)
                    
                    avg_similarity = np.mean(gesture_similarities) if gesture_similarities else 0
                    similarity_matrix[i, j] = avg_similarity
                    similarity_matrix[j, i] = avg_similarity
        
        # Plot similarity matrix
        im = ax4.imshow(similarity_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(n_gestures))
        ax4.set_yticks(range(n_gestures))
        ax4.set_xticklabels(gestures, rotation=45)
        ax4.set_yticklabels(gestures)
        ax4.set_title('Gesture Similarity Matrix\n(Based on Regional Patterns)', 
                    fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(n_gestures):
            for j in range(n_gestures):
                text = ax4.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                            ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax4, label='Pattern Similarity')
        
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'cross_gesture_consistency_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()

    def plot_gesture_comparison_matrix(self, band_name='delta', output_dir=None, figsize=(20, 16)):
        """
        NEW: Create a comprehensive gesture comparison matrix.
        Shows effect sizes and differences between all gesture pairs.
        """
        if (band_name not in self.regional_stats or 
            'effect_sizes' not in self.regional_stats[band_name]):
            print(f"No effect size data for {band_name} band. Run enhanced analysis first.")
            return
        
        effect_sizes = self.regional_stats[band_name]['effect_sizes']
        
        if not effect_sizes:
            print("No effect size data available")
            return
        
        # Get all gesture pairs and regions
        gesture_pairs = list(effect_sizes.keys())
        all_regions = set()
        
        for pair_data in effect_sizes.values():
            all_regions.update(pair_data.keys())
        
        all_regions = sorted(list(all_regions))
        
        # Create figure with subplots for different aspects
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1])
        
        # Plot 1: Overall effect size heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Create matrix of overall effect sizes
        effect_matrix = np.zeros((len(all_regions), len(gesture_pairs)))
        
        for j, pair in enumerate(gesture_pairs):
            for i, region in enumerate(all_regions):
                if region in effect_sizes[pair] and 'overall' in effect_sizes[pair][region]:
                    effect_matrix[i, j] = effect_sizes[pair][region]['overall']['mean_cohens_d']
        
        im1 = ax1.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', 
                        vmin=-2, vmax=2)  # Typical Cohen's d range
        
        ax1.set_xticks(range(len(gesture_pairs)))
        ax1.set_yticks(range(len(all_regions)))
        ax1.set_xticklabels([pair.replace('_vs_', ' vs ') for pair in gesture_pairs], 
                        rotation=45, ha='right')
        ax1.set_yticklabels(all_regions)
        ax1.set_title('Effect Sizes Between Gesture Pairs (Cohen\'s d)\nOverall Across All Modes', 
                    fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Cohen\'s d Effect Size', fontsize=12)
        
        # Plot 2: Significance pattern
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Count significant differences for each region
        sig_counts = {}
        for region in all_regions:
            total_comparisons = 0
            significant_comparisons = 0
            
            for pair_data in effect_sizes.values():
                if region in pair_data:
                    for mode_idx in range(1, self.n_modes + 1):
                        mode_key = f'mode_{mode_idx}'
                        if mode_key in pair_data[region]:
                            total_comparisons += 1
                            if pair_data[region][mode_key]['significant']:
                                significant_comparisons += 1
            
            sig_counts[region] = (significant_comparisons / total_comparisons 
                                if total_comparisons > 0 else 0)
        
        # Plot as horizontal bar chart
        regions_sorted = sorted(all_regions, key=lambda r: sig_counts[r], reverse=True)
        sig_values = [sig_counts[region] for region in regions_sorted]
        
        bars = ax2.barh(range(len(regions_sorted)), sig_values, alpha=0.7, color='coral')
        ax2.set_yticks(range(len(regions_sorted)))
        ax2.set_yticklabels(regions_sorted)
        ax2.set_xlabel('Fraction Significant\nComparisons', fontsize=12)
        ax2.set_title('Regional Gesture\nDifferentiation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Mode-specific effect sizes for top differentiating regions
        ax3 = fig.add_subplot(gs[1, :])
        
        # Get top 8 most differentiating regions
        top_diff_regions = regions_sorted[:8]
        
        # Create grouped bar plot for mode-specific effects
        x = np.arange(len(top_diff_regions))
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for mode_idx in range(1, self.n_modes + 1):
            mode_effects = []
            for region in top_diff_regions:
                # Average effect size across all gesture pairs for this mode
                region_mode_effects = []
                for pair_data in effect_sizes.values():
                    if (region in pair_data and 
                        f'mode_{mode_idx}' in pair_data[region]):
                        effect = abs(pair_data[region][f'mode_{mode_idx}']['cohens_d'])
                        region_mode_effects.append(effect)
                
                avg_effect = np.mean(region_mode_effects) if region_mode_effects else 0
                mode_effects.append(avg_effect)
            
            ax3.bar(x + (mode_idx-1)*width, mode_effects, width, 
                label=f'Mode {mode_idx}', color=colors[mode_idx-1], alpha=0.8)
        
        ax3.set_xlabel('Brain Region', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Mean |Cohen\'s d|', fontsize=14, fontweight='bold')
        ax3.set_title('Mode-Specific Effect Sizes for Most Differentiating Regions', 
                    fontsize=16, fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(top_diff_regions, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Effect size distribution
        ax4 = fig.add_subplot(gs[2, :])
        
        all_effect_sizes = []
        effect_magnitudes = []
        
        for pair_data in effect_sizes.values():
            for region_data in pair_data.values():
                for mode_idx in range(1, self.n_modes + 1):
                    mode_key = f'mode_{mode_idx}'
                    if mode_key in region_data:
                        effect = region_data[mode_key]['cohens_d']
                        magnitude = region_data[mode_key]['effect_size_magnitude']
                        all_effect_sizes.append(effect)
                        effect_magnitudes.append(magnitude)
        
        # Create histogram with magnitude coloring
        magnitude_colors = {'Negligible': 'lightgray', 'Small': 'lightblue', 
                        'Medium': 'orange', 'Large': 'red'}
        
        # Separate data by magnitude
        for magnitude in ['Negligible', 'Small', 'Medium', 'Large']:
            magnitude_effects = [effect for effect, mag in zip(all_effect_sizes, effect_magnitudes) 
                            if mag == magnitude]
            if magnitude_effects:
                ax4.hist(magnitude_effects, bins=30, alpha=0.7, 
                        label=f'{magnitude} (n={len(magnitude_effects)})',
                        color=magnitude_colors[magnitude])
        
        ax4.set_xlabel('Cohen\'s d Effect Size', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distribution of Effect Sizes by Magnitude Category', 
                    fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axvline(0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'gesture_comparison_matrix_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()

    # STEP 3.3 Enhanced Summary Reports and Integration
    def print_enhanced_analysis_summary(self, band_name='delta'):
        """
        ENHANCED: Print comprehensive summary of the analysis results.
        Includes mode specialization, consistency, and effect size insights.
        """
        if band_name not in self.regional_stats:
            print(f"No statistics computed for {band_name} band")
            return
        
        print(f"\n{'='*80}")
        print(f"ENHANCED ANALYSIS SUMMARY - {band_name.upper()} BAND")
        print(f"{'='*80}")
        
        # Overall statistics
        combined_stats = self.regional_stats[band_name]['combined']
        if 'overall' in combined_stats:
            overall_data = combined_stats['overall']
            
            print(f"\nOVERALL REGIONAL ENGAGEMENT:")
            print("-" * 40)
            
            top_regions = overall_data.sort_values('overall_mean', ascending=False).head(10)
            for i, (region, row) in enumerate(top_regions.iterrows()):
                mean_val = row['overall_mean']
                std_val = row['overall_std'] 
                count = int(row['overall_count'])
                print(f"{i+1:2d}. {region:<25} {mean_val:.4f} ± {std_val:.4f} (n={count})")
        
        # Mode specialization summary
        if 'specialization' in combined_stats:
            spec_data = combined_stats['specialization']
            
            print(f"\nMODE SPECIALIZATION ANALYSIS:")
            print("-" * 40)
            
            # Most specialized regions
            specialized = spec_data[spec_data['is_specialized'] == True]
            if len(specialized) > 0:
                specialized = specialized.sort_values('specialization_ratio', ascending=False)
                
                print(f"Most Specialized Regions ({len(specialized)} total):")
                for i, (region, row) in enumerate(specialized.head(8).iterrows()):
                    ratio = row['specialization_ratio']
                    dominant = row['dominant_mode']
                    strength = row['specialization_strength']
                    print(f"  {i+1}. {region:<20} {dominant} (ratio: {ratio:.2f}, strength: {strength:.3f})")
            
            # Mode distribution summary
            mode_counts = spec_data['dominant_mode'].value_counts()
            print(f"\nDominant Mode Distribution:")
            for mode, count in mode_counts.items():
                percentage = (count / len(spec_data)) * 100
                print(f"  {mode}: {count} regions ({percentage:.1f}%)")
            
            # Generalist regions (low specialization)
            generalists = spec_data[spec_data['specialization_ratio'] < 1.2]
            if len(generalists) > 0:
                # Sort by overall engagement from the correct DataFrame
                generalist_regions = generalists.index.tolist()
                
                # Get engagement scores for generalist regions from overall_data
                generalist_engagement = {}
                for region in generalist_regions:
                    if region in overall_data.index:
                        generalist_engagement[region] = overall_data.loc[region, 'overall_mean']
                    else:
                        generalist_engagement[region] = 0.0
                
                # Sort generalist regions by engagement
                sorted_generalists = sorted(generalist_engagement.items(), 
                                        key=lambda x: x[1], reverse=True)
                
                print(f"\nMost Engaged Generalist Regions (specialization < 1.2):")
                for i, (region, engagement) in enumerate(sorted_generalists[:5]):
                    ratio = spec_data.loc[region, 'specialization_ratio']
                    print(f"  {i+1}. {region:<20} engagement: {engagement:.3f}, specialization: {ratio:.2f}")
        
        # Cross-gesture consistency summary
        if 'consistency' in self.regional_stats[band_name]['mode_specific']:
            consistency_data = self.regional_stats[band_name]['mode_specific']['consistency']
            
            print(f"\nCROSS-GESTURE CONSISTENCY:")
            print("-" * 40)
            
            # Convert to DataFrame for analysis
            consistency_df = pd.DataFrame(consistency_data).T
            
            if len(consistency_df) > 0:
                # Convert consistency_score to numeric, handling any problematic values
                consistency_df['consistency_score'] = pd.to_numeric(
                    consistency_df['consistency_score'], errors='coerce'
                )
                
                # Filter out regions with invalid consistency scores
                valid_consistency = consistency_df.dropna(subset=['consistency_score'])
                
                if len(valid_consistency) > 0:
                    # Most consistent regions
                    most_consistent = valid_consistency.nlargest(5, 'consistency_score')
                    print(f"Most Consistent Regions:")
                    for i, (region, row) in enumerate(most_consistent.iterrows()):
                        score = row['consistency_score']
                        level = row.get('consistency_level', 'Unknown')
                        print(f"  {i+1}. {region:<20} score: {score:.3f} ({level})")
                    
                    # Most variable regions
                    most_variable = valid_consistency.nsmallest(5, 'consistency_score')
                    print(f"\nMost Variable Regions:")
                    for i, (region, row) in enumerate(most_variable.iterrows()):
                        score = row['consistency_score']
                        level = row.get('consistency_level', 'Unknown')
                        print(f"  {i+1}. {region:<20} score: {score:.3f} ({level})")
                    
                    # Consistency level distribution (if available)
                    if 'consistency_level' in valid_consistency.columns:
                        level_counts = valid_consistency['consistency_level'].value_counts()
                        print(f"\nConsistency Level Distribution:")
                        for level, count in level_counts.items():
                            percentage = (count / len(valid_consistency)) * 100
                            print(f"  {level}: {count} regions ({percentage:.1f}%)")
                    
                    print(f"\nAnalyzed consistency for {len(valid_consistency)} regions")
                else:
                    print("No valid consistency scores computed")
                    print("(This may happen with single-subject data or insufficient gesture overlap)")
        
        # Effect size summary
        if 'effect_sizes' in self.regional_stats[band_name]:
            effect_data = self.regional_stats[band_name]['effect_sizes']
            
            print(f"\nGESTURE DIFFERENTIATION (Effect Sizes):")
            print("-" * 40)
            
            # Find regions with largest effect sizes
            region_max_effects = {}
            
            for pair, pair_data in effect_data.items():
                for region, region_data in pair_data.items():
                    if 'overall' in region_data:
                        max_effect = region_data['overall']['max_cohens_d']
                        if region not in region_max_effects or max_effect > region_max_effects[region][1]:
                            region_max_effects[region] = (pair, max_effect)
            
            # Sort by effect size
            sorted_effects = sorted(region_max_effects.items(), 
                                key=lambda x: x[1][1], reverse=True)
            
            print(f"Regions with Largest Gesture Differences:")
            for i, (region, (pair, effect)) in enumerate(sorted_effects[:8]):
                magnitude = self._interpret_effect_size(effect)
                print(f"  {i+1}. {region:<20} max |d|: {effect:.3f} ({magnitude}) in {pair.replace('_vs_', ' vs ')}")
        
        # Summary statistics
        total_channels = 0
        total_regions = 0
        
        if 'overall' in combined_stats:
            total_channels = int(combined_stats['overall']['overall_count'].sum())
            total_regions = len(combined_stats['overall'])
        
        print(f"\nDATASET SUMMARY:")
        print("-" * 20)
        print(f"Total channels analyzed: {total_channels}")
        print(f"Total brain regions: {total_regions}")
        print(f"Subjects included: {len(self.subjects)}")
        print(f"Gestures included: {len(self.gestures)}")
        print(f"Neural modes analyzed: {self.n_modes}")
        
        if 'specialization' in combined_stats:
            n_specialized = len(combined_stats['specialization'][
                combined_stats['specialization']['is_specialized'] == True])
            print(f"Specialized regions: {n_specialized}/{total_regions} ({100*n_specialized/total_regions:.1f}%)")
        
        print(f"{'='*80}")

    def run_complete_enhanced_analysis(self, band_name='delta', output_dir=None):
        """
        ENHANCED: Run the complete enhanced regional analysis workflow.
        Includes all new visualizations and summary reports.
        """
        print(f"\n{'='*80}")
        print(f"RUNNING COMPLETE ENHANCED SPATIAL LOADINGS ANALYSIS")
        print(f"{'='*80}")
        
        # Set default output directory
        if output_dir is None:
            output_dir = f"output/enhanced_spatial_analysis/{band_name}_band/"
            print(f"Using default output directory: {output_dir}")
        
        # Phase 1: Load data (existing)
        print(f"\nPHASE 1: DATA LOADING")
        print("-" * 30)
        self.load_all_spatial_loadings()
        
        if not self.spatial_data:
            print("No spatial loadings data found. Exiting.")
            return
        
        # Phase 2: Enhanced computation
        print(f"\nPHASE 2: ENHANCED COMPUTATION")
        print("-" * 30)
        self.compute_regional_statistics_enhanced(band_name)
        
        # Phase 3: Create all enhanced visualizations
        print(f"\nPHASE 3: ENHANCED VISUALIZATIONS")
        print("-" * 30)
        
        # Original plots (enhanced versions)
        self.plot_regional_engagement_by_gesture(band_name, output_dir)
        self.plot_regional_engagement_combined(band_name, output_dir)
        self.plot_regional_ranking(band_name, output_dir)
        
        # NEW: Mode-specific plots for each gesture
        for gesture in self.gestures:
            self.plot_mode_specific_regional_engagement(
                band_name, gesture, output_dir)
        
        # NEW: Specialization landscape
        self.plot_specialization_landscape(band_name, output_dir)
        
        # NEW: Cross-gesture consistency (if multiple gestures)
        if len(self.gestures) > 1:
            self.plot_cross_gesture_consistency(band_name, output_dir)
            
            # Fix the typo from earlier artifact
            self.plot_gesture_comparison_matrix(band_name, output_dir)
        
        # NEW: Gesture specialization comparison (if multiple gestures)
        if len(self.gestures) > 1:
            print("Creating gesture specialization comparison...")
            self.plot_gesture_specialization_comparison(band_name, output_dir)
            
            print("Creating mode contribution heatmap...")
            self.plot_mode_contribution_heatmap(band_name, output_dir)
        
        # Phase 4: Enhanced summary reports
        print(f"\nPHASE 4: ENHANCED SUMMARY REPORTS")
        print("-" * 30)
        
        self.print_enhanced_analysis_summary(band_name)
        self.export_detailed_results_table(band_name, output_dir)
        
        print(f"\n{'='*80}")
        print(f"ENHANCED ANALYSIS COMPLETED!")
        if output_dir:
            print(f"Results saved to: {output_dir}")
        print(f"{'='*80}")

    def export_detailed_results_table(self, band_name='delta', output_dir=None):
        """
        NEW: Export detailed results to CSV files for further analysis.
        """
        if band_name not in self.regional_stats:
            return
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export 1: Mode-specific regional statistics
            if 'combined' in self.regional_stats[band_name]:
                mode_specific = self.regional_stats[band_name]['combined']['mode_specific']
                mode_specific.to_csv(os.path.join(output_dir, f'mode_specific_stats_{band_name}.csv'))
                print(f"Exported: mode_specific_stats_{band_name}.csv")
                
                # Export 2: Specialization metrics
                if 'specialization' in self.regional_stats[band_name]['combined']:
                    specialization = self.regional_stats[band_name]['combined']['specialization']
                    specialization.to_csv(os.path.join(output_dir, f'specialization_metrics_{band_name}.csv'))
                    print(f"Exported: specialization_metrics_{band_name}.csv")
            
            # Export 3: Consistency data
            if 'consistency' in self.regional_stats[band_name]['mode_specific']:
                consistency_data = self.regional_stats[band_name]['mode_specific']['consistency']
                consistency_df = pd.DataFrame(consistency_data).T
                consistency_df.to_csv(os.path.join(output_dir, f'consistency_analysis_{band_name}.csv'))
                print(f"Exported: consistency_analysis_{band_name}.csv")
            
            # Export 4: Effect sizes summary
            if 'effect_sizes' in self.regional_stats[band_name]:
                # Create summary table of effect sizes
                effect_summary = []
                
                for pair, pair_data in self.regional_stats[band_name]['effect_sizes'].items():
                    for region, region_data in pair_data.items():
                        if 'overall' in region_data:
                            row = {
                                'gesture_pair': pair,
                                'region': region,
                                'mean_cohens_d': region_data['overall']['mean_cohens_d'],
                                'max_cohens_d': region_data['overall']['max_cohens_d'],
                                'n_significant_modes': region_data['overall']['n_significant_modes']
                            }
                            effect_summary.append(row)
                
                if effect_summary:
                    effect_df = pd.DataFrame(effect_summary)
                    effect_df.to_csv(os.path.join(output_dir, f'effect_sizes_summary_{band_name}.csv'), 
                                    index=False)
                    print(f"Exported: effect_sizes_summary_{band_name}.csv")

    # STEP 4: Gesture Specialization Comparison Visualizations
    def plot_gesture_specialization_comparison(self, band_name='delta', output_dir=None, 
                                         figsize=(20, 12), top_regions=12, 
                                         region_order_by='overall_engagement'):
        """
        NEW: Create side-by-side comparison of specialization patterns across all gestures.
        Shows how regional specialization changes (or remains consistent) across different gestures.
        
        Parameters:
        -----------
        band_name : str
            Frequency band to analyze
        output_dir : str
            Directory to save the plot
        figsize : tuple
            Figure size (width, height)
        top_regions : int
            Number of top regions to display
        region_order_by : str
            How to order regions ('overall_engagement', 'specialization_ratio', 'alphabetical')
        """
        if band_name not in self.regional_stats:
            print(f"No statistics computed for {band_name} band")
            return
        
        # Check that we have gesture-specific data
        available_gestures = []
        for gesture in self.gestures:
            if (gesture in self.regional_stats[band_name]['by_gesture'] and
                'specialization' in self.regional_stats[band_name]['by_gesture'][gesture]):
                available_gestures.append(gesture)
        
        if len(available_gestures) < 2:
            print(f"Need at least 2 gestures for comparison. Found: {available_gestures}")
            return
        
        print(f"Creating gesture specialization comparison for {len(available_gestures)} gestures...")
        
        # Determine consistent region ordering across all gestures
        region_order = self._determine_consistent_region_order(
            band_name, available_gestures, top_regions, region_order_by
        )
        
        print(f"Comparing {len(region_order)} regions across {len(available_gestures)} gestures")
        
        # Create figure with subplots for each gesture
        # REMOVED sharey=True to avoid y-axis label conflicts
        fig, axes = plt.subplots(1, len(available_gestures), figsize=figsize, 
                                gridspec_kw={'wspace': 0.2})
        
        # Handle single gesture case (shouldn't happen, but safety)
        if len(available_gestures) == 1:
            axes = [axes]
        
        # Color scheme for modes
        mode_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        mode_labels = [f'Mode {i}' for i in range(1, self.n_modes + 1)]
        
        # Track global max for consistent scaling
        global_max_ratio = 0
        y_positions = np.arange(len(region_order))
        
        # Plot each gesture
        for gesture_idx, gesture in enumerate(available_gestures):
            ax = axes[gesture_idx]
            
            # Get specialization data for this gesture
            spec_data = self.regional_stats[band_name]['by_gesture'][gesture]['specialization']
            mode_data = self.regional_stats[band_name]['by_gesture'][gesture]['mode_specific']
            
            # Filter to selected regions in consistent order
            plot_spec_data = spec_data.loc[region_order]
            plot_mode_data = mode_data.loc[region_order]
            
            # Extract data for plotting
            specialization_ratios = plot_spec_data['specialization_ratio'].values
            dominant_modes = plot_spec_data['dominant_mode'].values
            
            # Update global max for scaling
            global_max_ratio = max(global_max_ratio, max(specialization_ratios))
            
            # Color bars by dominant mode
            bar_colors = []
            for dominant_mode in dominant_modes:
                if 'Mode 1' in dominant_mode:
                    bar_colors.append(mode_colors[0])
                elif 'Mode 2' in dominant_mode:
                    bar_colors.append(mode_colors[1])
                elif 'Mode 3' in dominant_mode:
                    bar_colors.append(mode_colors[2])
                else:
                    bar_colors.append('gray')
            
            # Plot horizontal bars
            bars = ax.barh(y_positions, specialization_ratios, 
                        color=bar_colors, alpha=0.8, height=0.7)
            
            # Add specialization threshold line
            ax.axvline(x=self.specialization_threshold, color='red', 
                    linestyle='--', alpha=0.7, linewidth=2)
            
            # Add value labels on bars for specialized regions
            for i, (bar, ratio, dominant) in enumerate(zip(bars, specialization_ratios, dominant_modes)):
                if ratio > self.specialization_threshold:
                    # Add ratio value
                    ax.text(ratio + 0.05, bar.get_y() + bar.get_height()/2, 
                        f'{ratio:.2f}', va='center', ha='left', fontsize=9, fontweight='bold')
                    
                    # Add mode indicator
                    mode_symbol = dominant.replace('Mode ', 'M')
                    ax.text(0.05, bar.get_y() + bar.get_height()/2, 
                        mode_symbol, va='center', ha='left', fontsize=8, 
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=bar.get_facecolor(), alpha=0.8))
            
            # Set consistent y-axis for all subplots
            ax.set_ylim(-0.5, len(region_order) - 0.5)
            ax.set_yticks(y_positions)
            
            # EXPLICIT y-axis labeling: Only on the first subplot
            if gesture_idx == 0:
                ax.set_yticklabels(region_order, fontsize=10)
            else:
                ax.set_yticklabels([])
            
            ax.set_xlabel('Specialization Ratio', fontsize=11, fontweight='bold')
            ax.set_title(f'{gesture.capitalize()}\nGesture', fontsize=12, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_xlim(0, max(3.0, global_max_ratio * 1.1))
            
            # Add background shading for specialization levels
            ax.axvspan(1.0, 1.2, alpha=0.1, color='gray', label='Generalist' if gesture_idx == 0 else "")
            ax.axvspan(1.2, 1.5, alpha=0.1, color='orange', label='Moderate' if gesture_idx == 0 else "")
            ax.axvspan(1.5, ax.get_xlim()[1], alpha=0.1, color='red', label='Specialized' if gesture_idx == 0 else "")
        
        # Add overall title and legend
        fig.suptitle(f'Regional Specialization Comparison Across Gestures - {band_name.upper()} Band\n'
                    f'Regions Ordered by {region_order_by.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Create custom legend for modes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=mode_colors[0], label='Mode 1 Dominant'),
            Patch(facecolor=mode_colors[1], label='Mode 2 Dominant'),
            Patch(facecolor=mode_colors[2], label='Mode 3 Dominant'),
            plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7, 
                    label=f'Specialization Threshold ({self.specialization_threshold})')
        ]
        
        # Place legend below the plots
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
                bbox_to_anchor=(0.5, 0.02), fontsize=10)
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'gesture_specialization_comparison_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()

    def _determine_consistent_region_order(self, band_name, available_gestures, top_regions, order_by):
        """
        Determine consistent region ordering across all gestures for comparison.
        """
        if order_by == 'overall_engagement':
            # Order by overall engagement across all gestures
            if 'combined' in self.regional_stats[band_name] and 'overall' in self.regional_stats[band_name]['combined']:
                overall_data = self.regional_stats[band_name]['combined']['overall']
                region_order = overall_data.nlargest(top_regions, 'overall_mean').index.tolist()
            else:
                # Fallback: use first gesture's overall data
                first_gesture = available_gestures[0]
                overall_data = self.regional_stats[band_name]['by_gesture'][first_gesture]['overall']
                region_order = overall_data.nlargest(top_regions, 'overall_mean').index.tolist()
                
        elif order_by == 'specialization_ratio':
            # Order by average specialization ratio across gestures
            all_spec_ratios = {}
            for gesture in available_gestures:
                spec_data = self.regional_stats[band_name]['by_gesture'][gesture]['specialization']
                for region in spec_data.index:
                    if region not in all_spec_ratios:
                        all_spec_ratios[region] = []
                    all_spec_ratios[region].append(spec_data.loc[region, 'specialization_ratio'])
            
            # Calculate average specialization ratio
            avg_spec_ratios = {region: np.mean(ratios) for region, ratios in all_spec_ratios.items()}
            sorted_regions = sorted(avg_spec_ratios.items(), key=lambda x: x[1], reverse=True)
            region_order = [region for region, _ in sorted_regions[:top_regions]]
            
        elif order_by == 'alphabetical':
            # Simple alphabetical ordering
            all_regions = set()
            for gesture in available_gestures:
                spec_data = self.regional_stats[band_name]['by_gesture'][gesture]['specialization']
                all_regions.update(spec_data.index)
            region_order = sorted(list(all_regions))[:top_regions]
            
        else:
            raise ValueError(f"Unknown ordering method: {order_by}")
        
        return region_order

    def plot_mode_contribution_heatmap(self, band_name='delta', output_dir=None, 
                                    figsize=(16, 10), top_regions=12):
        """
        BONUS: Create a heatmap showing mode contributions across gestures and regions.
        Complements the specialization comparison by showing raw mode values.
        """
        if band_name not in self.regional_stats:
            print(f"No statistics computed for {band_name} band")
            return
        
        # Get available gestures
        available_gestures = []
        for gesture in self.gestures:
            if (gesture in self.regional_stats[band_name]['by_gesture'] and
                'mode_specific' in self.regional_stats[band_name]['by_gesture'][gesture]):
                available_gestures.append(gesture)
        
        if len(available_gestures) < 2:
            print(f"Need at least 2 gestures for heatmap. Found: {available_gestures}")
            return
        
        # Determine region order
        region_order = self._determine_consistent_region_order(
            band_name, available_gestures, top_regions, 'overall_engagement'
        )
        
        # Create data matrix for heatmap
        # Rows: Regions, Columns: Gesture-Mode combinations
        column_labels = []
        for gesture in available_gestures:
            for mode_idx in range(1, self.n_modes + 1):
                column_labels.append(f'{gesture.capitalize()}\nMode {mode_idx}')
        
        # Initialize data matrix
        heatmap_data = np.zeros((len(region_order), len(column_labels)))
        
        # Fill data matrix
        col_idx = 0
        for gesture in available_gestures:
            mode_data = self.regional_stats[band_name]['by_gesture'][gesture]['mode_specific']
            for mode_idx in range(1, self.n_modes + 1):
                mode_col = f'mode_{mode_idx}_mean'
                for row_idx, region in enumerate(region_order):
                    if region in mode_data.index:
                        heatmap_data[row_idx, col_idx] = mode_data.loc[region, mode_col]
                col_idx += 1
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Use a colormap that shows both magnitude and differences
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Set labels
        ax.set_xticks(range(len(column_labels)))
        ax.set_xticklabels(column_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(len(region_order)))
        ax.set_yticklabels(region_order, fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Mode Magnitude', fontsize=12, fontweight='bold')
        
        # Add text annotations for high values
        for i in range(len(region_order)):
            for j in range(len(column_labels)):
                value = heatmap_data[i, j]
                if value > np.percentile(heatmap_data, 90):  # Top 10% values
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=8)
        
        # Add vertical lines to separate gestures
        for i in range(1, len(available_gestures)):
            ax.axvline(x=i * self.n_modes - 0.5, color='white', linewidth=2)
        
        # Title and formatting
        ax.set_title(f'Mode Contribution Heatmap Across Gestures - {band_name.upper()} Band\n'
                    f'Regional Mode Magnitudes for {len(available_gestures)} Gestures', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'mode_contribution_heatmap_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()

    # BACKWARD COMPATIBILITY METHODS (add after your existing methods)
    def compute_regional_statistics(self, band_name='delta'):
        """Backward compatibility: Calls enhanced version."""
        print("Using enhanced regional statistics computation...")
        self.compute_regional_statistics_enhanced(band_name)

    def run_complete_analysis(self, band_name='delta', output_dir=None):
        """Backward compatibility: Calls enhanced version."""
        print("Using enhanced complete analysis...")
        self.run_complete_enhanced_analysis(band_name, output_dir)

    # ORIGINAL PLOTTING METHODS (needed by run_complete_enhanced_analysis)
    def plot_regional_engagement_by_gesture(self, band_name='delta', output_dir=None, 
                                        figsize=(15, 10), top_regions=10):
        """
        Create boxplots showing regional engagement for each gesture separately.
        Enhanced version of the original method.
        """
        if band_name not in self.regional_stats:
            print(f"No statistics computed for {band_name} band. Run compute_regional_statistics_enhanced() first.")
            return
        
        # Get the top N most engaged regions overall
        if 'combined' not in self.regional_stats[band_name] or 'overall' not in self.regional_stats[band_name]['combined']:
            print(f"No combined statistics available for {band_name} band")
            return
            
        combined_stats = self.regional_stats[band_name]['combined']['overall']
        top_region_names = combined_stats.nlargest(top_regions, 'overall_mean').index.tolist()
        
        print(f"Creating regional engagement plots for top {len(top_region_names)} regions...")
        
        # Calculate subplot layout
        n_gestures = len(self.gestures)
        n_cols = min(3, n_gestures)
        n_rows = int(np.ceil(n_gestures / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
        
        # Handle single subplot case
        if n_gestures == 1:
            axes = [axes]
        elif n_rows == 1:
            pass  # axes is already a 1D array
        else:
            axes = axes.flatten()
        
        # Plot each gesture
        for i, gesture in enumerate(self.gestures):
            ax = axes[i] if n_gestures > 1 else axes[0]
            
            if gesture in self.regional_stats[band_name]['by_gesture']:
                raw_data = self.regional_stats[band_name]['by_gesture'][gesture]['raw_data']
                
                # Filter to top regions and compute avg_magnitude if needed
                plot_data = raw_data[raw_data['region'].isin(top_region_names)]
                
                if len(plot_data) > 0:
                    # Create boxplot
                    sns.boxplot(data=plot_data, x='region', y='avg_magnitude', ax=ax)
                    
                    ax.set_title(f'{gesture.capitalize()}\n(n={len(plot_data)} channels)', 
                            fontsize=12, fontweight='bold')
                    ax.set_xlabel('Brain Region', fontsize=10)
                    ax.set_ylabel('Average Magnitude (|PCA Weight|)', fontsize=10)
                    ax.tick_params(axis='x', rotation=45, labelsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Add sample sizes
                    region_counts = plot_data['region'].value_counts()
                    for j, region in enumerate(plot_data['region'].unique()):
                        if region in region_counts:
                            count = region_counts[region]
                            ax.text(j, ax.get_ylim()[1] * 0.95, f'n={count}', 
                                ha='center', va='top', fontsize=7)
                else:
                    ax.text(0.5, 0.5, f'No data\nfor {gesture}', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{gesture.capitalize()}', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'No data\nfor {gesture}', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{gesture.capitalize()}', fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for j in range(n_gestures, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Regional Engagement by Gesture - {band_name.upper()} Band\n'
                    f'Average Magnitude Across First {self.n_modes} Neural Modes', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'regional_engagement_by_gesture_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()

    def plot_regional_engagement_combined(self, band_name='delta', output_dir=None, 
                                        figsize=(12, 8), top_regions=15):
        """
        Create boxplot showing regional engagement across all gestures combined.
        Enhanced version of the original method.
        """
        if band_name not in self.regional_stats:
            print(f"No statistics computed for {band_name} band")
            return
        
        if 'combined' not in self.regional_stats[band_name] or 'raw_data' not in self.regional_stats[band_name]['combined']:
            print(f"No combined raw data available for {band_name} band")
            return
            
        combined_data = self.regional_stats[band_name]['combined']['raw_data']
        
        # Get top regions by mean engagement
        region_means = combined_data.groupby('region')['avg_magnitude'].mean().sort_values(ascending=False)
        top_region_names = region_means.head(top_regions).index.tolist()
        
        print(f"Creating combined regional engagement plot for top {len(top_region_names)} regions...")
        
        # Filter data
        plot_data = combined_data[combined_data['region'].isin(top_region_names)]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create boxplot with gesture-wise coloring
        sns.boxplot(data=plot_data, x='region', y='avg_magnitude', hue='gesture', ax=ax)
        
        # Formatting
        ax.set_title(f'Regional Engagement Across All Gestures - {band_name.upper()} Band\n'
                    f'Average Magnitude Across First {self.n_modes} Neural Modes', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Brain Region', fontsize=12)
        ax.set_ylabel('Average Magnitude (|PCA Weight|)', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(True, alpha=0.3)
        
        # Move legend outside plot
        ax.legend(title='Gesture', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'regional_engagement_combined_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()

    def plot_regional_ranking(self, band_name='delta', output_dir=None, figsize=(10, 8)):
        """
        Create a ranking plot showing mean engagement for each region.
        Enhanced version of the original method.
        """
        if band_name not in self.regional_stats:
            print(f"No statistics computed for {band_name} band")
            return
        
        if 'combined' not in self.regional_stats[band_name] or 'overall' not in self.regional_stats[band_name]['combined']:
            print(f"No combined statistics available for {band_name} band")
            return
            
        combined_stats = self.regional_stats[band_name]['combined']['overall']
        
        # Sort by mean magnitude
        ranking_data = combined_stats.sort_values('overall_mean', ascending=True)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(ranking_data))
        bars = ax.barh(y_pos, ranking_data['overall_mean'], 
                    xerr=ranking_data['overall_std'], 
                    capsize=3, alpha=0.7, color='skyblue')
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranking_data.index, fontsize=10)
        ax.set_xlabel('Mean Magnitude (|PCA Weight|)', fontsize=12)
        ax.set_title(f'Regional Engagement Ranking - {band_name.upper()} Band\n'
                    f'Mean ± SD Across All Gestures and Subjects', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(ranking_data.iterrows()):
            mean_val = row['overall_mean']
            count = row['overall_count']
            ax.text(mean_val + ranking_data['overall_std'].iloc[i], i, 
                f'{mean_val:.3f} (n={int(count)})', 
                va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'regional_ranking_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()

# TEST FUNCTION
def test_enhanced_implementation():
        """Test the enhanced implementation with your data."""
        print("🧪 TESTING ENHANCED IMPLEMENTATION")
        print("=" * 50)
        
        try:
            # Test 1: Initialization
            print("1. Testing initialization...")
            analyzer = SpatialLoadingsAnalyzer()
            print("   ✓ Enhanced initialization successful")
            
            # Test 2: Data loading
            print("2. Testing data loading...")
            analyzer.load_all_spatial_loadings()
            
            if analyzer.spatial_data:
                print(f"   ✓ Data loading successful: {len(analyzer.spatial_data)} subjects")
                print(f"     Subjects: {analyzer.subjects}")
                print(f"     Gestures: {analyzer.gestures}")
            else:
                print("   ⚠️  No data loaded - check your data directory")
                print(f"     Looking in: {analyzer.data_dir}")
                return False
            
            # Test 3: Enhanced computation
            print("3. Testing enhanced computation...")
            analyzer.compute_regional_statistics_enhanced('delta')
            print("   ✓ Enhanced computation successful")
            
            # Test 4: Check data structures
            print("4. Checking enhanced data structures...")
            assert 'delta' in analyzer.regional_stats
            assert 'combined' in analyzer.regional_stats['delta']
            assert 'mode_specific' in analyzer.regional_stats['delta']['combined']
            print("   ✓ Enhanced data structures created correctly")
            
            # Test 5: Test one visualization
            print("5. Testing enhanced visualization...")
            test_output = 'test_enhanced_output/'
            analyzer.plot_mode_specific_regional_engagement(
                'delta', analyzer.gestures[0], test_output
            )
            print("   ✓ Enhanced visualization successful")
            
            # Test 6: Test summary
            print("6. Testing enhanced summary...")
            analyzer.print_enhanced_analysis_summary('delta')
            print("   ✓ Enhanced summary successful")
            
            print("\n🎉 ALL TESTS PASSED!")
            print("Your enhanced implementation is ready to use!")
            return True
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze spatial loadings by brain region')
    parser.add_argument('--data-dir', 
                       default='output/spatial_loadings/brain_wide/delta_band/',
                       help='Directory containing spatial loadings files (default: output/spatial_loadings/brain_wide/delta_band/)')
    parser.add_argument('--band', default='delta', help='Frequency band to analyze')
    parser.add_argument('--output-dir', help='Output directory for plots (default: auto-generated based on band)')
    parser.add_argument('--n-modes', type=int, default=3, help='Number of neural modes to analyze')
    parser.add_argument('--gestures', nargs='+', help='Specific gestures to include')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to include')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SpatialLoadingsAnalyzer(
        data_dir=args.data_dir,
        n_modes=args.n_modes,
        gestures=args.gestures,
        subjects=args.subjects
    )
    
    # Run complete analysis
    analyzer.run_complete_enhanced_analysis(
        band_name=args.band,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

