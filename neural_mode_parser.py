"""
Neural Mode Brain Mapping Parser
================================

This script parses spatial loadings files from neural manifold analysis 
and prepares the data for brain visualization using MATLAB toolbox.

Usage:
    python neural_mode_parser.py --input_dir output/spatial_loadings/brain_wide/delta_band --subject 41
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy.io as sio
import argparse

class NeuralModeParser:
    """Parser for spatial loadings files from neural manifold analysis."""
    
    def __init__(self, input_dir, subject_id):
        """
        Initialize parser.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing gesture folders with spatial loadings
        subject_id : int
            Subject ID to process
        """
        self.input_dir = Path(input_dir)
        self.subject_id = subject_id
        self.gesture_data = {}
        self.electrode_info = {}
        
    def parse_spatial_loadings_file(self, file_path):
        """
        Parse a single spatial loadings file.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to spatial loadings file
            
        Returns:
        --------
        dict : Dictionary containing parsed data
        """
        print(f"Parsing: {file_path}")
        
        try:
            # Read the file, skipping header lines
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the data start (after "Channel_Name\tRegion\t...")
            data_start_idx = None
            for i, line in enumerate(lines):
                if line.startswith('Channel_Name\tRegion'):
                    data_start_idx = i + 1
                    break
            
            if data_start_idx is None:
                raise ValueError("Could not find data header in file")
            
            # Read data portion
            data_lines = lines[data_start_idx:]
            
            # Parse each data line
            channels = []
            regions = []
            mode_1_weights = []
            mode_2_weights = []
            mode_3_weights = []
            
            for line in data_lines:
                line = line.strip()
                if not line or line.startswith('='):  # Skip empty lines and separators
                    break
                    
                parts = line.split('\t')
                if len(parts) >= 5:  # Channel, Region, Mode1, Mode2, Mode3
                    channels.append(parts[0])
                    regions.append(parts[1])
                    mode_1_weights.append(float(parts[2]))
                    mode_2_weights.append(float(parts[3]))
                    mode_3_weights.append(float(parts[4]))
            
            # Create DataFrame
            data = pd.DataFrame({
                'channel': channels,
                'region': regions,
                'mode_1_weight': mode_1_weights,
                'mode_2_weight': mode_2_weights,
                'mode_3_weight': mode_3_weights
            })
            
            print(f"  Loaded {len(data)} electrodes")
            return data
            
        except Exception as e:
            print(f"  ERROR parsing {file_path}: {e}")
            return None
    
    def load_all_gestures(self):
        """Load spatial loadings for all available gestures."""
        
        gesture_folders = [
            'gesture_elbow', 'gesture_scissor', 'gesture_rock', 
            'gesture_rotation', 'gesture_thumb'
        ]
        
        for gesture_folder in gesture_folders:
            gesture_path = self.input_dir / gesture_folder
            
            if not gesture_path.exists():
                print(f"Warning: {gesture_path} not found, skipping...")
                continue
            
            # Look for spatial loadings file for this subject
            pattern = f"spatial_loadings_subject_{self.subject_id}_delta_*.txt"
            matching_files = list(gesture_path.glob(pattern))
            
            if not matching_files:
                print(f"Warning: No spatial loadings file found in {gesture_path}")
                continue
            
            # Parse the file
            file_path = matching_files[0]
            gesture_name = gesture_folder.replace('gesture_', '')
            
            data = self.parse_spatial_loadings_file(file_path)
            if data is not None:
                self.gesture_data[gesture_name] = data
                print(f"  Successfully loaded {gesture_name} gesture")
        
        print(f"\nLoaded {len(self.gesture_data)} gestures: {list(self.gesture_data.keys())}")
    
    def extract_electrode_info(self):
        """Extract electrode names and regions from the data."""
        
        if not self.gesture_data:
            raise ValueError("No gesture data loaded. Call load_all_gestures() first.")
        
        # Use first gesture to get electrode info (should be same across gestures)
        first_gesture = list(self.gesture_data.values())[0]
        
        self.electrode_info = {
            'channels': first_gesture['channel'].tolist(),
            'regions': first_gesture['region'].tolist(),
            'n_electrodes': len(first_gesture)
        }
        
        # Verify all gestures have same electrodes
        for gesture_name, data in self.gesture_data.items():
            if not data['channel'].equals(first_gesture['channel']):
                print(f"Warning: {gesture_name} has different electrodes!")
        
        print(f"Electrode info extracted: {self.electrode_info['n_electrodes']} electrodes")
    
    def create_neural_mode_matrices(self):
        """
        Create matrices for each neural mode across all gestures.
        
        Returns:
        --------
        dict : Dictionary with mode matrices (electrodes x gestures)
        """
        if not self.gesture_data:
            raise ValueError("No gesture data loaded")
        
        n_electrodes = self.electrode_info['n_electrodes']
        gesture_names = list(self.gesture_data.keys())
        n_gestures = len(gesture_names)
        
        # Initialize matrices
        mode_matrices = {
            'mode_1': np.zeros((n_electrodes, n_gestures)),
            'mode_2': np.zeros((n_electrodes, n_gestures)),
            'mode_3': np.zeros((n_electrodes, n_gestures))
        }
        
        # Fill matrices
        for j, gesture_name in enumerate(gesture_names):
            data = self.gesture_data[gesture_name]
            mode_matrices['mode_1'][:, j] = data['mode_1_weight'].values
            mode_matrices['mode_2'][:, j] = data['mode_2_weight'].values
            mode_matrices['mode_3'][:, j] = data['mode_3_weight'].values
        
        # Store gesture names for reference
        mode_matrices['gesture_names'] = gesture_names
        mode_matrices['electrode_names'] = self.electrode_info['channels']
        mode_matrices['regions'] = self.electrode_info['regions']
        
        print(f"Created neural mode matrices: {n_electrodes} electrodes x {n_gestures} gestures")
        return mode_matrices
    
    def save_for_matlab(self, mode_matrices, output_dir):
        """
        Save data in MATLAB format for visualization toolbox.
        
        Parameters:
        -----------
        mode_matrices : dict
            Neural mode matrices from create_neural_mode_matrices()
        output_dir : str or Path
            Directory to save MATLAB files
        """
        import numpy as np  # Ensure numpy is available
        
        output_dir = Path(output_dir)
        
        # Create subject-specific directory (P41 format)
        subject_dir = output_dir / f"P{self.subject_id}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert string lists to numpy arrays for better MATLAB compatibility
        # Convert gesture names to numpy array of objects (for MATLAB cell array)
        gesture_names_array = np.array(mode_matrices['gesture_names'], dtype=object)
        electrode_names_array = np.array(mode_matrices['electrode_names'], dtype=object)  
        regions_array = np.array(mode_matrices['regions'], dtype=object)
        
        # Save neural mode data
        matlab_data = {
            'subject_id': self.subject_id,
            'gesture_names': gesture_names_array,  # Numpy object array
            'electrode_names': electrode_names_array,  # Numpy object array
            'regions': regions_array,  # Numpy object array
            'mode_1_weights': mode_matrices['mode_1'],
            'mode_2_weights': mode_matrices['mode_2'], 
            'mode_3_weights': mode_matrices['mode_3'],
            'n_electrodes': len(mode_matrices['electrode_names']),
            'n_gestures': len(mode_matrices['gesture_names'])
        }
        
        # Save to .mat file in subject directory with proper string handling
        mat_file = subject_dir / f'neural_modes_subject_{self.subject_id}.mat'
        sio.savemat(mat_file, matlab_data, format='5', do_compression=False, oned_as='column')
        print(f"Saved MATLAB data: {mat_file}")
        
        # Save individual gesture files for easy loading
        for i, gesture in enumerate(mode_matrices['gesture_names']):
            gesture_data = {
                'gesture_name': gesture,
                'electrode_names': electrode_names_array,
                'regions': regions_array,
                'mode_1_activations': mode_matrices['mode_1'][:, i],
                'mode_2_activations': mode_matrices['mode_2'][:, i],
                'mode_3_activations': mode_matrices['mode_3'][:, i]
            }
            
            gesture_file = subject_dir / f'neural_modes_{gesture}_subject_{self.subject_id}.mat'
            sio.savemat(gesture_file, gesture_data, format='5', do_compression=False, oned_as='column')
            print(f"  Saved {gesture}: {gesture_file}")
        
        return mat_file
    
    def create_summary_visualizations(self, mode_matrices, output_dir):
        """Create summary plots of the neural mode data."""
        
        output_dir = Path(output_dir)
        
        # 1. Heatmap of neural mode weights across gestures
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, mode_name in enumerate(['mode_1', 'mode_2', 'mode_3']):
            weights = mode_matrices[mode_name]
            
            im = axes[i].imshow(weights, aspect='auto', cmap='RdBu_r', 
                              vmin=-np.max(np.abs(weights)), vmax=np.max(np.abs(weights)))
            axes[i].set_title(f'Neural Mode {i+1} Weights\nAcross Gestures', fontsize=12)
            axes[i].set_xlabel('Gestures')
            axes[i].set_ylabel('Electrodes')
            axes[i].set_xticks(range(len(mode_matrices['gesture_names'])))
            axes[i].set_xticklabels([g.capitalize() for g in mode_matrices['gesture_names']], 
                                   rotation=45)
            
            plt.colorbar(im, ax=axes[i], label='Weight')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'neural_modes_heatmap_subject_{self.subject_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribution of weights for each mode
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, mode_name in enumerate(['mode_1', 'mode_2', 'mode_3']):
            weights = mode_matrices[mode_name].flatten()
            
            axes[i].hist(weights, bins=50, alpha=0.7, color=f'C{i}')
            axes[i].set_title(f'Neural Mode {i+1} Weight Distribution')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(0, color='black', linestyle='--', alpha=0.5)
            
            # Add statistics
            mean_w = np.mean(weights)
            std_w = np.std(weights)
            axes[i].text(0.05, 0.95, f'Mean: {mean_w:.3f}\nStd: {std_w:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'neural_modes_distributions_subject_{self.subject_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top electrodes for each mode
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for i, mode_name in enumerate(['mode_1', 'mode_2', 'mode_3']):
            # Get mean absolute weight across gestures for each electrode
            mean_abs_weights = np.mean(np.abs(mode_matrices[mode_name]), axis=1)
            
            # Get top 10 electrodes
            top_indices = np.argsort(mean_abs_weights)[-10:][::-1]
            top_names = [mode_matrices['electrode_names'][idx] for idx in top_indices]
            top_values = mean_abs_weights[top_indices]
            
            bars = axes[i].barh(range(10), top_values, color=f'C{i}', alpha=0.7)
            axes[i].set_yticks(range(10))
            axes[i].set_yticklabels(top_names)
            axes[i].set_xlabel('Mean Absolute Weight')
            axes[i].set_title(f'Top 10 Electrodes - Neural Mode {i+1}')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels
            for j, (bar, val) in enumerate(zip(bars, top_values)):
                axes[i].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'top_electrodes_subject_{self.subject_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary visualizations saved to: {output_dir}")


def main():
    """Main function to run the neural mode parser."""
    
    parser = argparse.ArgumentParser(description='Parse neural mode spatial loadings for brain mapping')
    parser.add_argument('--input_dir', 
                       default='output/spatial_loadings/brain_wide/delta_band',
                       help='Directory containing gesture folders (e.g., output/spatial_loadings/precentral-rh/delta_band)')
    parser.add_argument('--subject', type=int, default=41,
                       help='Subject ID to process')
    parser.add_argument('--output_dir', default='output/matlab_brain_data',
                       help='Output directory for MATLAB files (will create P{subject} subdirectory)')
    
    args = parser.parse_args()
    
    print("🧠 Neural Mode Brain Mapping Parser")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Subject ID: {args.subject}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Verify input directory exists
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        print("\n🔍 To find your spatial loadings files, try:")
        print("   find . -name '*spatial_loadings_subject*' -type f")
        print("\n📂 Expected directory structure:")
        print("   output/spatial_loadings/[region_name]/[band_name]_band/")
        print("   ├── gesture_elbow/")
        print("   ├── gesture_scissor/")
        print("   ├── gesture_rock/")
        print("   ├── gesture_rotation/")
        print("   └── gesture_thumb/")
        return
    
    # Check if gesture folders exist
    gesture_folders = ['gesture_elbow', 'gesture_scissor', 'gesture_rock', 
                      'gesture_rotation', 'gesture_thumb']
    found_folders = [f for f in gesture_folders if (input_path / f).exists()]
    
    if not found_folders:
        print(f"❌ Error: No gesture folders found in {args.input_dir}")
        print(f"🔍 Directory contents:")
        try:
            contents = list(input_path.iterdir())
            for item in contents:
                print(f"   {item.name}")
        except:
            print("   (Could not list directory contents)")
        print(f"\n📂 Expected folders: {gesture_folders}")
        return
    
    print(f"✅ Found {len(found_folders)} gesture folders: {found_folders}")
    print()
    
    # Initialize parser
    parser_obj = NeuralModeParser(args.input_dir, args.subject)
    
    # Load all gesture data
    print("📁 Loading spatial loadings files...")
    parser_obj.load_all_gestures()
    
    if not parser_obj.gesture_data:
        print("❌ No gesture data loaded. Check input directory and files.")
        return
    
    # Extract electrode info
    print("\n🔍 Extracting electrode information...")
    parser_obj.extract_electrode_info()
    
    # Create neural mode matrices
    print("\n📊 Creating neural mode matrices...")
    mode_matrices = parser_obj.create_neural_mode_matrices()
    
    # Save for MATLAB
    print("\n💾 Saving data for MATLAB visualization...")
    output_dir = Path(args.output_dir)
    mat_file = parser_obj.save_for_matlab(mode_matrices, output_dir)
    
    # Create summary visualizations
    print("\n📈 Creating summary visualizations...")
    parser_obj.create_summary_visualizations(mode_matrices, output_dir)
    
    print("\n✅ Processing complete!")
    print(f"\nNext steps:")
    print(f"1. Copy files from {output_dir}/P{args.subject}/ to your MATLAB working directory")
    print(f"2. Ensure electrode file exists at: electrode_data/P{args.subject}/electrodes_Final_Norm.mat")
    print(f"3. Run MATLAB visualization: NeuralModeBrainViz")
    
    # Print some key statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Subject: P{args.subject}")
    print(f"   Gestures: {len(mode_matrices['gesture_names'])}")
    print(f"   Electrodes: {len(mode_matrices['electrode_names'])}")
    print(f"   Brain regions: {len(set(mode_matrices['regions']))}")
    
    # Show top electrodes for Mode 1
    mode_1_weights = mode_matrices['mode_1']
    mean_abs_weights = np.mean(np.abs(mode_1_weights), axis=1)
    top_idx = np.argmax(mean_abs_weights)
    top_electrode = mode_matrices['electrode_names'][top_idx]
    top_region = mode_matrices['regions'][top_idx]
    top_weight = mean_abs_weights[top_idx]
    
    print(f"\n🎯 Most Important Electrode (Mode 1):")
    print(f"   {top_electrode} ({top_region}): {top_weight:.3f}")
    
    # Show file locations
    print(f"\n📁 Files created:")
    print(f"   Neural data: {output_dir}/P{args.subject}/neural_modes_subject_{args.subject}.mat")
    print(f"   Expected electrode data: electrode_data/P{args.subject}/electrodes_Final_Norm.mat")


if __name__ == "__main__":
    main()

