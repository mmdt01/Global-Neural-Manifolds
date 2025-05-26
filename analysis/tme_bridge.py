"""
TME (Tensor Maximum Entropy) MATLAB-Python Bridge for Neural Manifold Analysis

This module provides the interface between Python neural data analysis and 
MATLAB TME toolbox for generating principled null distributions.
"""

import os
import numpy as np
import scipy.io as sio
import subprocess
import tempfile
import shutil
from pathlib import Path
import logging

class TMEBridge:
    """
    Bridge between Python and MATLAB TME toolbox for null hypothesis testing.
    
    This class handles:
    1. Data preparation and export to MATLAB
    2. TME execution in MATLAB
    3. Results import back to Python
    4. Integration with existing manifold analysis pipeline
    """
    
    def __init__(self, matlab_tme_path, matlab_executable='matlab'):
        """
        Initialize TME bridge.
        
        Parameters:
        -----------
        matlab_tme_path : str
            Path to TME MATLAB toolbox directory
        matlab_executable : str
            MATLAB executable command (e.g., 'matlab', '/usr/local/bin/matlab')
        """
        self.matlab_tme_path = Path(matlab_tme_path).resolve()  # Use absolute path
        self.matlab_executable = matlab_executable
        self.temp_dir = None
        
        # Verify TME toolbox exists and has required files
        required_files = ['fitMaxEntropy.m', 'sampleTME.m', 'logObjectiveMaxEntropyTensor.m']
        # Verify TME toolbox exists and has required files
        required_files = [
            'fitMaxEntropy.m', 
            'sampleTME.m', 
            'logObjectiveMaxEntropyTensor.m',
            'objectiveMaxEntropyTensor.m'
        ]
        
        # Helper functions that TME depends on
        helper_files = [
            'diagKronSum.m',
            'sumTensor.m', 
            'kron_mvprod.m',
            'minimize.m'
        ]
        
        missing_files = []
        missing_helpers = []
        
        # Check required files
        for file in required_files:
            file_path = self.matlab_tme_path / file
            if not file_path.exists():
                missing_files.append(file)
        
        # Check helper files
        for file in helper_files:
            file_path = self.matlab_tme_path / file
            if not file_path.exists():
                missing_helpers.append(file)
        
        if missing_files or missing_helpers:
            print(f"TME toolbox directory: {self.matlab_tme_path}")
            if missing_files:
                print(f"Missing required files: {missing_files}")
            if missing_helpers:
                print(f"Missing helper files: {missing_helpers}")
            
            # List what files ARE in the directory
            if self.matlab_tme_path.exists():
                actual_files = [f.name for f in self.matlab_tme_path.iterdir() if f.suffix == '.m']
                print(f"Files found in TME directory: {actual_files}")
            else:
                print(f"TME directory does not exist: {self.matlab_tme_path}")
            
            if missing_files:
                raise FileNotFoundError(f"Critical TME files {missing_files} not found in {self.matlab_tme_path}")
            elif missing_helpers:
                print("WARNING: Some helper files are missing, but attempting to continue...")
                print("If TME fails, please ensure you have the complete TME toolbox.")
        
        print(f"TME toolbox verified at: {self.matlab_tme_path}")
    
    def prepare_data_for_tme(self, region_epochs, band_name, subject_id):
        """
        Prepare neural data tensor for TME analysis.
        
        Parameters:
        -----------
        region_epochs : dict
            Nested dictionary: {subject_id: {band_name: epochs}}
        band_name : str
            Frequency band to analyze
        subject_id : int
            Subject ID to process
            
        Returns:
        --------
        data_tensor : np.ndarray
            3D tensor (channels × time × gestures)
        gesture_labels : list
            List of gesture names in order
        """
        if subject_id not in region_epochs:
            raise ValueError(f"Subject {subject_id} not found in data")
        
        if band_name not in region_epochs[subject_id]:
            raise ValueError(f"Band {band_name} not found for subject {subject_id}")
        
        epochs = region_epochs[subject_id][band_name]
        
        # Get gesture names and data
        gesture_labels = list(epochs.event_id.keys())
        n_gestures = len(gesture_labels)
        
        # Extract data for each gesture
        gesture_data = []
        for gesture in gesture_labels:
            gesture_epochs = epochs[gesture]
            data = gesture_epochs.get_data()  # (n_trials, n_channels, n_times)
            
            # Average across trials for this gesture
            gesture_mean = np.mean(data, axis=0)  # (n_channels, n_times)
            gesture_data.append(gesture_mean)
        
        # Stack to create 3D tensor: (n_channels, n_times, n_gestures)
        data_tensor = np.stack(gesture_data, axis=2)
        
        print(f"Prepared data tensor for Subject {subject_id}, {band_name} band:")
        print(f"  Shape: {data_tensor.shape} (channels × time × gestures)")
        print(f"  Gestures: {gesture_labels}")
        
        return data_tensor, gesture_labels
    
    def export_to_matlab(self, data_tensor, output_dir, preserve_dims=(2, 3)):
        """
        Export data tensor and configuration to MATLAB format.
        
        Parameters:
        -----------
        data_tensor : np.ndarray
            3D tensor (channels × time × gestures)
        output_dir : str or Path
            Directory to save MATLAB files
        preserve_dims : tuple
            Which dimensions to preserve covariance (1-indexed for MATLAB)
            Default: (2, 3) preserves time and gesture, randomizes spatial
            
        Returns:
        --------
        matlab_files : dict
            Dictionary with paths to created MATLAB files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data tensor
        data_file = output_dir / 'data_tensor.mat'
        sio.savemat(data_file, {
            'dataTensor': data_tensor,
            'tensor_shape': data_tensor.shape
        })
        
        # Save TME configuration
        config_file = output_dir / 'tme_config.mat'
        sio.savemat(config_file, {
            'preserve_dims': np.array(preserve_dims),
            'tensor_size': len(data_tensor.shape),
            'randomize_spatial': 1 not in preserve_dims  # True if spatial not preserved
        })
        
        matlab_files = {
            'data': data_file,
            'config': config_file,
            'output_dir': output_dir
        }
        
        print(f"Exported data to MATLAB format:")
        print(f"  Data file: {data_file}")
        print(f"  Config file: {config_file}")
        print(f"  Preserving dimensions: {preserve_dims}")
        
        return matlab_files
    
    def create_matlab_script(self, matlab_files, n_surrogates=1000):
        """
        Create MATLAB script to run TME analysis.
        
        Parameters:
        -----------
        matlab_files : dict
            Dictionary with paths to MATLAB data files
        n_surrogates : int
            Number of surrogate tensors to generate
            
        Returns:
        --------
        script_path : Path
            Path to created MATLAB script
        """
        output_dir = matlab_files['output_dir']
        script_path = output_dir / 'run_tme_analysis.m'
        
        # Convert Windows paths to MATLAB-friendly format and ensure absolute path
        def matlab_path(path_obj):
            return str(Path(path_obj).resolve()).replace('\\', '/')
        
        # Get absolute path to TME toolbox
        tme_toolbox_abs_path = matlab_path(self.matlab_tme_path)
        
        matlab_script = f"""
% Auto-generated TME analysis script for neural manifold null hypothesis testing
% Generated by Python TME Bridge

% Add TME toolbox to path (using absolute path)
fprintf('Adding TME toolbox to path: {tme_toolbox_abs_path}\\n');
addpath('{tme_toolbox_abs_path}');

% Verify TME functions are available
if ~exist('fitMaxEntropy', 'file')
    error('fitMaxEntropy function not found. Please check TME toolbox path.');
end

if ~exist('sampleTME', 'file')
    error('sampleTME function not found. Please check TME toolbox path.');
end

fprintf('TME toolbox loaded successfully\\n');

try
    % Load data and configuration
    fprintf('Loading data...\\n');
    load('{matlab_path(matlab_files['data'])}');
    load('{matlab_path(matlab_files['config'])}');
    
    % Display data info
    fprintf('Data tensor shape: [%s]\\n', num2str(size(dataTensor)));
    fprintf('Preserving dimensions: [%s]\\n', num2str(preserve_dims));
    fprintf('Generating %d surrogate tensors...\\n', {n_surrogates});
    
    % Extract features for TME (mean and marginal covariances)
    fprintf('Extracting tensor features...\\n');
    
    % Calculate mean tensor
    meanTensor = dataTensor;
    
    % Calculate marginal covariances
    tensor_dims = size(dataTensor);
    margCov = cell(3, 1);
    
    % Initialize arrays to store preserved covariances
    preserved_covs = [];
    preserved_traces = [];
    preserved_dims = [];
    
    % First pass: compute all covariances that should be preserved
    for dim = 1:3
        if ismember(dim, preserve_dims)
            fprintf('  Computing covariance for dimension %d (preserving)...\\n', dim);
            
            % Reshape tensor to matrix for this dimension's covariance
            other_dims = setdiff(1:3, dim);
            perm_order = [dim, other_dims];
            tensor_perm = permute(dataTensor, perm_order);
            
            % Reshape to (dim_size, other_dims_combined)
            dim_size = tensor_dims(dim);
            matrix_reshaped = reshape(tensor_perm, dim_size, []);
            
            % Compute covariance matrix
            cov_matrix = cov(matrix_reshaped');
            original_trace = trace(cov_matrix);
            
            fprintf('    Covariance matrix size: %dx%d\\n', size(cov_matrix));
            fprintf('    Original trace: %.2f\\n', original_trace);
            
            % Store for trace normalization
            preserved_covs{{end+1}} = cov_matrix;
            preserved_traces(end+1) = original_trace;
            preserved_dims(end+1) = dim;
        else
            fprintf('  Dimension %d: randomizing (setting to empty)...\\n', dim);
            margCov{{dim}} = [];
        end
    end
    
    % Second pass: normalize traces and assign to margCov
    num_preserved = length(preserved_dims);
    fprintf('Number of preserved dimensions: %d\\n', num_preserved);
    
    if num_preserved > 1
        % Multiple preserved dimensions - normalize traces
        target_trace = min(preserved_traces);
        fprintf('Target trace for all preserved covariances: %.2f\\n', target_trace);
        
        for i = 1:num_preserved
            dim = preserved_dims(i);
            cov_matrix = preserved_covs{{i}};
            original_trace = preserved_traces(i);
            
            % Scale to target trace
            normalized_cov = cov_matrix * (target_trace / original_trace);
            margCov{{dim}} = normalized_cov;
            
            fprintf('    Dimension %d: normalized trace from %.2f to %.2f\\n', ...
                    dim, original_trace, trace(normalized_cov));
        end
    elseif num_preserved == 1
        % Only one preserved dimension - no normalization needed
        dim = preserved_dims(1);
        margCov{{dim}} = preserved_covs{{1}};
        fprintf('Only one preserved dimension (dim %d) - no trace normalization needed\\n', dim);
    else
        error('No preserved dimensions found - this should not happen');
    end
    
    % Verify all preserved covariances have the same trace
    fprintf('\\nVerifying final marginal covariances:\\n');
    for dim = 1:3
        if ~isempty(margCov{{dim}})
            fprintf('  Dimension %d: size %dx%d, trace %.6f\\n', ...
                    dim, size(margCov{{dim}}, 1), size(margCov{{dim}}, 2), trace(margCov{{dim}}));
        else
            fprintf('  Dimension %d: empty (randomized)\\n', dim);
        end
    end
    
    % Set up TME parameters
    params.margCov = margCov;
    params.meanTensor = meanTensor;
    
    % Fit maximum entropy distribution
    fprintf('Fitting TME distribution...\\n');
    tic;
    maxEntropy = fitMaxEntropy(params);
    fit_time = toc;
    fprintf('TME fitting completed in %.2f seconds\\n', fit_time);
    
    % Generate surrogate tensors
    fprintf('Generating %d surrogate tensors...\\n', {n_surrogates});
    tic;
    surrTensors = sampleTME(maxEntropy, {n_surrogates});
    sample_time = toc;
    fprintf('Surrogate generation completed in %.2f seconds\\n', sample_time);
    
    % Verify surrogate tensor dimensions
    fprintf('Surrogate tensor dimensions: [%s]\\n', num2str(size(surrTensors)));
    
    % Save results
    fprintf('Saving results...\\n');
    save('{matlab_path(output_dir)}/tme_results.mat', 'maxEntropy', 'surrTensors', 'params', ...
         'fit_time', 'sample_time', '-v7.3');
    
    % Save summary info
    summary.n_surrogates = {n_surrogates};
    summary.tensor_shape = tensor_dims;
    summary.preserve_dims = preserve_dims;
    summary.fit_time = fit_time;
    summary.sample_time = sample_time;
    summary.success = true;
    summary.error_msg = '';
    
    save('{matlab_path(output_dir)}/tme_summary.mat', 'summary');
    
    fprintf('TME analysis completed successfully!\\n');
    fprintf('Results saved to: {matlab_path(output_dir)}/tme_results.mat\\n');
    
catch ME
    fprintf('ERROR in TME analysis: %s\\n', ME.message);
    fprintf('Error occurred in: %s at line %d\\n', ME.stack(1).name, ME.stack(1).line);
    
    % Print more detailed error info
    if length(ME.stack) > 1
        for i = 1:min(3, length(ME.stack))
            fprintf('  Stack %d: %s at line %d\\n', i, ME.stack(i).name, ME.stack(i).line);
        end
    end
    
    % Save error info
    summary.success = false;
    summary.error_msg = ME.message;
    summary.n_surrogates = {n_surrogates};
    save('{matlab_path(output_dir)}/tme_summary.mat', 'summary');
    
    % Don't rethrow - just report and exit
end

fprintf('MATLAB TME script completed.\\n');
"""
        
        # Write script to file
        with open(script_path, 'w') as f:
            f.write(matlab_script)
        
        print(f"Created MATLAB script: {script_path}")
        return script_path
    
    def run_matlab_tme(self, script_path, timeout=3600):
        """
        Execute MATLAB TME script.
        
        Parameters:
        -----------
        script_path : Path
            Path to MATLAB script
        timeout : int
            Timeout in seconds (default: 1 hour)
            
        Returns:
        --------
        success : bool
            Whether execution was successful
        """
        print(f"Executing MATLAB TME analysis...")
        print(f"Script: {script_path}")
        
        # Detect operating system and prepare appropriate MATLAB command
        import platform
        system = platform.system().lower()
        
        if system == 'windows':
            # Windows MATLAB command
            script_name = script_path.stem  # Get filename without extension
            matlab_cmd = [
                self.matlab_executable,
                '-batch', f"cd('{script_path.parent}'); {script_name}; exit;"
            ]
        else:
            # Linux/Mac MATLAB command
            matlab_cmd = [
                self.matlab_executable,
                '-nodisplay',           # No GUI
                '-nosplash',           # No splash screen  
                '-nodesktop',          # No desktop
                '-r', f"cd('{script_path.parent}'); run('{script_path}'); exit;"
            ]
        
        print(f"MATLAB command: {' '.join(matlab_cmd)}")
        
        try:
            # Execute MATLAB
            result = subprocess.run(
                matlab_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=script_path.parent
            )
            
            # Print MATLAB output
            if result.stdout:
                print("MATLAB Output:")
                print(result.stdout)
            
            if result.stderr:
                print("MATLAB Errors:")
                print(result.stderr)
            
            # Check if successful by looking for success indicators in output
            success_indicators = [
                "TME analysis completed successfully",
                "MATLAB TME script completed",
                "Results saved to"
            ]
            
            output_text = result.stdout + result.stderr
            success = any(indicator in output_text for indicator in success_indicators)
            
            if success:
                print("MATLAB TME analysis completed successfully!")
                return True
            else:
                print(f"MATLAB execution may have failed. Return code: {result.returncode}")
                print("Checking for output files...")
                
                # Check if output files were created as fallback
                output_dir = script_path.parent
                summary_file = output_dir / 'tme_summary.mat'
                results_file = output_dir / 'tme_results.mat'
                
                if summary_file.exists() and results_file.exists():
                    print("Output files found - considering successful despite return code")
                    return True
                else:
                    print("No output files found - execution failed")
                    return False
                
        except subprocess.TimeoutExpired:
            print(f"MATLAB execution timed out after {timeout} seconds")
            return False
        except Exception as e:
            print(f"Error executing MATLAB: {e}")
            return False
    
    def import_tme_results(self, output_dir):
        """
        Import TME results from MATLAB back to Python.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory containing TME results
            
        Returns:
        --------
        results : dict
            Dictionary containing surrogate tensors and metadata
        """
        output_dir = Path(output_dir)
        
        # Load summary first (this is regular .mat format)
        summary_file = output_dir / 'tme_summary.mat'
        if not summary_file.exists():
            raise FileNotFoundError(f"TME summary file not found: {summary_file}")
        
        summary = sio.loadmat(summary_file)['summary'][0, 0]
        
        if not summary['success'][0, 0]:
            error_msg = str(summary['error_msg'][0])
            raise RuntimeError(f"TME analysis failed: {error_msg}")
        
        # Load main results (this is v7.3 HDF5 format)
        results_file = output_dir / 'tme_results.mat'
        if not results_file.exists():
            raise FileNotFoundError(f"TME results file not found: {results_file}")
        
        print(f"Loading TME results from: {results_file}")
        
        # Try to load with h5py for v7.3 files
        try:
            import h5py
            
            with h5py.File(results_file, 'r') as f:
                # Load surrogate tensors
                surr_tensors = f['surrTensors'][:]
                
                # Load other data
                fit_time = f['fit_time'][0, 0] if 'fit_time' in f else 0.0
                sample_time = f['sample_time'][0, 0] if 'sample_time' in f else 0.0
                
                print(f"Loaded surrogate tensors using h5py")
                
        except ImportError:
            print("h5py not available, installing...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"])
            
            # Try again after installation
            import h5py
            with h5py.File(results_file, 'r') as f:
                surr_tensors = f['surrTensors'][:]
                fit_time = f['fit_time'][0, 0] if 'fit_time' in f else 0.0
                sample_time = f['sample_time'][0, 0] if 'sample_time' in f else 0.0
                
        except Exception as e:
            print(f"Failed to load with h5py: {e}")
            print("Attempting fallback with scipy.io.loadmat...")
            
            # Fallback to scipy (might fail for large files)
            try:
                matlab_results = sio.loadmat(results_file)
                surr_tensors = matlab_results['surrTensors']
                fit_time = matlab_results.get('fit_time', [[0.0]])[0, 0]
                sample_time = matlab_results.get('sample_time', [[0.0]])[0, 0]
            except Exception as e2:
                raise RuntimeError(f"Failed to load TME results with both h5py and scipy: {e2}")
        
        # Organize results
        results = {
            'surrogate_tensors': surr_tensors,
            'n_surrogates': summary['n_surrogates'][0, 0],
            'tensor_shape': summary['tensor_shape'][0],
            'preserve_dims': summary['preserve_dims'][0],
            'fit_time': summary['fit_time'][0, 0],
            'sample_time': summary['sample_time'][0, 0],
            'success': True
        }
        
        print(f"Successfully imported TME results:")
        print(f"  Surrogate tensors shape: {surr_tensors.shape}")
        print(f"  Number of surrogates: {results['n_surrogates']}")
        print(f"  TME fitting time: {results['fit_time']:.2f} seconds")
        print(f"  Sampling time: {results['sample_time']:.2f} seconds")
        
        return results
    
    def cleanup_temp_files(self, output_dir):
        """Clean up temporary files with Windows compatibility."""
        if output_dir and Path(output_dir).exists():
            import time
            max_attempts = 5
            
            for attempt in range(max_attempts):
                try:
                    shutil.rmtree(output_dir)
                    print(f"Cleaned up temporary files: {output_dir}")
                    break
                except (OSError, PermissionError) as e:
                    if attempt < max_attempts - 1:
                        print(f"Cleanup attempt {attempt + 1} failed, retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        print(f"Warning: Could not clean up temporary files: {output_dir}")
                        print(f"Error: {e}")
                        print("You may need to manually delete this directory later.")

# Convenience function for full TME workflow
def run_tme_analysis(region_epochs, band_name, subject_id, 
                     matlab_tme_path, n_surrogates=1000,
                     preserve_dims=(2, 3), cleanup=True):
    """
    Complete TME analysis workflow: Python → MATLAB → Python
    
    Parameters:
    -----------
    region_epochs : dict
        Nested dictionary: {subject_id: {band_name: epochs}}
    band_name : str
        Frequency band to analyze
    subject_id : int
        Subject ID to process
    matlab_tme_path : str
        Path to TME MATLAB toolbox
    n_surrogates : int
        Number of surrogate tensors to generate
    preserve_dims : tuple
        Which dimensions to preserve (1-indexed for MATLAB)
    cleanup : bool
        Whether to cleanup temporary files
        
    Returns:
    --------
    results : dict
        TME analysis results with surrogate tensors
    """
    # Initialize bridge
    bridge = TMEBridge(matlab_tme_path)
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='tme_analysis_')
        print(f"Using temporary directory: {temp_dir}")
        
        # Step 1: Prepare data
        print("Step 1: Preparing data for TME...")
        data_tensor, gesture_labels = bridge.prepare_data_for_tme(
            region_epochs, band_name, subject_id
        )
        
        # Step 2: Export to MATLAB
        print("Step 2: Exporting data to MATLAB...")
        matlab_files = bridge.export_to_matlab(
            data_tensor, temp_dir, preserve_dims
        )
        
        # Step 3: Create and run MATLAB script
        print("Step 3: Creating MATLAB script...")
        script_path = bridge.create_matlab_script(matlab_files, n_surrogates)
        
        print("Step 4: Running MATLAB TME analysis...")
        success = bridge.run_matlab_tme(script_path)
        
        if not success:
            raise RuntimeError("MATLAB TME analysis failed")
        
        # Wait a moment for MATLAB to fully exit and release file handles
        import time
        time.sleep(2)
        
        # Step 5: Import results
        print("Step 5: Importing results back to Python...")
        results = bridge.import_tme_results(temp_dir)
        
        # Add metadata
        results['original_tensor'] = data_tensor
        results['gesture_labels'] = gesture_labels
        results['subject_id'] = subject_id
        results['band_name'] = band_name
        
        return results
        
    except Exception as e:
        print(f"TME analysis failed: {e}")
        raise
        
    finally:
        # Cleanup
        if cleanup and temp_dir:
            bridge.cleanup_temp_files(temp_dir)

