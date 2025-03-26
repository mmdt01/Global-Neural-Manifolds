# Neural Analysis Toolkit

A modular Python toolkit for analyzing region-specific neural data from intracranial recordings.

## Project Structure

```
neural_analysis/
│
├── data_loading/
│   ├── __init__.py        # Subject data loading functions
│   ├── data_loader.py     # Raw data loading functions
│   └── mne_converter.py   # MNE object conversion functions
│
├── region_processing/
│   ├── __init__.py          # Region-specific data analysis
│   ├── region_extractor.py  # Brain region identification functions
│   └── epochs_extractor.py  # Region-specific epoch extraction
│
├── analysis/
│   ├── __init__.py         # Analysis module initialization
│   ├── time_frequency.py   # Time-frequency analysis functions
│   ├── band_power.py       # Band power analysis functions
│   └── manifold.py         # Neural manifold analysis functions
│
├── visualization/
│   ├── __init__.py         # Visualization module initialization
│   ├── tf_plots.py         # Time-frequency plotting functions
│   └── manifold_plots.py   # Neural manifold plotting functions
│
├── utils/
│   ├── __init__.py         # Utility module initialization
│   └── helpers.py          # Helper utility functions
│
└── main.py                 # Main program entry point
```

## Requirements

- Python 3.8+
- MNE-Python
- NumPy
- SciPy
- hdf5storage
- Matplotlib
- Seaborn
- scikit-learn

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the main analysis script:
   ```
   python main.py --subjects 2 3 4 --regions ctx-lh-precentral --analysis all --output-dir results
   ```

## Usage

```
usage: main.py [-h] [--subjects SUBJECTS [SUBJECTS ...]] [--regions REGIONS [REGIONS ...]]
              [--trigger {stim,emg}] [--tmin TMIN] [--tmax TMAX]
              [--analysis {tf,manifold,gesture,all}] [--output-dir OUTPUT_DIR] [--plot]

Neural data analysis for region-specific epochs.

optional arguments:
  -h, --help            show this help message and exit
  --subjects SUBJECTS [SUBJECTS ...]
                        List of subject IDs to analyze
  --regions REGIONS [REGIONS ...]
                        List of brain region labels to extract data for
  --trigger {stim,emg}  Type of trigger to use (stim or emg)
  --tmin TMIN           Start time for epochs in seconds
  --tmax TMAX           End time for epochs in seconds
  --analysis {tf,manifold,gesture,all}
                        Type of analysis to run
  --output-dir OUTPUT_DIR
                        Directory to save output files
  --plot                Plot results interactively
```

## Analysis Types

The toolkit provides three main types of neural data analysis:

1. **Time-Frequency Analysis**: Compute and visualize time-frequency representations of neural activity using Morlet wavelets.

2. **Neural Manifold Analysis**: Compute low-dimensional neural manifold representations using PCA on band-filtered data.

3. **Gesture-Specific Manifold Analysis**: Analyze neural manifolds for different gesture types.

## Data Structure

Input data should be organized as follows:

```
preprocessed/
└── P{subject_id}/
    └── preprocessed2.mat
    
EleCTX_Files/
└── P{subject_id}/
    ├── electrodes_Final_Norm.mat
    └── SignalChanel_Electrode_Registration.mat
```

## Output

Analysis results are saved to the specified output directory:

```
output/
├── tf_plots/             # Time-frequency plots
├── manifold_plots/       # Neural manifold plots 
└── gesture_manifold_plots/ # Gesture-specific manifold plots
```
