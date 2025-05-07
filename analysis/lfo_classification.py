"""
Functions for classifying gestures based on mean LFO neural data using Support Vector Machines (SVM).
Includes both pairwise and multi-class classification approaches.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import itertools
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap

# helper function to create a safe filename
def create_safe_filename(subject_id=None, region_label=None, prefix="output"):
    """
    Create a safe, short filename that won't exceed Windows path limits.
    
    Parameters:
    -----------
    subject_id : str or int, optional
        Subject ID
    region_label : str or list, optional
        Region label(s)
    prefix : str, optional
        Prefix for the filename
    
    Returns:
    --------
    str
        A safe filename that won't exceed path limits
    """
    filename = prefix
    
    # Add subject ID if provided
    if subject_id is not None:
        filename += f"_sub-{subject_id}"
    
    # Handle region label - very simple approach
    if region_label is not None:
        # Check if this is a multi-region case
        is_multi_region = False
        if isinstance(region_label, list) and len(region_label) > 1:
            is_multi_region = True
        elif isinstance(region_label, str) and ',' in region_label:
            is_multi_region = True
            
        if is_multi_region:
            filename += "_brain-wide"
        else:
            # For single region, use a very short identifier
            if isinstance(region_label, list) and len(region_label) > 0:
                region_str = region_label[0]
            else:
                region_str = str(region_label).split(',')[0].strip()
            
            # Extract just the core region name without prefixes
            core_name = region_str.split('-')[-1] if '-' in region_str else region_str
            filename += f"_{core_name[:10]}"  # Limit to 10 chars max
    
    return filename + ".png"

def prepare_data_for_classification(trial_data, n_components=None):
    """
    Prepare trial data for classification by extracting features and labels.
    
    Parameters:
    -----------
    trial_data : dict
        Dictionary mapping gesture names to arrays of trial data (trials x channels)
    n_components : int, optional
        Number of PCA components to use, or None to use all features
    
    Returns:
    --------
    X : np.ndarray
        Feature matrix (trials x features)
    y : np.ndarray
        Labels array (trials)
    gesture_labels : list
        List of unique gesture names
    """
    # Collect all trials and labels
    all_trials = []
    all_labels = []
    
    # Process each gesture
    for gesture, trials in trial_data.items():
        if len(trials) > 0:
            all_trials.append(trials)
            all_labels.extend([gesture] * len(trials))
    
    # Concatenate trials along the first dimension
    if all_trials:
        X = np.vstack(all_trials)
        y = np.array(all_labels)
    else:
        X = np.array([])
        y = np.array([])
    
    # Get unique gestures
    gesture_labels = sorted(set(all_labels))
    
    return X, y, gesture_labels

def run_pairwise_classification(X, y, n_folds=5, n_permutations=100, use_pca=False, 
                              pca_components=0.95, n_jobs=-1, random_state=42):
    """
    Run pairwise SVM classification for all pairs of gestures.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (trials x features)
    y : np.ndarray
        Labels array (trials)
    n_folds : int, optional
        Number of cross-validation folds
    n_permutations : int, optional
        Number of permutations for statistical testing, or 0 to skip
    use_pca : bool, optional
        Whether to apply PCA before classification
    pca_components : float or int, optional
        Number of PCA components to use or variance to explain
    n_jobs : int, optional
        Number of parallel jobs to run
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing classification results
    """
    # Get unique gesture labels
    gesture_labels = sorted(set(y))
    n_gestures = len(gesture_labels)
    
    # Initialize results dictionary
    results = {
        'accuracy_matrix': np.zeros((n_gestures, n_gestures)),
        'p_value_matrix': np.zeros((n_gestures, n_gestures)),
        'confusion_matrices': {},
        'gesture_labels': gesture_labels
    }
    
    # Skip if we don't have enough data
    if len(X) < 10 or n_gestures < 2:
        print("Not enough data for classification analysis")
        return results
    
    print(f"Running pairwise SVM classification for {n_gestures} gestures...")
    
    # Create all pairs of gestures
    gesture_pairs = list(itertools.combinations(range(n_gestures), 2))
    
    # Define a function to run classification for a single pair
    def classify_pair(pair_idx):
        i, j = pair_idx
        g1, g2 = gesture_labels[i], gesture_labels[j]
        
        # Select data for this pair
        mask = np.isin(y, [g1, g2])
        X_pair = X[mask]
        y_pair = y[mask]
        
        # Convert labels to binary
        y_binary = np.array([0 if label == g1 else 1 for label in y_pair])
        
        # Get number of trials per class
        n_g1 = np.sum(y_pair == g1)
        n_g2 = np.sum(y_pair == g2)
        
        # Skip if we don't have enough data
        if n_g1 < 3 or n_g2 < 3:
            print(f"Skipping {g1} vs {g2}: not enough data ({n_g1} vs {n_g2})")
            return i, j, {
                'accuracy': np.nan,
                'p_value': np.nan,
                'confusion_matrix': None,
                'cross_val_preds': None
            }
        
        # Create a pipeline with standardization and optional PCA
        steps = [('scaler', StandardScaler())]
        
        if use_pca:
            # For high-dimensional data, reduce dimensions
            if X_pair.shape[1] > 50:
                steps.append(('pca', PCA(n_components=pca_components)))
        
        # Add the SVM classifier
        steps.append(('svm', SVC(kernel='linear', random_state=random_state)))
        
        # Create pipeline
        pipeline = Pipeline(steps)
        
        # Set up cross-validation
        if min(n_g1, n_g2) < n_folds:
            # Use LOOCV if we have few samples
            cv = LeaveOneOut()
            print(f"Using LOOCV for {g1} vs {g2} due to small sample size")
        else:
            # Use stratified K-fold otherwise
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Run cross-validation
        preds = cross_val_predict(pipeline, X_pair, y_binary, cv=cv)
        accuracy = accuracy_score(y_binary, preds)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_binary, preds)
        
        # Run permutation test if requested
        p_value = np.nan
        if n_permutations > 0:
            # Compute null distribution of accuracies
            null_accs = []
            for _ in range(n_permutations):
                # Shuffle labels
                perm_y = np.random.permutation(y_binary)
                # Run cross-validation with shuffled labels
                perm_preds = cross_val_predict(pipeline, X_pair, perm_y, cv=cv)
                # Compute accuracy
                perm_acc = accuracy_score(perm_y, perm_preds)
                null_accs.append(perm_acc)
            
            # Compute p-value (proportion of permutation accuracies >= true accuracy)
            p_value = np.mean(np.array(null_accs) >= accuracy)
        
        # Return results for this pair
        return i, j, {
            'accuracy': accuracy,
            'p_value': p_value,
            'confusion_matrix': cm,
            'cross_val_preds': preds
        }
    
    # Run classification for all pairs in parallel
    pair_results = Parallel(n_jobs=n_jobs)(
        delayed(classify_pair)(pair) for pair in gesture_pairs
    )
    
    # Process pair results
    for i, j, res in pair_results:
        if not np.isnan(res['accuracy']):
            results['accuracy_matrix'][i, j] = res['accuracy']
            results['accuracy_matrix'][j, i] = res['accuracy']
            results['p_value_matrix'][i, j] = res['p_value']
            results['p_value_matrix'][j, i] = res['p_value']
            results['confusion_matrices'][(gesture_labels[i], gesture_labels[j])] = res['confusion_matrix']
    
    # Set diagonal to NaN for better visualization
    np.fill_diagonal(results['accuracy_matrix'], np.nan)
    np.fill_diagonal(results['p_value_matrix'], np.nan)
    
    return results

def run_multiclass_classification(X, y, n_folds=5, n_permutations=100, use_pca=False, 
                                pca_components=0.95, random_state=42):
    """
    Run multi-class SVM classification for all gestures.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (trials x features)
    y : np.ndarray
        Labels array (trials)
    n_folds : int, optional
        Number of cross-validation folds
    n_permutations : int, optional
        Number of permutations for statistical testing, or 0 to skip
    use_pca : bool, optional
        Whether to apply PCA before classification
    pca_components : float or int, optional
        Number of PCA components to use or variance to explain
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing classification results
    """
    # Get unique gesture labels
    gesture_labels = sorted(set(y))
    n_gestures = len(gesture_labels)
    n_samples = len(y)
    
    # Initialize results dictionary
    results = {
        'accuracy': np.nan,
        'p_value': np.nan,
        'confusion_matrix': None,
        'classification_report': None,
        'gesture_labels': gesture_labels,
        'cross_val_preds': None
    }
    
    # Skip if we don't have enough data
    if n_samples < 10 or n_gestures < 2:
        print("Not enough data for multi-class classification analysis")
        return results
    
    # Check if we have enough samples per class
    samples_per_class = {label: np.sum(y == label) for label in gesture_labels}
    min_samples = min(samples_per_class.values())
    
    if min_samples < 3:
        print(f"Not enough samples for some classes (minimum: {min_samples})")
        return results
    
    print(f"Running multi-class SVM classification for {n_gestures} gestures...")
    
    # Create a pipeline with standardization and optional PCA
    steps = [('scaler', StandardScaler())]
    
    if use_pca:
        # For high-dimensional data, reduce dimensions
        if X.shape[1] > 50:
            steps.append(('pca', PCA(n_components=pca_components)))
    
    # Add the SVM classifier
    steps.append(('svm', SVC(kernel='linear', random_state=random_state, decision_function_shape='ovo')))
    
    # Create pipeline
    pipeline = Pipeline(steps)
    
    # Set up cross-validation
    if min_samples < n_folds:
        # Use LOOCV if we have few samples
        cv = LeaveOneOut()
        print(f"Using LOOCV due to small sample size (minimum: {min_samples})")
    else:
        # Use stratified K-fold otherwise
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Run cross-validation
    cross_val_preds = cross_val_predict(pipeline, X, y, cv=cv)
    accuracy = accuracy_score(y, cross_val_preds)
    
    # Compute confusion matrix
    cm = confusion_matrix(y, cross_val_preds, labels=gesture_labels)
    
    # Generate classification report
    report = classification_report(y, cross_val_preds, labels=gesture_labels, output_dict=True)
    
    # Run permutation test if requested
    p_value = np.nan
    if n_permutations > 0:
        # Compute null distribution of accuracies
        null_accs = []
        for _ in range(n_permutations):
            # Shuffle labels
            perm_y = np.random.permutation(y)
            # Run cross-validation with shuffled labels
            perm_preds = cross_val_predict(pipeline, X, perm_y, cv=cv)
            # Compute accuracy
            perm_acc = accuracy_score(perm_y, perm_preds)
            null_accs.append(perm_acc)
        
        # Compute p-value (proportion of permutation accuracies >= true accuracy)
        p_value = np.mean(np.array(null_accs) >= accuracy)
    
    # Store results
    results['accuracy'] = accuracy
    results['p_value'] = p_value
    results['confusion_matrix'] = cm
    results['classification_report'] = report
    results['cross_val_preds'] = cross_val_preds
    
    return results

def visualize_pairwise_classification(results, subject_id=None, region_label=None, output_dir=None):
    """
    Visualize pairwise classification results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing pairwise classification results
    subject_id : str or int, optional
        Subject ID for the plot title
    region_label : str, optional
        Brain region label for the plot title
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    figs : dict
        Dictionary of figure objects
    """
    # Initialize dictionary to store figures
    figs = {}
    
    # Extract data from results
    accuracy_matrix = results['accuracy_matrix']
    p_value_matrix = results['p_value_matrix']
    gesture_labels = results['gesture_labels']
    
    # Skip if we don't have enough data
    if np.all(np.isnan(accuracy_matrix)):
        print("No valid pairwise classification results to visualize")
        return figs
    
    # 1. Create accuracy heatmap
    fig_acc, ax_acc = plt.subplots(figsize=(10, 8))
    
    # Create a custom colormap: red -> orange -> yellow for increasing accuracy
    # Similar to the one used in visualize_distance_matrix()
    colors = [(0.7, 0, 0),    # dark red (low accuracy)
              (1, 0, 0),      # bright red
              (1, 0.5, 0),    # orange
              (1, 1, 0)]      # yellow (highest accuracy)
    
    custom_cmap = LinearSegmentedColormap.from_list('red_to_yellow', colors)
    custom_cmap.set_bad(color='black')  # Set NaN values (diagonal) to black
    
    # Plot heatmap with the custom colormap
    sns.heatmap(accuracy_matrix, annot=True, fmt=".2f", cmap=custom_cmap,
                xticklabels=gesture_labels, yticklabels=gesture_labels,
                vmin=0.5, vmax=1, ax=ax_acc)
    
    # Add significance markers
    if not np.all(np.isnan(p_value_matrix)):
        # Add asterisks for significant results
        for i in range(len(gesture_labels)):
            for j in range(len(gesture_labels)):
                if i != j and not np.isnan(accuracy_matrix[i, j]):
                    p = p_value_matrix[i, j]
                    if not np.isnan(p):
                        if p < 0.001:
                            ax_acc.text(j + 0.5, i + 0.85, '***', ha='center', va='center', color='white', fontsize=12)
                        elif p < 0.01:
                            ax_acc.text(j + 0.5, i + 0.85, '**', ha='center', va='center', color='white', fontsize=12)
                        elif p < 0.05:
                            ax_acc.text(j + 0.5, i + 0.85, '*', ha='center', va='center', color='white', fontsize=12)
    
    # Set title
    title = "Pairwise SVM Classification Accuracy"
    if subject_id is not None and region_label is not None:
        title += f"\nSubject {subject_id}, Region: {region_label}"
    elif subject_id is not None:
        title += f"\nSubject {subject_id}"
    elif region_label is not None:
        title += f"\nRegion: {region_label}"
    
    ax_acc.set_title(title)
    
    # Set labels
    ax_acc.set_xlabel("Gesture")
    ax_acc.set_ylabel("Gesture")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir is not None:
        # Create a very short, safe filename
        output_file = create_safe_filename(
            subject_id=subject_id,
            region_label=region_label,
            prefix="pairwise_accuracy"
        )
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with the safe filename
        plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches='tight')
    
    # Store figure
    figs['accuracy_heatmap'] = fig_acc
    
    # 2. Create bar plot of accuracies
    # Extract upper triangle of accuracy matrix (ignoring diagonal)
    accuracies = []
    pair_labels = []
    p_values = []
    
    for i in range(len(gesture_labels)):
        for j in range(i+1, len(gesture_labels)):
            if not np.isnan(accuracy_matrix[i, j]):
                accuracies.append(accuracy_matrix[i, j])
                pair_labels.append(f"{gesture_labels[i]} vs {gesture_labels[j]}")
                p_values.append(p_value_matrix[i, j])
    
    # Sort by accuracy
    sorted_idx = np.argsort(accuracies)[::-1]  # Descending order
    sorted_accuracies = np.array(accuracies)[sorted_idx]
    sorted_pair_labels = np.array(pair_labels)[sorted_idx]
    sorted_p_values = np.array(p_values)[sorted_idx]
    
    # Create bar plot
    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
    
    # Plot bars
    bars = ax_bar.bar(np.arange(len(sorted_accuracies)), sorted_accuracies, color='skyblue')
    
    # Add chance level line
    ax_bar.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
    
    # Add error bars (if we had them)
    
    # Add significance markers
    for i, p in enumerate(sorted_p_values):
        if not np.isnan(p):
            if p < 0.001:
                ax_bar.text(i, sorted_accuracies[i] + 0.02, '***', ha='center', va='bottom', fontsize=12)
            elif p < 0.01:
                ax_bar.text(i, sorted_accuracies[i] + 0.02, '**', ha='center', va='bottom', fontsize=12)
            elif p < 0.05:
                ax_bar.text(i, sorted_accuracies[i] + 0.02, '*', ha='center', va='bottom', fontsize=12)
    
    # Set labels and title
    ax_bar.set_xlabel('Gesture pair')
    ax_bar.set_ylabel('Classification accuracy')
    title = "Pairwise SVM Classification Accuracy"
    if subject_id is not None and region_label is not None:
        title += f"\nSubject {subject_id}, Region: {region_label}"
    elif subject_id is not None:
        title += f"\nSubject {subject_id}"
    elif region_label is not None:
        title += f"\nRegion: {region_label}"
    
    ax_bar.set_title(title)
    
    # Set x-tick labels
    ax_bar.set_xticks(np.arange(len(sorted_accuracies)))
    if len(sorted_pair_labels) > 8:
        # Rotate labels if we have many
        ax_bar.set_xticklabels(sorted_pair_labels, rotation=45, ha='right')
    else:
        ax_bar.set_xticklabels(sorted_pair_labels)
    
    # Set y-axis limits
    ax_bar.set_ylim([0, 1.1])
    
    # Add legend
    ax_bar.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir is not None:
        # Create a very short, safe filename
        output_file = create_safe_filename(
            subject_id=subject_id,
            region_label=region_label,
            prefix="pairwise_barplot"
        )
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with the safe filename
        plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches='tight')
    
    # Store figure
    figs['accuracy_barplot'] = fig_bar
    
    return figs

def visualize_multiclass_classification(results, subject_id=None, region_label=None, output_dir=None):
    """
    Visualize multi-class classification results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing multi-class classification results
    subject_id : str or int, optional
        Subject ID for the plot title
    region_label : str, optional
        Brain region label for the plot title
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    figs : dict
        Dictionary of figure objects
    """
    # Initialize dictionary to store figures
    figs = {}
    
    # Extract data from results
    accuracy = results['accuracy']
    p_value = results['p_value']
    confusion_matrix = results['confusion_matrix']
    gesture_labels = results['gesture_labels']
    classification_report = results['classification_report']
    
    # Skip if we don't have valid results
    if np.isnan(accuracy) or confusion_matrix is None:
        print("No valid multi-class classification results to visualize")
        return figs
    
    # 1. Create confusion matrix heatmap
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    sns.heatmap(cm_norm, annot=confusion_matrix, fmt="d", cmap="Blues",
                xticklabels=gesture_labels, yticklabels=gesture_labels, ax=ax_cm)
    
    # Set title
    title = f"Multi-class SVM Classification Confusion Matrix\nAccuracy: {accuracy:.2f}"
    if not np.isnan(p_value):
        title += f", p-value: {p_value:.3f}"
        if p_value < 0.05:
            title += " (*)"
        if p_value < 0.01:
            title += "*"
        if p_value < 0.001:
            title += "*"
    
    if subject_id is not None and region_label is not None:
        title += f"\nSubject {subject_id}, Region: {region_label}"
    elif subject_id is not None:
        title += f"\nSubject {subject_id}"
    elif region_label is not None:
        title += f"\nRegion: {region_label}"
    
    ax_cm.set_title(title)
    
    # Set labels
    ax_cm.set_xlabel("Predicted gesture")
    ax_cm.set_ylabel("True gesture")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir is not None:
        # Create a very short, safe filename
        output_file = create_safe_filename(
            subject_id=subject_id,
            region_label=region_label,
            prefix="multiclass_confusion"
        )
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with the safe filename
        plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches='tight')
    
    # Store figure
    figs['confusion_matrix'] = fig_cm
    
    # 2. Create per-class performance bar plot
    if classification_report is not None:
        # Extract class-specific metrics
        class_metrics = {label: classification_report[label] for label in gesture_labels if label in classification_report}
        
        # Extract precision, recall, and F1-score
        precisions = []
        recalls = []
        f1_scores = []
        supports = []
        
        for label in gesture_labels:
            if label in class_metrics:
                precisions.append(class_metrics[label]['precision'])
                recalls.append(class_metrics[label]['recall'])
                f1_scores.append(class_metrics[label]['f1-score'])
                supports.append(class_metrics[label]['support'])
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
                supports.append(0)
        
        # Create bar plot
        fig_metrics, ax_metrics = plt.subplots(figsize=(12, 6))
        
        # Set width of bars
        bar_width = 0.25
        
        # Set positions of bars on X axis
        r1 = np.arange(len(gesture_labels))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Plot bars
        ax_metrics.bar(r1, precisions, width=bar_width, label='Precision', color='#5DA5DA')
        ax_metrics.bar(r2, recalls, width=bar_width, label='Recall', color='#FAA43A')
        ax_metrics.bar(r3, f1_scores, width=bar_width, label='F1-score', color='#60BD68')
        
        # Add labels and title
        ax_metrics.set_xlabel('Gesture')
        ax_metrics.set_ylabel('Score')
        title = "Per-class Classification Performance"
        if subject_id is not None and region_label is not None:
            title += f"\nSubject {subject_id}, Region: {region_label}"
        elif subject_id is not None:
            title += f"\nSubject {subject_id}"
        elif region_label is not None:
            title += f"\nRegion: {region_label}"
        
        ax_metrics.set_title(title)
        
        # Set x-tick labels
        ax_metrics.set_xticks([r + bar_width for r in range(len(gesture_labels))])
        ax_metrics.set_xticklabels(gesture_labels)
        
        # Add legend
        ax_metrics.legend()
        
        # Set y-axis limits
        ax_metrics.set_ylim([0, 1.1])
        
        # Add number of samples as text
        for i, support in enumerate(supports):
            ax_metrics.annotate(f"n={support}", 
                              xy=(r2[i], 0.05), 
                              ha='center', 
                              va='bottom',
                              rotation=90)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if output directory is provided
        if output_dir is not None:
            # Create a very short, safe filename
            output_file = create_safe_filename(
                subject_id=subject_id,
                region_label=region_label,
                prefix="multiclass_metrics"
            )
            
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save with the safe filename
            plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches='tight')
        
        # Store figure
        figs['class_metrics'] = fig_metrics
    
    return figs

def analyze_cross_region_classification(classification_results_by_region, output_dir=None):
    """
    Analyze pairwise classification performance across different brain regions
    and create a box plot visualization.
    
    Parameters:
    -----------
    classification_results_by_region : dict
        Dictionary mapping region names to dictionaries of classification results by subject
    output_dir : str, optional
        Directory to save visualizations
    
    Returns:
    --------
    region_comparison : dict
        Dictionary containing cross-region pairwise classification results
    """
    # Initialize simplified results dictionary - only keep what's needed for box plot
    region_comparison = {
        'pairwise': {
            'pair_accuracies_by_region': {}
        }
    }
    
    # Process each region
    for region_name, region_results in classification_results_by_region.items():
        # Skip if no results for this region
        if not region_results:
            print(f"No classification results for {region_name}")
            continue
            
        print(f"Collecting pairwise classification results for {region_name}...")
        
        # Process pairwise results only
        pair_accuracies = {}
        
        for subject_id, subject_results in region_results.items():
            if 'pairwise' in subject_results and subject_results['pairwise'] is not None:
                acc_matrix = subject_results['pairwise']['accuracy_matrix']
                gesture_labels = subject_results['pairwise']['gesture_labels']
                
                # Skip if no valid accuracy matrix
                if np.all(np.isnan(acc_matrix)):
                    continue
                
                # Extract pairwise accuracies
                for i in range(len(gesture_labels)):
                    for j in range(i+1, len(gesture_labels)):
                        if not np.isnan(acc_matrix[i, j]):
                            pair_key = f"{gesture_labels[i]}_vs_{gesture_labels[j]}"
                            
                            if pair_key not in pair_accuracies:
                                pair_accuracies[pair_key] = []
                                
                            pair_accuracies[pair_key].append(acc_matrix[i, j])
        
        # Store pairwise results for this region if we have any
        if pair_accuracies:
            region_comparison['pairwise']['pair_accuracies_by_region'][region_name] = pair_accuracies
    
    # Create box plot visualization if we have data and an output directory
    if output_dir is not None and region_comparison['pairwise']['pair_accuracies_by_region']:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create box plot
        visualize_region_boxplot_comparison(region_comparison, output_dir)
        
        # Print summary info
        print("\nRegion Pairwise Classification Summary:")
        for region, pair_accuracies in region_comparison['pairwise']['pair_accuracies_by_region'].items():
            # Flatten all accuracies for this region
            all_accs = [acc for accs in pair_accuracies.values() for acc in accs]
            if all_accs:
                print(f"  {region}: {len(pair_accuracies)} pairs, {len(all_accs)} samples, " +
                      f"median acc: {np.median(all_accs):.2f}")
    
    return region_comparison

def visualize_region_boxplot_comparison(region_comparison, output_dir):
    """
    Create a box plot comparing the distribution of pairwise classification 
    accuracies across different brain regions using Seaborn.
    
    Parameters:
    -----------
    region_comparison : dict
        Dictionary containing cross-region comparison results
    output_dir : str
        Directory to save visualizations
    """
    # Check if we have pairwise accuracy data
    if not region_comparison['pairwise']['pair_accuracies_by_region']:
        print("No pairwise accuracy data available for box plot visualization")
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Prepare data for box plot - convert to format suitable for seaborn
    all_accuracies = []
    
    # Sort regions by median accuracy (highest to lowest)
    region_medians = {}
    
    for region, pair_accuracies in region_comparison['pairwise']['pair_accuracies_by_region'].items():
        # Flatten all accuracies for this region into a single list
        region_accuracies = []
        for pair, accs in pair_accuracies.items():
            region_accuracies.extend(accs)
        
        if region_accuracies:
            region_medians[region] = np.median(region_accuracies)
            # Store each accuracy with its region for seaborn
            for acc in region_accuracies:
                all_accuracies.append({'Region': region, 'Accuracy': acc})
    
    # Sort regions by median accuracy
    sorted_regions = sorted(region_medians.keys(), key=lambda r: region_medians[r], reverse=True)
    
    # Convert to DataFrame for Seaborn
    import pandas as pd
    df = pd.DataFrame(all_accuracies)
    
    # Create custom order for regions
    if sorted_regions:
        df['Region'] = pd.Categorical(df['Region'], categories=sorted_regions, ordered=True)
    
    # Create box plot using seaborn
    ax = sns.boxplot(
        x='Region', 
        y='Accuracy', 
        data=df,
        palette='crest',  # Use crest color palette
        width=0.6,
        notch=False,
        showfliers=True,
        boxprops={'alpha': 0.8, 'edgecolor': 'black', 'linewidth': 1.5},
        whiskerprops={'color': 'black', 'linewidth': 1.5},
        capprops={'color': 'black', 'linewidth': 1.5},
        medianprops={'color': 'darkred', 'linewidth': 2},
        flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 4, 'alpha': 0.6}
    )
    
    # Add chance level line (0.5 for binary classification)
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Chance level')
    
    # Set labels and title
    plt.xlabel('Brain Region', fontsize=12)
    plt.ylabel('Pairwise Classification Accuracy', fontsize=12)
    plt.title('Distribution of Pairwise Classification Accuracies Across Brain Regions', fontsize=14)
    
    # Set x-tick labels (rotate if necessary)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Set y-axis limits with some padding
    plt.ylim([0.45, 1.05])
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Add grid lines for easier reading of values
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text annotations with sample sizes
    for i, region in enumerate(sorted_regions):
        pair_accuracies = region_comparison['pairwise']['pair_accuracies_by_region'][region]
        total_samples = sum(len(accs) for accs in pair_accuracies.values())
        total_pairs = len(pair_accuracies)
        
        plt.annotate(f'n={total_samples}\n({total_pairs} pairs)', 
                    xy=(i, 0.47), 
                    ha='center', 
                    va='top',
                    fontsize=9)
    
    # Tight layout to ensure everything fits
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'region_pairwise_boxplot.png'), dpi=300, bbox_inches='tight')
    
    # Close the figure to free up memory
    plt.close()
    
    print(f"Box plot visualization saved to {os.path.join(output_dir, 'region_pairwise_boxplot.png')}")
    
