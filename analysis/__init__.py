"""
Analysis module for neural data.
"""

from .time_frequency import (
    compute_time_frequency,
    perform_time_frequency_analysis
)

from .band_power import (
    get_frequency_bands,
    calculate_instantaneous_band_power,
    compute_band_power
)

from .manifold import (
    compute_neural_manifold,
    compute_gesture_manifolds,
    plot_gesture_manifolds,
    analyze_neural_manifolds,
    analyze_gesture_manifolds
)
