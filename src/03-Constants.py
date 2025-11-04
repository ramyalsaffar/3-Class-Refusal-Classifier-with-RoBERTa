# Constants
#----------
# This file contains global constants used throughout the project.
# These are fixed values that should not change between experiments.
###############################################################################


# =============================================================================
# CLASS LABELS AND MAPPINGS
# =============================================================================

# Refusal Classifier - Class names for display
#----------------------------------------------
CLASS_NAMES = ['No Refusal', 'Hard Refusal', 'Soft Refusal']

# String to integer mapping
CLASS_MAPPING = {
    'no_refusal': 0,
    'hard_refusal': 1,
    'soft_refusal': 2
}

# Integer to string mapping (reverse)
CLASS_MAPPING_REVERSE = {
    0: 'no_refusal',
    1: 'hard_refusal',
    2: 'soft_refusal'
}

# Jailbreak Detector - Class names for display
#----------------------------------------------
JAILBREAK_CLASS_NAMES = ['Jailbreak Failed', 'Jailbreak Succeeded']

# String to integer mapping
JAILBREAK_MAPPING = {
    'jailbreak_failed': 0,
    'jailbreak_succeeded': 1
}

# Integer to string mapping (reverse)
JAILBREAK_MAPPING_REVERSE = {
    0: 'jailbreak_failed',
    1: 'jailbreak_succeeded'
}


# =============================================================================
# VISUALIZATION COLORS
# =============================================================================

# Refusal Class colors (for plots)
#-----------------------------------
PLOT_COLORS = {
    'no_refusal': '#2ecc71',        # Green
    'hard_refusal': '#e74c3c',      # Red
    'soft_refusal': '#f39c12'       # Orange
}

# Class colors as list (for matplotlib)
PLOT_COLORS_LIST = ['#2ecc71', '#e74c3c', '#f39c12']

# Jailbreak Class colors (for plots)
#------------------------------------
JAILBREAK_PLOT_COLORS = {
    'jailbreak_failed': '#27ae60',          # Dark Green (success - model defended)
    'jailbreak_succeeded': '#c0392b'        # Dark Red (failure - model broken)
}

# Jailbreak colors as list (for matplotlib)
JAILBREAK_PLOT_COLORS_LIST = ['#27ae60', '#c0392b']

# Model colors (for per-model analysis)
#---------------------------------------
# WHY: Matches DATASET_CONFIG['models'] naming convention
MODEL_COLORS = {
    'claude-sonnet-4-20250514': '#3498db',     # Blue
    'gpt-5': '#9b59b6',                         # Purple
    'gemini-2.5-flash': '#e67e22'               # Orange
}


# =============================================================================
# ERROR HANDLING
# =============================================================================

# Error indicator value (used to mark failed evaluations)
ERROR_VALUE = -1

# Error response placeholder
ERROR_RESPONSE = "[ERROR]"


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
