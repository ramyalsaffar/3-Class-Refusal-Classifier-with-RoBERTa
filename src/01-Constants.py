# Constants & Environment Configuration
#----------------------------------------
# This file contains:
# 1. Environment detection and system configuration
# 2. Path configuration (local vs AWS)
# 3. Device configuration (CPU/GPU/MPS)
# 4. Python display settings
# 5. Class labels and mappings
# 6. Visualization colors
# 7. Error handling constants
#
# NOTE: This file is loaded BEFORE Config.py so Config can reference these constants.
# All imports are handled in 00-Imports.py before loading this file.
###############################################################################


# =============================================================================
# ENVIRONMENT DETECTION (Smart Defaults)
# =============================================================================

# Detects if running locally or in AWS/Docker
# Defaults to 'local' if ENVIRONMENT variable not set
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
IS_MAC = sys.platform == 'darwin'
IS_AWS = ENVIRONMENT == 'aws'

print(f"🌍 Running in: {ENVIRONMENT.upper()} mode")
if IS_MAC:
    print("🍎 Mac detected - using MPS acceleration if available")


# =============================================================================
# ENVIRONMENT-AWARE PATHS
# =============================================================================

# Sets appropriate paths based on where code is running
if IS_AWS:
    # AWS/Docker paths
    project_path = "/app"
    base_results_path = "/app/results/"
    CodeFilePath = "/app/src/"
else:
    # Local paths (Mac default)
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_results_path = project_path + "/results/"
    CodeFilePath = project_path + "/src/"


# Specific Subdirectories
#------------------------
data_path = project_path + "/data/"
data_raw_path = data_path + "raw/"
data_responses_path = data_path + "responses/"
data_processed_path = data_path + "processed/"
data_splits_path = data_path + "splits/"
data_checkpoints_path = data_processed_path + "checkpoints/"  # New: for error recovery
models_path = project_path + "/models/"
results_path = base_results_path
visualizations_path = project_path + "/visualizations/"
reports_path = project_path + "/reports/"


# Create directories if they don't exist
#----------------------------------------
for path in [base_results_path, data_path, data_raw_path, data_responses_path,
             data_processed_path, data_splits_path, data_checkpoints_path,
             models_path, results_path, visualizations_path, reports_path]:
    os.makedirs(path, exist_ok=True)


# =============================================================================
# PYTHON DISPLAY SETTINGS
# =============================================================================

# Pandas display options for better output formatting
pd.set_option('display.max_rows', 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", 250)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.4f}'.format


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

# Auto-detect best available device for PyTorch
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"🚀 CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
elif IS_MAC and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🚀 MPS available - using Apple Silicon GPU")
else:
    DEVICE = torch.device('cpu')
    print("⚠️  No GPU available - using CPU (training will be slow)")


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
MODEL_COLORS = {
    'claude-sonnet-4.5': '#3498db',     # Blue
    'gpt-5': '#9b59b6',                  # Purple
    'gemini-2.5-flash': '#e67e22'        # Orange
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
3-Class Refusal Classifier with RoBERTa
Created on October 28, 2025
@author: ramyalsaffar
"""
