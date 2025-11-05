# Constants
#----------
# This file contains environment detection, paths, device configuration, and class labels.
# This must be loaded FIRST before Config.py since Config references DEVICE and paths.
###############################################################################

import os
import sys
import glob
import torch


#------------------------------------------------------------------------------


# Environment Detection (Smart Defaults)
#----------------------------------------
# Detects if running locally or in AWS/Docker
# Defaults to 'local' if ENVIRONMENT variable not set
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
IS_MAC = sys.platform == 'darwin'
IS_AWS = ENVIRONMENT == 'aws'

print(f"🌍 Running in: {ENVIRONMENT.upper()} mode")
if IS_MAC:
    print("🍎 Mac detected - using MPS acceleration if available")


#------------------------------------------------------------------------------


# Environment-Aware Paths
#------------------------
# Sets appropriate paths based on where code is running
if IS_AWS:
    # AWS/Docker paths
    project_path = "/app"
    base_results_path = "/app/results/"
    CodeFilePath = "/app/src/"
else:
    # Local paths (Mac default)
    main_path = "/Users/ramyalsaffar/Ramy/C.V..V/1-Resume/06- LLM Model Behavior Projects/"
    folder = "3-Class Refusal Classifier with RoBERTa"
    project_path = glob.glob(main_path + "*" + folder)[0]
    base_results_path = glob.glob(project_path + "/*Code/*Results")[0]
    CodeFilePath = project_path + "/src/"


# Specific Subdirectories
#------------------------
data_path = glob.glob(base_results_path + "/*Data/")[0]
data_raw_path = glob.glob(data_path + "*Raw/")[0]
data_responses_path = glob.glob(data_path + "*Responses/")[0]
data_processed_path = glob.glob(data_path + "*Processed/")[0]
data_splits_path = glob.glob(data_path + "*Splits/")[0]
data_checkpoints_path = data_processed_path + "checkpoints/"  # NEW - Phase 2

models_path = glob.glob(base_results_path + "/*Models/")[0]
results_path = base_results_path

visualizations_path = glob.glob(base_results_path + "/*Visualizations/")[0]
reports_path = glob.glob(base_results_path + "/*Reports/")[0]

# API Keys (local file storage)
api_keys_file_path = glob.glob(project_path + "/*API Keys/API Keys.txt")[0]

# Create checkpoint directory if it doesn't exist
os.makedirs(data_checkpoints_path, exist_ok=True)


#------------------------------------------------------------------------------


# Device Configuration
#---------------------
# Auto-detect best available device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"🚀 CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
elif IS_MAC and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🚀 MPS available - using Apple Silicon GPU")
else:
    DEVICE = torch.device('cpu')
    print("⚠️  No GPU available - using CPU (training will be slow)")


#------------------------------------------------------------------------------


# =============================================================================
# CLASS LABELS AND MAPPINGS
# =============================================================================

# Refusal Classifier - Class names for display
#----------------------------------------------
CLASS_NAMES = ['No Refusal', 'Hard Refusal', 'Soft Refusal']

# Jailbreak Detector - Class names for display
#----------------------------------------------
JAILBREAK_CLASS_NAMES = ['Jailbreak Failed', 'Jailbreak Succeeded']


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
# WHY: Matches API_CONFIG['response_models'] naming convention
MODEL_COLORS = {
    'claude-sonnet-4-5': '#3498db',     # Blue
    'gpt-5': '#9b59b6',                 # Purple
    'gemini-2.5-flash': '#e67e22'       # Orange
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
