# Imports
#--------
#
# This file has all of the libraries needed for the project.
# It also imports configuration and constants.
# Paths to load from or to.
#
###############################################################################


#------------------------------------------------------------------------------


# Libraries
#----------
import os
import sys
import json
import time
import re
import glob
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import getpass
import atexit

# Data Science
import pandas as pd
import numpy as np
from tqdm import tqdm

# PyTorch & Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # Moved from transformers to torch.optim in newer versions
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    RobertaConfig,
    get_linear_schedule_with_warmup
)

# ML Tools
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    recall_score,
    precision_score,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
    ConfusionMatrixDisplay
)

# Statistical Testing
from scipy.stats import chisquare, shapiro
import pickle

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# API Clients
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# AWS (optional - only needed for cloud deployment)
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("ℹ️  boto3 not available - AWS features disabled")


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

models_path = glob.glob(base_results_path + "/*Models/")[0]
results_path = base_results_path

visualizations_path = glob.glob(base_results_path + "/*Visualizations/")[0]
reports_path = glob.glob(base_results_path + "/*Reports/")[0]


# Create directories if they don't exist
#----------------------------------------
for path in [base_results_path, data_path, data_raw_path, data_responses_path,
             data_processed_path, data_splits_path, models_path, results_path,
             visualizations_path, reports_path]:
    os.makedirs(path, exist_ok=True)


#------------------------------------------------------------------------------


# Python Display Settings
#------------------------
pd.set_option('display.max_rows', 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", 250)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.4f}'.format


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


# Execute Code Files
#-------------------

# Import Configuration and Constants FIRST (before other modules)
#-----------------------------------------------------------------
print("\n" + "="*60)
print("LOADING CONFIGURATION")
print("="*60)

# FIX: Use context manager to properly close file descriptors
with open(CodeFilePath+"02-Config.py") as f:
    exec(f.read())
print("✓ Loaded 02-Config.py")

with open(CodeFilePath+"03-Constants.py") as f:
    exec(f.read())
print("✓ Loaded 03-Constants.py")


# Load remaining code files (04-34, excluding 29-34)
#--------------------------------------------------------
# Files are numbered 01-34:
#   01-Imports.py (this file)
#   02-Config.py (loaded above) - includes AWS_CONFIG
#   03-Constants.py (loaded above)
#   04-SecretsHandler.py (AWS Secrets Manager)
#   05-06: Data Collection (PromptGenerator, ResponseCollector)
#   07: DataCleaner (comprehensive data quality validation and cleaning)
#   08: DataLabeler (dual-task labeling: refusal + jailbreak detection)
#   09: LabelingQualityAnalyzer (analyze judge confidence and labeling quality)
#   10: ClassificationDataset (generic PyTorch Dataset - works for both classifiers)
#   11: RefusalClassifier (3-class: No/Hard/Soft Refusal)
#   12: JailbreakDetector (2-class: Jailbreak Failed/Succeeded)
#   13-14: Training Methods (Trainer, CrossValidator with k-fold CV)
#   15-17: Analysis - Performance (PerModelAnalyzer, ConfidenceAnalyzer, AdversarialTester)
#   18-19: Analysis - Security (JailbreakAnalysis, CorrelationAnalysis)
#   20-21: Analysis - Interpretability (AttentionVisualizer, ShapAnalyzer)
#   22-24: Analysis - Statistical (PowerLawAnalyzer, HypothesisTesting, ErrorAnalysis)
#   25-26: Visualization & Reporting (Visualizer, ReportGenerator)
#   27-28: Orchestration (RefusalPipeline, ExperimentRunner)
#   29-Execute.py (main entry point - don't load)
#   30-Analyze.py (analysis entry point - don't load)
#   31-ProductionAPI.py (production API server - don't load)
#   32-MonitoringSystem.py (production monitoring - don't load)
#   33-RetrainingPipeline.py (production retraining - don't load)
#   34-DataManager.py (production data management - don't load)

print("\nLoading modules...")
code_files_ls = sorted([x for x in os.listdir(CodeFilePath) if x.endswith('.py')])

# Remove files we don't want to auto-load
code_files_ls = [x for x in code_files_ls if x not in [
    "01-Imports.py",      # This file
    "02-Config.py",        # Already loaded
    "03-Constants.py",     # Already loaded
    "29-Execute.py",       # Execution script
    "30-Analyze.py",       # Execution script
    "31-ProductionAPI.py",      # Production API server (load manually)
    "32-MonitoringSystem.py",   # Production monitoring (load manually)
    "33-RetrainingPipeline.py", # Production retraining (load manually)
    "34-DataManager.py"         # Production data management (load manually)
]]

# Loop over code files and load them
# FIX: Use context manager to properly close file descriptors
for file in code_files_ls:
    try:
        with open(CodeFilePath+file) as f:
            exec(f.read())
        print(f"✓ Loaded {file}")
    except Exception as e:
        print(f"✗ Error loading {file}: {e}")


print("="*60)
print("✅ All modules loaded successfully")
print("="*60 + "\n")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-Class Refusal Classifier with RoBERTa
Created on October 28, 2025
@author: ramyalsaffar
"""
