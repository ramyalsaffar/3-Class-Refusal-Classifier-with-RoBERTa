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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    recall_score,
    precision_score,
    precision_recall_curve,
    average_precision_score
)

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


# Execute Code Files
#-------------------

# Import Constants and Configuration FIRST (before other modules)
#------------------------------------------------------------------
# NEW LOADING ORDER (after refactoring):
# 1. 01-Constants.py (environment, paths, device, class labels)
# 2. 02-Config.py (user configuration - can reference constants)
# 3. Remaining modules (03-28)

print("\n" + "="*60)
print("LOADING CONSTANTS & CONFIGURATION")
print("="*60)

# Step 1: Load Constants (environment, paths, device detection)
# This must load FIRST so Config can reference DEVICE
exec(open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/01-Constants.py").read())
print("✓ Loaded 01-Constants.py")

# Step 2: Load Configuration (user settings)
exec(open(CodeFilePath+"02-Config.py").read())
print("✓ Loaded 02-Config.py")


# Load remaining code files (03-28, excluding execution scripts)
#-----------------------------------------------------------------
# Files are numbered 00-28:
#   00-Imports.py (this file)
#   01-Constants.py (loaded above - environment, paths, device, class labels)
#   02-Config.py (loaded above - user configuration)
#   03-AWSConfig.py (AWS configuration)
#   04-SecretsHandler.py (AWS Secrets Manager)
#   05-07: Data Collection (PromptGenerator, ResponseCollector, DataLabeler - with dual-task labeling)
#   08: DataCleaner (comprehensive data quality validation and cleaning)
#   09: Dataset (generic PyTorch Dataset - works for both classifiers)
#   10: RefusalClassifier (3-class: No/Hard/Soft Refusal)
#   11: JailbreakClassifier (2-class: Jailbreak Failed/Succeeded)
#   12: WeightedLoss (handles class imbalance)
#   13: Trainer (generic trainer - works for both classifiers)
#   14-16: Analysis (PerModelAnalyzer, ConfidenceAnalyzer, AdversarialTester)
#   17: JailbreakAnalysis (security-critical jailbreak analysis with cross-analysis)
#   18-19: Interpretability (AttentionVisualizer, ShapAnalyzer)
#   20: Visualization (Visualizer)
#   21-22: Orchestration (RefusalPipeline - trains both classifiers, ExperimentRunner)
#   23-Execute.py (main entry point - don't load)
#   24-Analyze.py (analysis entry point - don't load)
#   25-ProductionAPI.py (production API server - don't load)
#   26-MonitoringSystem.py (production monitoring - don't load)
#   27-RetrainingPipeline.py (production retraining - don't load)
#   28-DataManager.py (production data management - don't load)

print("\nLoading modules...")
code_files_ls = sorted([x for x in os.listdir(CodeFilePath) if x.endswith('.py')])

# Remove files we don't want to auto-load
code_files_ls = [x for x in code_files_ls if x not in [
    "00-Imports.py",            # This file
    "01-Constants.py",          # Already loaded (environment, paths, device)
    "02-Config.py",             # Already loaded (user configuration)
    "23-Execute.py",            # Execution script
    "24-Analyze.py",            # Execution script
    "25-ProductionAPI.py",      # Production API server (load manually)
    "26-MonitoringSystem.py",   # Production monitoring (load manually)
    "27-RetrainingPipeline.py", # Production retraining (load manually)
    "28-DataManager.py"         # Production data management (load manually)
]]

# Loop over code files and load them
for file in code_files_ls:
    try:
        exec(open(CodeFilePath+file).read())
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
