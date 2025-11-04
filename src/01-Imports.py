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
from io import BytesIO
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

# PDF Report Generation (optional - only needed for report generation)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.platypus.flowables import HRFlowable
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ℹ️  reportlab not available - PDF report generation disabled")

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

# API Keys (local file storage)
api_keys_file_path = glob.glob(project_path + "/*API Keys/API Keys.txt")[0]


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

CodeFilePath = glob.glob(project_path + "/*Code/*Python/")[0]
code_files_ls = os.listdir(CodeFilePath)
code_files_ls.sort()
code_files_ls = [x for x in code_files_ls if "py" in x]
code_files_ls = code_files_ls[1:30]


# Loop over cde files
#--------------------
for i in range(0,len(code_files_ls)):

    file = code_files_ls[i]

    print(file)

    exec(open(CodeFilePath+file).read())
    print(f"✓ Loaded {file}")


print("="*60)
print("✅ All modules loaded successfully")
print("="*60 + "\n")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
