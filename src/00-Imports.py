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
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    RobertaConfig,
    AdamW,
    get_linear_schedule_with_warmup
)

# ML Tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support
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
models_path = project_path + "/models/"
results_path = base_results_path
visualizations_path = project_path + "/visualizations/"
reports_path = project_path + "/reports/"


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


# AWS Secrets Handler Class
#---------------------------
# Only loaded if AWS is available
if AWS_AVAILABLE:
    class SecretsHandler:
        """Handle AWS Secrets Manager operations"""

        def __init__(self, region='us-east-1'):
            self.region = region
            self.client = boto3.client('secretsmanager', region_name=region)

        def get_secret(self, secret_name):
            """Retrieve secret from AWS Secrets Manager"""
            try:
                response = self.client.get_secret_value(SecretId=secret_name)

                if 'SecretString' in response:
                    secret = response['SecretString']
                    try:
                        return json.loads(secret)
                    except json.JSONDecodeError:
                        return secret
                else:
                    return response['SecretBinary']

            except ClientError as e:
                error_code = e.response['Error']['Code']
                print(f"❌ Error retrieving secret '{secret_name}': {error_code}")
                raise

        def get_api_key(self, secret_name, key_field='api_key'):
            """Retrieve API key from Secrets Manager"""
            secret = self.get_secret(secret_name)

            if isinstance(secret, dict):
                return secret.get(key_field) or secret.get(key_field.upper())
            else:
                return secret


#------------------------------------------------------------------------------


# Execute Code Files
#-------------------

# Import Configuration and Constants FIRST (before other modules)
#-----------------------------------------------------------------
print("\n" + "="*60)
print("LOADING CONFIGURATION")
print("="*60)

exec(open(CodeFilePath+"01-Config.py").read())
print("✓ Loaded 01-Config.py")

exec(open(CodeFilePath+"02-Constants.py").read())
print("✓ Loaded 02-Constants.py")


# Load remaining code files (03-17, excluding 16 and 17)
#--------------------------------------------------------
# Files are numbered 00-17:
#   00-Imports.py (this file)
#   01-Config.py (loaded above)
#   02-Constants.py (loaded above)
#   03-05: Data Collection (PromptGenerator, ResponseCollector, DataLabeler)
#   06-08: Model Components (Dataset, Classifier, WeightedLoss)
#   09: Training (Trainer)
#   10-13: Analysis (PerModelAnalyzer, ConfidenceAnalyzer, AdversarialTester, Visualizer)
#   14-15: Orchestration (RefusalPipeline, ExperimentRunner)
#   16-Execute.py (main entry point - don't load)
#   17-Analyze.py (analysis entry point - don't load)

print("\nLoading modules...")
code_files_ls = sorted([x for x in os.listdir(CodeFilePath) if x.endswith('.py')])

# Remove files we don't want to auto-load
code_files_ls = [x for x in code_files_ls if x not in [
    "00-Imports.py",      # This file
    "01-Config.py",        # Already loaded
    "02-Constants.py",     # Already loaded
    "16-Execute.py",       # Execution script
    "17-Analyze.py"        # Execution script
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
