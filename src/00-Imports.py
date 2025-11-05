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
import subprocess
import signal

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from io import BytesIO
import getpass
import atexit

# Async and Parallel Processing (NEW - for Phase 2)
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
import tiktoken  # Token counting for API usage tracking

# AWS (optional - only needed for cloud deployment)
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("ℹ️  boto3 not available - AWS features disabled")


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


# macOS Sleep Prevention
#-----------------------
# Prevents Mac from sleeping during long-running experiments
# Uses native 'caffeinate' command to keep system awake

class KeepAwake:
    """
    Context manager to prevent macOS from sleeping during long operations.

    Uses the native 'caffeinate' command which prevents:
    - Display sleep
    - System sleep
    - Disk sleep

    Automatically restores normal sleep behavior when done or on error.
    Only active on macOS - no-op on other platforms.

    Usage:
        with KeepAwake():
            # Your long-running code here
            pipeline.run_full_pipeline()
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize KeepAwake context manager.

        Args:
            verbose: Print status messages (default: True)
        """
        self.verbose = verbose
        self.process = None
        self.is_mac = sys.platform == 'darwin'

    def __enter__(self):
        """Start preventing sleep when entering context."""
        if not self.is_mac:
            if self.verbose:
                print("ℹ️  Sleep prevention only available on macOS")
            return self

        try:
            # Start caffeinate process
            # -d: Prevent display sleep
            # -i: Prevent system idle sleep
            # -m: Prevent disk sleep
            self.process = subprocess.Popen(
                ['caffeinate', '-dims'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            if self.verbose:
                print("☕ caffeinate activated - Mac will stay awake during execution")

            # Register cleanup in case of unexpected termination
            atexit.register(self._cleanup)

        except FileNotFoundError:
            if self.verbose:
                print("⚠️  caffeinate command not found - sleep prevention unavailable")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not activate sleep prevention: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop preventing sleep when exiting context."""
        self._cleanup()
        return False  # Don't suppress exceptions

    def _cleanup(self):
        """Terminate caffeinate process and restore normal sleep behavior."""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                if self.verbose:
                    print("☕ caffeinate deactivated - Mac sleep settings restored")
            except Exception:
                # Force kill if terminate doesn't work
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None
                # Unregister from atexit to avoid double cleanup
                try:
                    atexit.unregister(self._cleanup)
                except Exception:
                    pass


#------------------------------------------------------------------------------


# Execute Code Files
#-------------------

# Import Constants and Configuration FIRST (before other modules)
#------------------------------------------------------------------
# NEW LOADING ORDER (after refactoring):
# 1. 01-Constants.py (environment, paths, device, class labels)
# 2. 02-Config.py (user configuration - can reference constants)
# 3. Remaining modules (03-34)

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


# Load remaining code files (03-34, excluding 29-34)
#--------------------------------------------------------
# Files are numbered 00-34:
#   00-Imports.py (this file)
#   01-Constants.py (loaded above - environment, paths, device, class labels)
#   02-Config.py (loaded above - user configuration)
#   03-Constants.py (old file - will be deleted)
#   04-AWS.py (AWS configuration)
#   05-PromptGenerator.py (3-stage prompt generation)
#   06-ResponseCollector.py (multi-LLM response collection)
#   07-DataCleaner.py (comprehensive data cleaning)
#   08-DataLabeler.py (LLM judge labeling)
#   09-LabelingQualityAnalyzer.py (labeling quality metrics)
#   10-ClassificationDataset.py (PyTorch Dataset)
#   11-RefusalClassifier.py (3-class RoBERTa model)
#   12-JailbreakDetector.py (2-class RoBERTa model)
#   13-Trainer.py (standard trainer with weighted loss)
#   14-CrossValidator.py (k-fold cross-validation)
#   15-PerModelAnalyzer.py (per-model performance analysis)
#   16-ConfidenceAnalyzer.py (confidence score analysis)
#   17-AdversarialTester.py (paraphrasing robustness tests)
#   18-JailbreakAnalysis.py (security-focused jailbreak analysis)
#   19-CorrelationAnalysis.py (refusal ↔ jailbreak correlation)
#   20-AttentionVisualizer.py (attention heatmaps)
#   21-ShapAnalyzer.py (SHAP interpretability)
#   22-PowerLawAnalyzer.py (power law analysis)
#   23-HypothesisTesting.py (statistical hypothesis tests)
#   24-ErrorAnalysis.py (comprehensive error analysis)
#   25-Visualizer.py (basic plotting functions)
#   26-ReportGenerator.py (PDF report generation)
#   27-RefusalPipeline.py (main training pipeline)
#   28-ExperimentRunner.py (experiment orchestration)
#   29-Execute.py (main entry point - don't load)
#   30-Analyze.py (analysis script - don't load)
#   31-ProductionAPI.py (FastAPI server - don't load)
#   32-MonitoringSystem.py (production monitoring - don't load)
#   33-RetrainingPipeline.py (automated retraining - don't load)
#   34-DataManager.py (production data management - don't load)

print("\nLoading modules...")

# Get CodeFilePath from Constants
CodeFilePath = glob.glob(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/*Code/*Python/")[0]
code_files_ls = os.listdir(CodeFilePath)
code_files_ls.sort()
code_files_ls = [x for x in code_files_ls if "py" in x]

# Remove files we don't want to auto-load
code_files_ls = [x for x in code_files_ls if x not in [
    "00-Imports.py",      # This file
    "01-Constants.py",    # Already loaded (environment, paths, device)
    "02-Config.py",       # Already loaded (user configuration)
    "03-Constants.py",    # Old file (will be deleted)
    "29-Execute.py",      # Execution script
    "30-Analyze.py",      # Execution script
    "31-ProductionAPI.py",      # Production API server (load manually)
    "32-MonitoringSystem.py",   # Production monitoring (load manually)
    "33-RetrainingPipeline.py", # Production retraining (load manually)
    "34-DataManager.py"         # Production data management (load manually)
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
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
