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


# Execute Code Files
#-------------------

CodeFilePath = glob.glob(project_path + "/*Code/*Python/")[0]
code_files_ls = os.listdir(CodeFilePath)
code_files_ls.sort()
code_files_ls = [x for x in code_files_ls if "py" in x]
code_files_ls = code_files_ls[1:31]  # Load files 02-31 (skip 01-Imports.py, skip production files 32-35)


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
