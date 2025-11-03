# Constants & AWS Configuration
#-------------------------------
# This file contains:
# - Global constants used throughout the project
# - AWS configuration for cloud deployment
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


# =============================================================================
# AWS CONFIGURATION
# =============================================================================

# AWS Configuration (Optional)
#-----------------------------
AWS_CONFIG = {
    'enabled': IS_AWS,
    'region': os.getenv('AWS_REGION', 'us-east-1'),
    's3_bucket': os.getenv('S3_BUCKET_NAME', 'refusal-classifier-results'),
    's3_results_prefix': 'runs/',
    's3_logs_prefix': 'logs/',

    # AWS Secrets Manager keys
    'secrets': {
        'openai': os.getenv('SECRETS_OPENAI_KEY_NAME', 'refusal-classifier/openai-api-key'),
        'anthropic': os.getenv('SECRETS_ANTHROPIC_KEY_NAME', 'refusal-classifier/anthropic-api-key'),
        'google': os.getenv('SECRETS_GOOGLE_KEY_NAME', 'refusal-classifier/google-api-key')
    },

    # EC2 Configuration
    'ec2_instance_type': os.getenv('EC2_INSTANCE_TYPE', 'g4dn.xlarge'),  # GPU instance
    'ec2_security_group': 'refusal-classifier-sg',
    'iam_role_name': 'refusal-classifier-ec2-role'
}


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
