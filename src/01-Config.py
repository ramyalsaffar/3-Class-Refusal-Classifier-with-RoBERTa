# Control Room Configuration
#---------------------------
# This file contains all configuration settings.
# This is your CONTROL ROOM - modify parameters here.
###############################################################################


# =============================================================================
# CONTROL ROOM - All Configuration Settings
# =============================================================================


# API Configuration
#------------------
API_CONFIG = {
    # Prompt Generation (GPT-4 generates diverse prompts)
    'prompt_model': 'gpt-4',

    # Response Collection (3 models: Claude, GPT-5, Gemini)
    'response_models': {
        'claude': 'claude-sonnet-4-20250514',
        'gpt5': 'gpt-5',
        'gemini': 'gemini-2.5-flash'
    },

    # Adversarial Testing (GPT-4 for paraphrasing)
    'paraphrase_model': 'gpt-4',

    # Rate Limiting & Retries
    'rate_limit_delay': 0.5,                    # Seconds between API calls
    'max_retries': 5,                           # Max retries for failed API calls

    # Temperature Settings
    'temperature_generate': 0.9,                # High = diverse prompts
    'temperature_response': 0.7,                # Medium = varied responses
    'temperature_paraphrase': 0.7,              # Medium = natural paraphrases

    # Token Limits
    'max_tokens_generate': 4000,                # For prompt generation
    'max_tokens_response': 1024,                # For LLM responses
    'max_tokens_paraphrase': 500                # For paraphrasing
}


# Model Configuration (RoBERTa Classifier)
#------------------------------------------
MODEL_CONFIG = {
    'model_name': 'roberta-base',               # Pretrained RoBERTa model
    'num_classes': 3,                           # 0=No Refusal, 1=Hard, 2=Soft
    'max_length': 512,                          # Max sequence length (tokens)
    'dropout': 0.1,                             # Dropout probability
    'freeze_layers': 6                          # Freeze bottom 6 layers (0-11 available)
}


# Training Configuration
#-----------------------
TRAINING_CONFIG = {
    'batch_size': 16,                           # Batch size (reduce if OOM)
    'epochs': 5,                                # Number of training epochs
    'learning_rate': 2e-5,                      # Learning rate (AdamW)
    'warmup_steps': 100,                        # Warmup steps for LR scheduler
    'weight_decay': 0.01,                       # Weight decay (L2 regularization)
    'gradient_clip': 1.0,                       # Gradient clipping max norm
    'early_stopping_patience': 3,               # Epochs to wait before early stopping
    'save_best_only': True,                     # Only save best model checkpoint
    'num_workers': 0,                           # DataLoader workers (0 for Mac, 4 for AWS)
    'pin_memory': True,                         # Pin memory for faster GPU transfer
    'device': DEVICE                            # Auto-detected in 00-Imports.py
}


# Dataset Configuration
#----------------------
DATASET_CONFIG = {
    'total_prompts': 2000,                      # Total prompts to generate

    # Models to collect responses from (3 models × 2000 prompts = 6000 responses)
    'models': ['claude-sonnet-4.5', 'gpt-5', 'gemini-2.5-flash'],

    # Train/Val/Test splits
    'train_split': 0.7,                         # 70% training
    'val_split': 0.15,                          # 15% validation
    'test_split': 0.15,                         # 15% test
    'random_seed': 42,                          # For reproducibility

    # Category distribution (must sum to total_prompts = 2000)
    # Hard Refusal: 795 prompts across 5 categories
    # Soft Refusal: 205 prompts across 4 categories
    # No Refusal: 1000 prompts across 4 categories
    'categories': {
        # HARD REFUSAL CATEGORIES (795 total)
        'violence_illegal': {
            'hard': 285,        # Weapons, violence, illegal activities
            'soft': 15          # Edge cases (self-defense, security)
        },
        'hate_speech': {
            'hard': 150,        # Explicit hate speech
            'soft': 0
        },
        'jailbreaks': {
            'hard': 225,        # Role-play, DAN-style, adversarial
            'soft': 25          # Borderline hypotheticals
        },
        'privacy_misinfo': {
            'hard': 90,         # PII requests, doxxing, fake news
            'soft': 60          # Privacy tips, misinfo education
        },
        'regulated_content': {
            'hard': 45,         # Medical diagnosis, legal strategy
            'soft': 105         # General health/legal info with disclaimers
        },

        # NO REFUSAL CATEGORIES (1000 total)
        'technical_educational': {
            'no_refusal': 350   # Programming, math, science
        },
        'creative_professional': {
            'no_refusal': 300   # Writing, business, career
        },
        'general_assistance': {
            'no_refusal': 200   # Recipes, travel, summaries
        },
        'edge_cases': {
            'no_refusal': 150   # Legitimate but suspicious-sounding
        }
    }
}


# Analysis Configuration
#-----------------------
ANALYSIS_CONFIG = {
    'adversarial_samples': 200,                 # Samples for adversarial testing
    'paraphrase_dimensions': [
        'synonym',                               # Replace words with synonyms
        'restructure',                           # Restructure sentences
        'formality',                             # Change formality level
        'compression'                            # Make more concise
    ],
    'confidence_bins': 20,                      # Bins for confidence histograms
    'low_confidence_threshold': 0.6             # Threshold for low-confidence samples
}


# Experiment Configuration
#-------------------------
EXPERIMENT_CONFIG = {
    'experiment_name': f'refusal_classifier_{datetime.now().strftime("%Y%m%d_%H%M")}',
    'save_intermediate': True,                  # Save intermediate results
    'verbose': True,                            # Print detailed logs
    'show_progress': True,                      # Show progress bars
    'prompt_buffer_percentage': 20,             # Extra prompts as buffer (%)
    'max_prompt_generation_attempts': 3,        # Retries for prompt generation
    'test_sample_size': 10                      # Sample size for --test mode
}


# Timing Configuration
#---------------------
TIMING_CONFIG = {
    'api_delay': 0.5,                           # Delay between API calls (seconds)
    'show_progress': True,                      # Show progress bars
    'estimate_time': True                       # Show time estimates
}


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
Created on October 28, 2025
@author: ramyalsaffar
"""
