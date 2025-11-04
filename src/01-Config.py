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
    'rate_limit_delay': 0.2,                    # Seconds between API calls
    'max_retries': 5,                           # Max retries for failed API calls

    # Temperature Settings
    'temperature_generate': 0.9,                # High = diverse prompts
    'temperature_response': 0.7,                # Medium = varied responses
    'temperature_paraphrase': 0.7,              # Medium = natural paraphrases

    # Token Limits
    'max_tokens_generate': 2000,                # For prompt generation
    'max_tokens_response': 1024,                # For LLM responses
    'max_tokens_paraphrase': 500,               # For paraphrasing

    # LLM Judge Settings (for data labeling)
    'judge_temperature': 0.0,                   # Deterministic for consistency
    'judge_max_tokens': 50,                     # Small JSON response

    # Batch Sizes
    'prompt_generation_batch_size': 50,         # Batch size for generating prompts
    'inference_batch_size': 16                  # Batch size for model inference/analysis
}


# Model Configuration (RoBERTa Classifiers)
#--------------------------------------------
MODEL_CONFIG = {
    'model_name': 'roberta-base',               # Pretrained RoBERTa model
    'num_classes': 3,                           # Refusal: 0=No Refusal, 1=Hard, 2=Soft
    'max_length': 512,                          # Max sequence length (tokens)
    'dropout': 0.1,                             # Dropout probability
    'freeze_layers': 6                          # Freeze bottom 6 layers (0-11 available)
}

# Jailbreak Detector Configuration
#-----------------------------------
JAILBREAK_CONFIG = {
    'model_name': 'roberta-base',               # Same base model as refusal classifier
    'num_classes': 2,                           # 0=Jailbreak Failed, 1=Jailbreak Succeeded
    'max_length': 512,                          # Max sequence length (tokens)
    'dropout': 0.1,                             # Dropout probability
    'freeze_layers': 6,                         # Freeze bottom 6 layers
    'enabled': True                             # Enable jailbreak detection training
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

    # Category distribution (percentages - will scale with total_prompts)
    # WHY: Using percentages allows automatic scaling for any total_prompts value
    # Hard Refusal: 39.75% across 5 categories
    # Soft Refusal: 10.25% across 4 categories
    # No Refusal: 50% across 4 categories
    'categories': {
        # HARD REFUSAL CATEGORIES (39.75% total)
        'violence_illegal': {
            'hard': 14.25,      # Weapons, violence, illegal activities
            'soft': 0.75        # Edge cases (self-defense, security)
        },
        'hate_speech': {
            'hard': 7.5,        # Explicit hate speech
            'soft': 0
        },
        'jailbreaks': {
            'hard': 11.25,      # Role-play, DAN-style, adversarial
            'soft': 1.25        # Borderline hypotheticals
        },
        'privacy_misinfo': {
            'hard': 4.5,        # PII requests, doxxing, fake news
            'soft': 3.0         # Privacy tips, misinfo education
        },
        'regulated_content': {
            'hard': 2.25,       # Medical diagnosis, legal strategy
            'soft': 5.25        # General health/legal info with disclaimers
        },

        # NO REFUSAL CATEGORIES (50% total)
        'technical_educational': {
            'no_refusal': 17.5  # Programming, math, science
        },
        'creative_professional': {
            'no_refusal': 15.0  # Writing, business, career
        },
        'general_assistance': {
            'no_refusal': 10.0  # Recipes, travel, summaries
        },
        'edge_cases': {
            'no_refusal': 7.5   # Legitimate but suspicious-sounding
        }
    }
}


# Prompt Generation Configuration
#----------------------------------
# Three-stage generation: Generate → Self-Evaluate → Regenerate Failed
PROMPT_GENERATION_CONFIG = {
    # Three-stage process control
    'stages': {
        'generation': True,                         # Stage 1: Generate prompts
        'self_evaluation': True,                    # Stage 2: GPT-4 quality check
        'regeneration': True,                       # Stage 3: Fix failed prompts
        'max_regeneration_attempts': 2              # Max attempts per failed prompt
    },

    # Human-like characteristics (Stage 1 requirements)
    'human_characteristics': {
        'greeting_probability': 0.30,               # Casual greetings (hey, yo, etc.)
        'typo_probability': 0.20,                   # Grammar errors/typos
        'repeated_word_probability': 0.15,          # Accidental word repetition
        'omit_pronoun_probability': 0.15,           # Omit pronouns or articles
        'filler_words_probability': 0.10,           # Fillers (oh, btw, lol, etc.)
        'partial_caps_probability': 0.05,           # Partial capitalization (heLLo)
        'punctuation_variations': True,             # Vary punctuation always
        'never_use_em_dash': True,                  # Never use em-dash (too formal)
        'tone_distribution': {
            'casual': 0.40,
            'formal': 0.25,
            'rude': 0.15,
            'desperate': 0.10,
            'friendly': 0.10
        }
    },

    # Persona variation (distribute across prompts)
    'persona_distribution': {
        'technical': 0.25,                          # Knows terminology
        'naive': 0.25,                              # Doesn't understand topic
        'urgent': 0.20,                             # Time pressure/emergency
        'casual_troll': 0.15,                       # Joking/provocative
        'friendly': 0.15                            # Friendly tone
    },

    # Token length distribution (realistic user patterns)
    'token_length_distribution': {
        'short': {'probability': 0.50, 'range': (10, 40)},    # Mobile users
        'medium': {'probability': 0.40, 'range': (40, 100)},  # Most common
        'long': {'probability': 0.10, 'range': (100, 150)}    # Detailed requests
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
    'low_confidence_threshold': 0.6,            # Threshold for low-confidence samples
    'error_examples_count': 5                   # Number of error examples to analyze in detail
}


# Interpretability Configuration
#--------------------------------
INTERPRETABILITY_CONFIG = {
    # Attention Visualization
    'attention_samples_per_class': 10,          # Samples per class for attention analysis
    'attention_layer_index': -1,                # Which layer to visualize (-1 = last layer)
    'attention_top_k_tokens': 15,               # Number of top tokens to highlight
    'visualize_all_layers': False,              # Whether to create all-layer comparison plots

    # SHAP Analysis
    'shap_enabled': True,                       # Enable SHAP analysis (requires shap package)
    'shap_samples': 20,                         # Number of samples for SHAP analysis
    'shap_background_samples': 50,              # Background samples for SHAP explainer
    'shap_max_display': 20,                     # Max features to display in summary plots

    # General Interpretability
    'save_interpretability_results': True,      # Save interpretation results
    'generate_example_visualizations': True     # Generate example plots per class
}


# Data Cleaning Configuration
#-----------------------------
DATA_CLEANING_CONFIG = {
    # Length Thresholds
    'min_response_length': 5,                   # Minimum response length (characters)
    'max_response_length': 10000,               # Maximum response length (characters)
    'min_prompt_length': 5,                     # Minimum prompt length (characters)
    'max_prompt_length': 2000,                  # Maximum prompt length (characters)

    # Cleaning Strategy
    'default_strategy': 'auto',                 # auto, conservative, aggressive, none

    # Quality Thresholds
    'excellent_threshold': 2.0,                 # % removal for "Excellent" rating
    'good_threshold': 5.0,                      # % removal for "Good" rating
    'acceptable_threshold': 10.0,               # % removal for "Acceptable" rating

    # Near-Duplicate Detection
    'similarity_threshold': 0.9,                # Jaccard similarity threshold for near-duplicates

    # Verbose Output
    'verbose': True                             # Print detailed cleaning reports
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


# Production Configuration (Optional)
#-------------------------------------
# Configuration for production deployment with real-time monitoring,
# A/B testing, and automated retraining pipeline
PRODUCTION_CONFIG = {
    # PostgreSQL Database
    'database': {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'refusal_classifier_prod'),
        'user': os.getenv('DB_USER', 'refusal_admin'),
        'password': os.getenv('DB_PASSWORD', ''),  # Use env var or secrets manager
        'pool_size': 10,
        'max_overflow': 20
    },

    # Monitoring Thresholds
    'monitoring': {
        'daily_check_size': 100,                    # Samples for daily monitoring check
        'escalated_check_size': 1000,               # Samples for escalated check
        'warning_threshold': 0.10,                  # 10% disagreement triggers warning
        'escalate_threshold': 0.15,                 # 15% disagreement triggers escalation
        'retrain_threshold': 0.20,                  # 20% disagreement triggers retrain
        'check_interval_hours': 24,                 # Daily monitoring
        'trend_window_days': 7,                     # Days to analyze trends
        'escalation_ui_delay': 2                    # Seconds to pause before escalated check (UX)
    },

    # LLM Judge for Production Monitoring
    'judge': {
        'model': 'gpt-4',                           # GPT-4 for high accuracy
        'temperature': 0.0,                         # Deterministic
        'max_tokens': 10,                           # Only need score
        'rate_limit_delay': 0.5,                    # Delay between judge calls
        'max_retries': 3                            # Retries for failed calls
    },

    # A/B Testing Configuration
    'ab_testing': {
        'enabled': False,                           # Start disabled
        'stages': [0.05, 0.25, 0.5, 1.0],          # Rollout stages: 5% → 25% → 50% → 100%
        'min_samples_per_stage': 1000,              # Min samples before next stage
        'max_degradation': 0.02,                    # Max 2% F1 drop to continue
        'shadow_mode': False,                       # Shadow mode (log but don't serve)
        'automatic_rollback': True,                 # Auto-rollback if degradation
        'manual_promotion': True                    # Require manual promotion (MVP)
    },

    # Retraining Configuration
    'retraining': {
        'enabled': True,                            # Enable automated retraining
        'schedule': 'weekly',                       # Weekly retraining schedule
        'trigger_on_drift': True,                   # Trigger if drift detected
        'validation_f1_threshold': 0.85,            # Min F1 for deployment
        'validation_confidence_threshold': 0.80,    # Min avg confidence
        'retain_historical_samples': True,          # Prevent catastrophic forgetting
        'freeze_layers': 6,                         # Transfer learning (freeze bottom layers)
        'max_epochs': 5,                            # Max training epochs
        'early_stopping_patience': 2                # Early stopping patience
    },

    # Data Retention Strategy
    'retention': {
        'recent_days': 7,                           # Recent: keep 100% problematic + 20% correct
        'recent_problematic_rate': 1.0,             # 100% of problematic samples
        'recent_correct_rate': 0.2,                 # 20% of correct samples
        'medium_days': 30,                          # Medium: keep 50% stratified
        'medium_rate': 0.5,                         # 50% stratified sampling
        'longterm_days': 180,                       # Long-term: keep 10% representative
        'longterm_rate': 0.1,                       # 10% representative sampling
        'archive_after_days': 365                   # Archive after 1 year
    },

    # API Configuration
    'api': {
        'host': os.getenv('API_HOST', '0.0.0.0'),
        'port': int(os.getenv('API_PORT', '8000')),
        'workers': int(os.getenv('API_WORKERS', '4')),
        'timeout': 30,                              # Request timeout (seconds)
        'max_request_size': 1024 * 1024,           # 1MB max request
        'enable_cors': True,                        # Enable CORS for frontend
        'log_level': os.getenv('LOG_LEVEL', 'info')
    },

    # Cost Control
    'cost': {
        'max_daily_judge_calls': 5000,              # Max judge calls per day
        'alert_daily_cost': 100.0,                  # Alert if daily cost exceeds $100
        'max_monthly_cost': 2000.0                  # Hard limit: $2000/month
    }
}


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
