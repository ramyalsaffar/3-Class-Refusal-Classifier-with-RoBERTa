# Control Room Configuration
#---------------------------
# This file contains all configuration settings.
# This is your CONTROL ROOM - modify parameters here.
###############################################################################


# =============================================================================
# PART 01: CORE STUDY DESIGN CONFIGURATION
# =============================================================================
# This section contains all critical experimental design parameters


# API Configuration (Centralized LLM Settings)
#-----------------------------------------------
API_CONFIG = {
    # Model Selection
    'prompt_model': 'gpt-4o',                   # Model for prompt generation
    'judge_model': 'gpt-4o',                    # Model for response labeling/judging
    'paraphrase_model': 'gpt-4o',               # Model for adversarial paraphrasing

    # Response Collection Models (3 models to test)
    'response_models': {
        'claude': 'claude-sonnet-4-5',  # Alias (auto-updates to latest snapshot)
        'gpt5': 'gpt-5',
        'gemini': 'gemini-2.5-flash'
    },

    # Temperature Settings
    'temperature_generate': 1.0,                # Default = diverse prompts (max randomness)
    'temperature_judge': 0.0,                   # Deterministic judging
    'temperature_paraphrase': 0.7,              # Medium = natural paraphrases
    'temperature_response': 1.0,                # Default = 1.0 for all models (Claude, GPT-5, Gemini)

    # Token Limits
    'max_tokens_generate': 2000,                # For prompt generation (reduced for speed)
    'max_tokens_regenerate': 500,               # For regenerating failed prompts
    'max_tokens_judge': 50,                     # Small JSON response for judging
    'max_tokens_paraphrase': 200,               # For paraphrasing
    'max_tokens_response': 4096,                # For LLM responses (increased for GPT-5 reasoning tokens)

    # Rate Limiting & Retries
    'rate_limit_delay': 0.2,                    # Seconds between API calls (reduced for speed)
    'max_retries': 5,                           # Max retries for failed API calls

    # Batch Sizes
    'prompt_generation_batch_size': 10,         # Batch size for generating prompts (smaller = more progress updates)
    'inference_batch_size': 16,                 # Batch size for model inference/analysis

    # Parallel Processing (NEW - Phase 2)
    'parallel_workers': 5 if not IS_AWS else 10,  # Concurrent API calls (5 local, 10 AWS)
    'use_async': True,                          # Enable async/parallel processing
    'labeling_batch_size': 100,                 # Checkpoint every N labeled samples
    'collection_batch_size': 500                # Checkpoint every N collected responses
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
    'enabled': True,                            # Enable jailbreak detection training

    # Performance Thresholds
    'min_recall_succeeded': 0.95,               # Minimum recall threshold for jailbreak detection (95%)

    # Smart Validation (NEW - Phase 2/3)
    'min_samples_per_class': 10                 # Skip training if insufficient samples (modern LLMs defend well)
}


# WildJailbreak Supplementation Configuration
#--------------------------------------------
# Supplements jailbreak training data when modern LLMs successfully defend against all attacks
WILDJAILBREAK_CONFIG = {
    # Enable/Disable Supplementation
    'enabled': True,                            # Enable WildJailbreak supplementation

    # Dataset Source
    'dataset_name': 'allenai/wildjailbreak',    # HuggingFace dataset ID
    'dataset_url': 'https://huggingface.co/datasets/allenai/wildjailbreak',

    # Supplementation Thresholds
    'min_threshold_percentage': 50,             # Minimum % of total_prompts needed for training
                                                # Example: 2000 prompts → need 1000 jailbreak samples
                                                # If real samples < 1000 → supplement from WildJailbreak

    'max_supplement_percentage': 80,            # Maximum % of training data from WildJailbreak
                                                # Ensures real data remains primary source when available

    # Sampling Strategy
    'random_seed': DATASET_CONFIG['random_seed'],  # Use same seed for reproducibility
    'diversity_sampling': True,                 # Sample diverse jailbreak tactics/categories
    'diversity_fields': ['tactics'],            # Fields to use for stratification

    # Quality Filters (applied to WildJailbreak samples)
    'min_prompt_length': 10,                    # Minimum prompt length (characters)
    'max_prompt_length': 2000,                  # Maximum prompt length (characters)
    'min_response_length': 5,                   # Minimum response length (characters)
    'max_response_length': 10000,               # Maximum response length (characters)

    # Dataset Loading Parameters
    'dataset_split': 'train',                   # Dataset split to load (train/test/validation)
    'data_type_filter': 'adversarial_harmful',  # WildJailbreak data type to filter for
                                                # Options: adversarial_harmful, adversarial_benign,
                                                #          vanilla_harmful, vanilla_benign

    # Transparency & Reporting
    'warn_threshold': 30,                       # Warn if >30% of data from WildJailbreak
    'track_source': True,                       # Add 'data_source' column to track provenance
    'log_composition': True                     # Print real vs WildJailbreak breakdown during training
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
    'device': DEVICE                            # Auto-detected in 01-Imports.py
}


# Dataset Configuration
#----------------------
DATASET_CONFIG = {
    'total_prompts': 2000,                      # Total prompts to generate

    # Models to collect responses from (3 models × 2000 prompts = 6000 responses)
    # WHY: Derived from API_CONFIG['response_models'] - single source of truth
    # Access via: list(API_CONFIG['response_models'].values())

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
        'self_evaluation': True,                    # Stage 2: GPT-4o quality check
        'regeneration': True,                       # Stage 3: Fix failed prompts
        'max_regeneration_attempts': 2              # Max attempts per failed prompt (reduced for speed)
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


# Labeling Configuration
#-----------------------
LABELING_CONFIG = {
    'low_confidence_threshold': 60,             # Confidence threshold (0-100 scale)
}


# Checkpointing Configuration (NEW - Phase 2)
#----------------------------------------------
# Automatic checkpointing for error recovery during long-running operations
CHECKPOINT_CONFIG = {
    # Checkpoint Frequency
    'labeling_checkpoint_every': 100,           # Save checkpoint every N labeled samples
    'collection_checkpoint_every': 500,         # Save checkpoint every N collected responses

    # Resume Settings
    'labeling_resume_enabled': True,            # Auto-resume labeling from checkpoint
    'collection_resume_enabled': True,          # Auto-resume collection from checkpoint

    # Cleanup Settings
    'auto_cleanup': True,                       # Auto-delete old checkpoints
    'keep_last_n': 2,                           # Keep only last N checkpoints
    'max_checkpoint_age_hours': 48              # Delete checkpoints older than N hours
}


# Adversarial Testing Configuration
#------------------------------------
ADVERSARIAL_CONFIG = {
    # Paraphrase Validation Thresholds
    'min_semantic_similarity': 0.85,            # Cosine similarity threshold for paraphrase validation
    'min_length_ratio': 0.3,                    # Minimum length ratio (paraphrase/original)
    'max_length_ratio': 3.0,                    # Maximum length ratio (paraphrase/original)
    'max_paraphrase_attempts': 3,               # Maximum retry attempts for paraphrasing

    # Quality Thresholds
    'min_success_rate': 70,                     # Minimum paraphrase success rate (percent)
    'min_avg_semantic_similarity': 0.90,        # Warning threshold for avg semantic similarity
    'retry_effectiveness_threshold': 30,        # Threshold for retry effectiveness warning (percent)
    'failed_retries_warning_threshold': 20,     # Threshold for failed retries warning (percent)
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
    'experiment_name': f'dual_RoBERTa_classifier_{datetime.now().strftime("%Y%m%d_%H%M")}',
    'save_intermediate': True,                  # Save intermediate results
    'verbose': True,                            # Print detailed logs
    'show_progress': True,                      # Show progress bars
    'prompt_buffer_percentage': 20,             # Extra prompts as buffer (%)
    'max_prompt_generation_attempts': 3,        # Retries for prompt generation
    'test_sample_size': 50                      # Sample size for --test mode (min 50 for 3-class stratified splits)
}


# Cross-Validation Configuration
#--------------------------------
CROSS_VALIDATION_CONFIG = {
    'default_folds': 5,                         # Default number of folds
    'final_val_split': 0.1,                     # Validation split for final model monitoring (10%)
}


# =============================================================================
# PART 02: ANALYSIS, VISUALIZATION & INFRASTRUCTURE
# =============================================================================
# This section contains auxiliary settings for analysis, visualization, and production


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
    'low_confidence_threshold': 0.6,            # Threshold for low-confidence samples (0.0-1.0 scale)
    'high_confidence_threshold': 0.5,           # Threshold for high-confidence region in analysis
    'error_examples_count': 5,                  # Number of error examples to analyze in detail
    'attention_sample_size': 100,               # Samples for power law attention analysis

    # Pareto Principle / Power Law Analysis
    'pareto_threshold': 80,                     # Pareto principle 80% threshold
    'pareto_strict_threshold': 30,              # Allow up to 30% of groups for 80% of errors
    'overconfidence_threshold': 0.7,            # High confidence on errors threshold

    # Attention Concentration Analysis
    'attention_top_k_percentage': 5,            # Top K as divisor (1/5 = 20% of tokens)
    'attention_power_law_fit_limit': 1000,      # Number of samples for power law fitting
    'min_attention_concentration': 0.6,         # Minimum attention concentration for top-k

    # Correlation & Agreement Thresholds
    'high_agreement_threshold': 0.95,           # High agreement threshold (95%)
    'moderate_agreement_threshold': 0.85,       # Moderate agreement threshold (85%)
    'min_samples_per_class': 30,                # Minimum samples per class for reliable analysis

    # Disagreement Analysis
    'top_k_disagreements': 50                   # Number of top disagreement cases to extract
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
    'shap_samples_per_class': 2,                # Number of SHAP samples to visualize per class
    'shap_background_samples': 50,              # Background samples for SHAP explainer
    'shap_max_display': 20,                     # Max features to display in summary plots

    # Statistical Agreement Thresholds (Cohen's Kappa, etc.)
    'kappa_thresholds': {
        'almost_perfect': 0.80,                 # Almost perfect agreement (κ > 0.80)
        'substantial': 0.60,                    # Substantial agreement (0.60 < κ ≤ 0.80)
        'moderate': 0.40,                       # Moderate agreement (0.40 < κ ≤ 0.60)
        'fair': 0.20,                           # Fair agreement (0.20 < κ ≤ 0.40)
        'slight': 0.0                           # Slight agreement (0.0 < κ ≤ 0.20)
    },

    # Power Law Analysis Thresholds
    'power_law_exponent_range': (1.0, 3.0),    # Valid range for power law exponent
    'zipf_exponent_range': (0.8, 1.5),         # Valid range for Zipfian distribution exponent

    # Calibration Thresholds (Expected Calibration Error)
    'ece_thresholds': {
        'excellent': 0.05,                      # ECE < 0.05 = excellent calibration
        'good': 0.10,                           # ECE < 0.10 = good calibration
        'acceptable': 0.15                      # ECE < 0.15 = acceptable calibration
    },

    # General Interpretability
    'save_interpretability_results': True,      # Save interpretation results
    'generate_example_visualizations': True     # Generate example plots per class
}


# Error Analysis Configuration
#------------------------------
ERROR_ANALYSIS_CONFIG = {
    'min_confidence': 0.3,                      # Minimum confidence for analysis
    'high_confidence_threshold': 0.8,           # High confidence threshold (confident mistakes)
    'top_k_misclassifications': 5,              # Top K errors to show in summaries
    'top_k_errors': 50,                         # Number of failure cases to extract for detailed analysis
    'quantiles': [0.25, 0.5, 0.75],            # Quantiles for distribution analysis

    # Length Analysis
    'length_bins': [0, 20, 50, 100, 200],      # Token length bins for length-based analysis
}


# Hypothesis Testing Configuration
#-----------------------------------
HYPOTHESIS_TESTING_CONFIG = {
    'alpha': 0.05,                              # Significance level for tests
}


# Visualization Configuration
#-----------------------------
VISUALIZATION_CONFIG = {
    'dpi': 150,                      # DPI for all saved plots
    'figure_format': 'png',          # Output format
    'style': 'whitegrid',            # Seaborn style
    'color_palette': 'Set2',
    'font_scale': 1.0,
    'f1_target': 0.8                 # Target F1 score threshold for visualization
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

    # Monitoring Thresholds (used by MonitoringSystem)
    'monitoring_thresholds': {
        'small_sample_size': 100,                   # Samples for daily monitoring check
        'large_sample_size': 1000,                  # Samples for escalated check
        'warning_threshold': 0.10,                  # 10% disagreement triggers warning
        'escalate_threshold': 0.15,                 # 15% disagreement triggers escalation
        'retrain_threshold': 0.20,                  # 20% disagreement triggers retrain
        'check_interval_hours': 24,                 # Daily monitoring
        'trend_window_days': 7                      # Days to analyze trends
    },

    # Rate limiting and delays (root level for easy access)
    'judge_rate_limit': 1.0,                        # Delay between LLM judge calls (seconds)
    'escalation_ui_delay': 2.0,                     # Seconds to pause before escalated check (UX)

    # LLM Judge for Production Monitoring
    'judge': {
        'model': 'gpt-4o',                          # GPT-4o for high accuracy
        'temperature': 0.0,                         # Deterministic
        'max_tokens': 10,                           # Only need score
        'max_retries': 3                            # Retries for failed calls
    },

    # A/B Testing Configuration
    'ab_test_stages': [0.05, 0.25, 0.5, 1.0],      # Gradual rollout stages: 5% → 25% → 50% → 100%
    'ab_testing': {
        'enabled': False,                           # Start disabled
        'min_samples_per_stage': 1000,              # Min samples before next stage
        'max_degradation': 0.02,                    # Max 2% F1 drop to continue
        'shadow_mode': False,                       # Shadow mode (log but don't serve)
        'automatic_rollback': True,                 # Auto-rollback if degradation
        'manual_promotion': True                    # Require manual promotion (MVP)
    },

    # Validation Thresholds (used by RetrainingPipeline)
    'validation_thresholds': {
        'min_f1_score': 0.85,                       # Min F1 score for model deployment
        'min_avg_confidence': 0.80                  # Min average confidence score
    },

    # Retraining Configuration
    'retraining': {
        'enabled': True,                            # Enable automated retraining
        'schedule': 'weekly',                       # Weekly retraining schedule
        'trigger_on_drift': True,                   # Trigger if drift detected
        'retain_historical_samples': True,          # Prevent catastrophic forgetting
        'freeze_layers': 4,                         # Transfer learning (freeze bottom layers) - less than initial for adaptation
        'max_epochs': 5,                            # Max training epochs
        'early_stopping_patience': 2,               # Early stopping patience
        'min_training_samples': 100,                # Minimum samples required for retraining
        'lr_multiplier': 0.5,                       # Learning rate multiplier vs initial training (lower for fine-tuning)
        'test_split': 0.3,                          # Test set ratio
        'val_split': 0.5,                           # Validation split (from remaining 70%)
        'high_confidence_threshold': 0.8            # Confidence threshold for data filtering
    },

    # Data Retention Strategy (Implemented in 34-DataManager.py)
    # WHY: Comprehensive strategy to balance storage costs with data quality
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
        'max_text_length': 10000,                   # Maximum input text length
        'enable_cors': True,                        # Enable CORS for frontend
        'cors_origins': os.getenv('CORS_ORIGINS', '*').split(',') if os.getenv('CORS_ORIGINS') else ['*'],  # Allowed origins (use env var in production)
        'log_level': os.getenv('LOG_LEVEL', 'info')
    },

    # Cost Control (Future Feature - Not yet implemented)
    # WHY: Reserved for cost tracking and budget alerts in production
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
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
