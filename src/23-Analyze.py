#!/usr/bin/env python3
# Analyze Pipeline
#-----------------
# Standalone analysis script for 3-Class Refusal Classifier.
# Loads trained model and runs analysis on test data.
#
# Usage:
#   python src/17-Analyze.py
#   python src/17-Analyze.py --model models/my_model.pt
#
###############################################################################


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("📊 REFUSAL CLASSIFIER - ANALYSIS MODE")
    print("="*60)

    # Parse arguments
    model_path = None
    if len(sys.argv) > 2 and sys.argv[1] == '--model':
        model_path = sys.argv[2]
    elif len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Initialize runner and run analysis
    runner = ExperimentRunner()
    runner.analyze_only(model_path)


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-Class Refusal Classifier with RoBERTa
Created on October 28, 2025
@author: ramyalsaffar
"""
