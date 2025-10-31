#!/usr/bin/env python3
# Analyze Pipeline
#-----------------
# Standalone analysis script for 3-Class Refusal Classifier.
# Loads trained models and runs comprehensive analysis on test data.
#
# Usage:
#   python src/Analyze.py                                    # Interactive mode
#   python src/Analyze.py --help                             # Show help
#   python src/Analyze.py --refusal-model models/refusal.pt  # Analyze with custom refusal model
#   python src/Analyze.py --refusal-model models/refusal.pt --jailbreak-model models/jailbreak.pt
#   python src/Analyze.py --test-data data/splits/test.pkl   # Use custom test data
#
###############################################################################


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze trained Refusal Classifier and Jailbreak Detector models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode (default)
  python src/Analyze.py

  # Analyze with default model paths
  python src/Analyze.py --auto

  # Specify refusal model only (jailbreak uses default)
  python src/Analyze.py --refusal-model models/my_refusal.pt

  # Specify both models
  python src/Analyze.py --refusal-model models/refusal.pt --jailbreak-model models/jailbreak.pt

  # Use custom test data
  python src/Analyze.py --test-data data/custom_test.pkl

  # Combine options
  python src/Analyze.py --refusal-model models/refusal.pt --test-data data/test.pkl
        '''
    )

    parser.add_argument(
        '--refusal-model',
        type=str,
        default=None,
        help='Path to trained refusal classifier model (.pt file)'
    )

    parser.add_argument(
        '--jailbreak-model',
        type=str,
        default=None,
        help='Path to trained jailbreak detector model (.pt file)'
    )

    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data (.pkl file). If not specified, uses data/splits/test.pkl'
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help='Automatically use default model paths without prompting'
    )

    return parser.parse_args()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_file_exists(file_path: str, file_type: str) -> bool:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file
        file_type: Description of file type (for error messages)

    Returns:
        True if valid, False otherwise
    """
    if file_path and not os.path.exists(file_path):
        print(f"❌ Error: {file_type} not found at {file_path}")
        return False
    return True


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode():
    """Run analysis in interactive mode with user prompts."""
    print("\n" + "="*60)
    print("📊 REFUSAL CLASSIFIER - INTERACTIVE ANALYSIS")
    print("="*60)

    print("\nThis will analyze BOTH classifiers:")
    print("  1. Refusal Classifier (3 classes)")
    print("  2. Jailbreak Detector (2 classes)")

    print("\n" + "-"*60)
    print("Model Configuration")
    print("-"*60)

    # Get refusal model path
    print("\nRefusal Classifier Model:")
    print("  Press Enter to use default path")
    print(f"  Default: models/{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt")
    refusal_path = input("  Custom path: ").strip()
    if not refusal_path:
        refusal_path = None

    # Validate refusal model
    if refusal_path and not validate_file_exists(refusal_path, "Refusal model"):
        return

    # Get jailbreak model path
    print("\nJailbreak Detector Model:")
    print("  Press Enter to use default path")
    print(f"  Default: models/{EXPERIMENT_CONFIG['experiment_name']}_jailbreak_best.pt")
    jailbreak_path = input("  Custom path: ").strip()
    if not jailbreak_path:
        jailbreak_path = None

    # Validate jailbreak model
    if jailbreak_path and not validate_file_exists(jailbreak_path, "Jailbreak model"):
        return

    # Get test data path
    print("\nTest Data:")
    print("  Press Enter to use default path")
    print(f"  Default: {os.path.join(data_splits_path, 'test.pkl')}")
    test_data_path = input("  Custom path: ").strip()
    if not test_data_path:
        test_data_path = None

    # Validate test data
    if test_data_path and not validate_file_exists(test_data_path, "Test data"):
        return

    # Confirm and run
    print("\n" + "-"*60)
    print("Ready to analyze:")
    print(f"  Refusal model: {refusal_path or 'default'}")
    print(f"  Jailbreak model: {jailbreak_path or 'default'}")
    print(f"  Test data: {test_data_path or 'default'}")
    print("-"*60)

    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Analysis cancelled.")
        return

    # Run analysis
    runner = ExperimentRunner()

    # If custom test data provided, we need to handle it differently
    if test_data_path:
        print(f"\n⚠️  Note: Custom test data path specified but not yet implemented")
        print(f"    Will use default test data from data/splits/test.pkl")
        print(f"    TODO: Add test data override to ExperimentRunner.analyze_only()")

    runner.analyze_only(refusal_path, jailbreak_path)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # If --auto flag or any arguments provided, use command-line mode
    if args.auto or args.refusal_model or args.jailbreak_model or args.test_data:
        print("\n" + "="*60)
        print("📊 REFUSAL CLASSIFIER - ANALYSIS MODE")
        print("="*60)

        # Validate file paths
        if not validate_file_exists(args.refusal_model, "Refusal model"):
            sys.exit(1)
        if not validate_file_exists(args.jailbreak_model, "Jailbreak model"):
            sys.exit(1)
        if not validate_file_exists(args.test_data, "Test data"):
            sys.exit(1)

        # Show configuration
        print("\nConfiguration:")
        print(f"  Refusal model: {args.refusal_model or 'default'}")
        print(f"  Jailbreak model: {args.jailbreak_model or 'default'}")
        print(f"  Test data: {args.test_data or 'default'}")

        # Warn if custom test data provided
        if args.test_data:
            print(f"\n⚠️  Note: Custom test data path specified but not yet implemented")
            print(f"    Will use default test data from data/splits/test.pkl")
            print(f"    TODO: Add test data override to ExperimentRunner.analyze_only()")

        # Run analysis
        runner = ExperimentRunner()
        runner.analyze_only(args.refusal_model, args.jailbreak_model)

    else:
        # Interactive mode
        interactive_mode()


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-Class Refusal Classifier with RoBERTa
Created on October 28, 2025
@author: ramyalsaffar
"""
