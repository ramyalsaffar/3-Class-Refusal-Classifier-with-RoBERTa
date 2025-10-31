#!/usr/bin/env python3
# Execute Pipeline
#-----------------
# Main entry point for 3-Class Refusal Classifier experiments.
# Provides both command-line and interactive interfaces.
#
# Usage:
#   python src/Execute.py                # Interactive mode
#   python src/Execute.py --test         # Quick test
#   python src/Execute.py --full         # Full experiment
#   python src/Execute.py --train-only   # Train only
#   python src/Execute.py --analyze-only [model_path]  # Analyze only
#
###############################################################################


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Initialize the experiment runner
    runner = ExperimentRunner()

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            runner.quick_test()

        elif sys.argv[1] == '--full':
            runner.full_experiment()

        elif sys.argv[1] == '--train-only':
            runner.train_only()

        elif sys.argv[1] == '--analyze-only':
            refusal_path = sys.argv[2] if len(sys.argv) > 2 else None
            jailbreak_path = sys.argv[3] if len(sys.argv) > 3 else None
            runner.analyze_only(refusal_path, jailbreak_path)

        else:
            print("Usage: python src/Execute.py [--test|--full|--train-only|--analyze-only]")
            print("\nOptions:")
            print("  --test           Run quick test with reduced samples")
            print("  --full           Run full experiment as configured")
            print("  --train-only     Train on existing data")
            print("  --analyze-only   Analyze existing models (optionally specify refusal and jailbreak model paths)")
            print("\nExamples:")
            print("  python src/Execute.py --full")
            print("  python src/Execute.py --analyze-only")
            print("  python src/Execute.py --analyze-only models/exp_refusal_best.pt models/exp_jailbreak_best.pt")

    else:
        # Interactive mode
        print("\n" + "="*60)
        print("🤖 3-CLASS REFUSAL CLASSIFIER WITH ROBERTA")
        print("="*60)
        print(f"Environment: {'AWS' if IS_AWS else 'Local (Mac)' if IS_MAC else 'Local'}")
        print(f"Device: {DEVICE}")
        print(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
        print("="*60)

        print("\n📋 Experiment Launcher")
        print("-" * 40)
        print("1. Quick Test (reduced samples)")
        print("2. Full Experiment (complete pipeline)")
        print("3. Train Only (use existing data)")
        print("4. Analyze Only (use existing model)")
        print("5. Exit")

        choice = input("\nSelect mode (1-5): ")

        if choice == '1':
            runner.quick_test()
        elif choice == '2':
            runner.full_experiment()
        elif choice == '3':
            runner.train_only()
        elif choice == '4':
            print("\nModel paths (press Enter for default):")
            refusal_path = input("  Refusal model path: ").strip()
            jailbreak_path = input("  Jailbreak model path: ").strip()
            if not refusal_path:
                refusal_path = None
            if not jailbreak_path:
                jailbreak_path = None
            runner.analyze_only(refusal_path, jailbreak_path)
        elif choice == '5':
            print("Exiting...")
        else:
            print("Invalid choice. Exiting...")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-Class Refusal Classifier with RoBERTa
Created on October 28, 2025
@author: ramyalsaffar
"""
