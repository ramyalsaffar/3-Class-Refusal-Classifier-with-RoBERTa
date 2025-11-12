# Execute Pipeline
#-----------------
# Main entry point for Dual RoBERTa Classifiers experiments.
# Provides both command-line and interactive interfaces.
#
# ALL MODES USE PHASE 2 METHODOLOGY:
#   - K-fold cross-validation
#   - Statistical hypothesis testing  
#   - Comprehensive error analysis
#
# Usage:
#   python src/Execute.py                      # Interactive mode
#   python src/Execute.py --test               # Quick test (Phase 2)
#   python src/Execute.py --full               # Full experiment (Phase 2)
#   python src/Execute.py --analyze-only       # Analyze existing models
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
            # Quick test with Phase 2 methodology
            print()
            print_banner("QUICK TEST MODE (PHASE 2)", width=60)

            resume_from_checkpoint, start_step = runner._check_and_prompt_for_resume()
            api_keys = runner._get_api_keys()
            runner.pipeline = RefusalPipeline(api_keys, resume_from_checkpoint=resume_from_checkpoint)

            original_total_prompts = DATASET_CONFIG['total_prompts']
            DATASET_CONFIG['total_prompts'] = EXPERIMENT_CONFIG['test_sample_size']

            try:
                with KeepAwake():
                    if start_step == 1:
                        runner.pipeline.run_full_pipeline()
                    else:
                        runner.pipeline.run_partial_pipeline(start_step=start_step)
                DATASET_CONFIG['total_prompts'] = original_total_prompts
                runner.train_with_cross_validation()
            finally:
                DATASET_CONFIG['total_prompts'] = original_total_prompts

        elif sys.argv[1] == '--full':
            # Full experiment with Phase 2 methodology
            print()
            print_banner("FULL EXPERIMENT MODE (PHASE 2)", width=60)

            resume_from_checkpoint, start_step = runner._check_and_prompt_for_resume()
            api_keys = runner._get_api_keys()
            runner.pipeline = RefusalPipeline(api_keys, resume_from_checkpoint=resume_from_checkpoint)

            with KeepAwake():
                if start_step == 1:
                    runner.pipeline.run_full_pipeline()
                else:
                    runner.pipeline.run_partial_pipeline(start_step=start_step)
            runner.train_with_cross_validation()

        elif sys.argv[1] == '--analyze-only':
            refusal_path = sys.argv[2] if len(sys.argv) > 2 else None
            jailbreak_path = sys.argv[3] if len(sys.argv) > 3 else None
            runner.analyze_only(refusal_path, jailbreak_path)

        else:
            print("Usage: python src/Execute.py [--test|--full|--analyze-only]")
            print("\nOptions:")
            print("  --test           Quick test with Phase 2 CV (reduced samples)")
            print("  --full           Full experiment with Phase 2 CV (complete dataset)")
            print("  --analyze-only   Analyze existing models (optionally specify model paths)")
            print("\nExamples:")
            print("  python src/Execute.py --test")
            print("  python src/Execute.py --full")
            print("  python src/Execute.py --analyze-only")
            print("  python src/Execute.py --analyze-only models/exp_refusal_best.pt models/exp_jailbreak_best.pt")

    else:
        # Interactive mode
        print()
        print_banner("🤖 3-CLASS REFUSAL CLASSIFIER WITH ROBERTA", width=60)
        print(f"Environment: {'AWS' if IS_AWS else 'Local (Mac)' if IS_MAC else 'Local'}")
        print(f"Device: {DEVICE}")
        print(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
        print_banner("", width=60)

        print("\n📋 Experiment Launcher")
        print("-" * 40)
        print("1. Quick Test (Phase 2: CV + Hypothesis Testing)")
        print(f"   → {EXPERIMENT_CONFIG['test_sample_size']} prompts, {CROSS_VALIDATION_CONFIG['default_folds']}-fold CV, comprehensive analysis")
        print("\n2. Full Experiment (Phase 2: CV + Hypothesis Testing)")
        print(f"   → {DATASET_CONFIG['total_prompts']} prompts, {CROSS_VALIDATION_CONFIG['default_folds']}-fold CV, comprehensive analysis")
        print("\n3. Analyze Existing Models")
        print("   → Load trained models and run analysis on test data")
        print("\n4. Exit")

        choice = input("\nSelect mode (1-4): ")

        if choice == '1':
            # Quick test with Phase 2 methodology
            print()
            print_banner("QUICK TEST MODE (PHASE 2)", width=60)
            print(f"Using {EXPERIMENT_CONFIG['test_sample_size']} prompts with full Phase 2 methodology:")
            print(f"  ✓ Data collection and labeling")
            print(f"  ✓ {CROSS_VALIDATION_CONFIG['default_folds']}-fold cross-validation")
            print(f"  ✓ Statistical hypothesis testing")
            print(f"  ✓ Comprehensive error analysis")
            print_banner("", width=60)

            # Check for existing checkpoints and prompt user
            resume_from_checkpoint, start_step = runner._check_and_prompt_for_resume()

            # Get API keys
            api_keys = runner._get_api_keys()

            # Initialize pipeline with quick test mode
            runner.pipeline = RefusalPipeline(api_keys, resume_from_checkpoint=resume_from_checkpoint)

            # Temporarily override dataset config for quick testing
            original_total_prompts = DATASET_CONFIG['total_prompts']
            DATASET_CONFIG['total_prompts'] = EXPERIMENT_CONFIG['test_sample_size']

            try:
                with KeepAwake():
                    # Run data collection pipeline
                    if start_step == 1:
                        runner.pipeline.run_full_pipeline()
                    else:
                        runner.pipeline.run_partial_pipeline(start_step=start_step)
                    
                # Restore original config before CV
                DATASET_CONFIG['total_prompts'] = original_total_prompts
                
                # Run Phase 2 cross-validation with hypothesis testing
                print()
                print_banner("PHASE 2: CROSS-VALIDATION + HYPOTHESIS TESTING", width=60)
                runner.train_with_cross_validation()
                
            finally:
                # Ensure config is restored
                DATASET_CONFIG['total_prompts'] = original_total_prompts
            
            print()
            print_banner("QUICK TEST COMPLETE", width=60, char='#')
            print("✅ All results saved to:")
            print(f"   - Models: {models_path}")
            print(f"   - Results: {results_path}")
            print(f"   - Visualizations: {visualizations_path}")
            print_banner("", width=60, char='#')
            
        elif choice == '2':
            # Full experiment with Phase 2 methodology
            print()
            print_banner("FULL EXPERIMENT MODE (PHASE 2)", width=60)
            print(f"Using {DATASET_CONFIG['total_prompts']} prompts with full Phase 2 methodology:")
            print(f"  ✓ Data collection and labeling")
            print(f"  ✓ {CROSS_VALIDATION_CONFIG['default_folds']}-fold cross-validation")
            print(f"  ✓ Statistical hypothesis testing")
            print(f"  ✓ Comprehensive error analysis")
            print_banner("", width=60)

            # Check for existing checkpoints and prompt user
            resume_from_checkpoint, start_step = runner._check_and_prompt_for_resume()

            # Get API keys
            api_keys = runner._get_api_keys()

            # Run pipeline with Phase 2 methodology
            runner.pipeline = RefusalPipeline(api_keys, resume_from_checkpoint=resume_from_checkpoint)

            with KeepAwake():
                # Run data collection pipeline
                print(f"\n🔍 DEBUG: start_step = {start_step}")
                if start_step == 1:
                    print("🔍 DEBUG: Calling run_full_pipeline()")
                    runner.pipeline.run_full_pipeline()
                else:
                    print(f"🔍 DEBUG: Calling run_partial_pipeline(start_step={start_step})")
                    runner.pipeline.run_partial_pipeline(start_step=start_step)
            
            # Run Phase 2 cross-validation with hypothesis testing
            print()
            print_banner("PHASE 2: CROSS-VALIDATION + HYPOTHESIS TESTING", width=60)
            runner.train_with_cross_validation()
            
            print()
            print_banner("FULL EXPERIMENT COMPLETE", width=60, char='#')
            print("✅ All results saved to:")
            print(f"   - Models: {models_path}")
            print(f"   - Results: {results_path}")
            print(f"   - Visualizations: {visualizations_path}")
            print()
            print("💡 TIP: Generate PDF reports with:")
            print("   python src/Analyze.py --auto --generate-report")
            print_banner("", width=60, char='#')
            
        elif choice == '3':
            # Analyze existing models
            print("\n📊 Analyze Existing Models")
            print("-" * 40)
            print("Model paths (press Enter for default):")
            refusal_path = input("  Refusal model path: ").strip()
            jailbreak_path = input("  Jailbreak model path: ").strip()
            
            if not refusal_path:
                refusal_path = None
            if not jailbreak_path:
                jailbreak_path = None
                
            result = runner.analyze_only(refusal_path, jailbreak_path)
            
            if result:
                print()
                print_banner("ANALYSIS COMPLETE", width=60)
                print("✅ Analysis results saved")
                print()
                print("💡 TIP: Generate PDF reports with:")
                print("   python src/Analyze.py --auto --generate-report")
                print_banner("", width=60)
                
        elif choice == '4':
            print("\n👋 Exiting...")
            
        else:
            print("\n❌ Invalid choice. Exiting...")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-Class Refusal Classifier with RoBERTa
Created on October 28, 2025
@author: ramyalsaffar
"""
