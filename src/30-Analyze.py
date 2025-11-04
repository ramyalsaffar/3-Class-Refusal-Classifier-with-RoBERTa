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

  # Generate PDF reports
  python src/Analyze.py --auto --generate-report
  python src/Analyze.py --auto --generate-report --report-type performance
  python src/Analyze.py --auto --generate-report --report-type interpretability
  python src/Analyze.py --auto --generate-report --report-type executive

  # Combine options
  python src/Analyze.py --refusal-model models/refusal.pt --test-data data/test.pkl --generate-report
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

    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate comprehensive PDF report after analysis'
    )

    parser.add_argument(
        '--report-type',
        type=str,
        choices=['performance', 'interpretability', 'executive', 'all'],
        default='all',
        help='Type of report to generate (default: all)'
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
# REPORT GENERATION
# =============================================================================

def _generate_reports(analysis_results: Dict, report_type: str):
    """
    Generate PDF reports using ReportGenerator.

    Args:
        analysis_results: Analysis results from ExperimentRunner
        report_type: Type of report ('performance', 'interpretability', 'executive', 'all')
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_generator = ReportGenerator(class_names=analysis_results['metadata']['refusal_class_names'])

    # Load visualization figures
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    if report_type in ['performance', 'all']:
        print("\n📊 Generating Performance Report...")

        # Load figures
        cm_fig = plt.figure(figsize=(10, 8))
        cm_img = mpimg.imread(os.path.join(visualizations_path, "confusion_matrix.png"))
        plt.imshow(cm_img)
        plt.axis('off')

        training_curves_fig = plt.figure(figsize=(12, 5))
        # Note: Training curves would need to be loaded from history
        plt.text(0.5, 0.5, "Training curves not available in analysis-only mode",
                ha='center', va='center', fontsize=14)
        plt.axis('off')

        class_dist_fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Class distribution: See per_class_f1.png",
                ha='center', va='center', fontsize=14)
        plt.axis('off')

        # Extract metrics
        metrics = {
            'accuracy': analysis_results['confidence'].get('overall_accuracy', 0),
            'macro_f1': analysis_results['per_model'].get('analysis', {}).get('macro_avg', {}).get('f1-score', 0),
            'weighted_f1': analysis_results['per_model'].get('analysis', {}).get('weighted_avg', {}).get('f1-score', 0),
            'macro_precision': analysis_results['per_model'].get('analysis', {}).get('macro_avg', {}).get('precision', 0),
            'macro_recall': analysis_results['per_model'].get('analysis', {}).get('macro_avg', {}).get('recall', 0),
        }

        # Add per-class metrics
        for i, class_name in enumerate(analysis_results['metadata']['refusal_class_names']):
            class_metrics = analysis_results['per_model'].get(class_name, {})
            metrics[f'class_{i}_precision'] = class_metrics.get('precision', 0)
            metrics[f'class_{i}_recall'] = class_metrics.get('recall', 0)
            metrics[f'class_{i}_f1'] = class_metrics.get('f1-score', 0)
            metrics[f'class_{i}_support'] = class_metrics.get('support', 0)

        output_path = os.path.join(reports_path, f"performance_report_{timestamp}.pdf")
        report_generator.generate_model_performance_report(
            model_name="Refusal Classifier",
            metrics=metrics,
            confusion_matrix_fig=cm_fig,
            training_curves_fig=training_curves_fig,
            class_distribution_fig=class_dist_fig,
            output_path=output_path
        )
        plt.close('all')

    if report_type in ['executive', 'all']:
        print("\n📊 Generating Executive Summary...")

        # Key metrics
        key_metrics = {
            'Overall Accuracy': f"{analysis_results['confidence'].get('overall_accuracy', 0):.2%}",
            'Weighted F1 Score': f"{analysis_results['per_model'].get('analysis', {}).get('weighted_avg', {}).get('f1-score', 0):.4f}",
            'Test Samples': analysis_results['metadata']['num_test_samples'],
            'Avg Confidence': f"{analysis_results['confidence'].get('avg_confidence', 0):.4f}",
        }

        # Performance chart
        perf_fig = plt.figure(figsize=(10, 6))
        perf_img = mpimg.imread(os.path.join(visualizations_path, "per_class_f1.png"))
        plt.imshow(perf_img)
        plt.axis('off')

        # Recommendations
        recommendations = [
            "Model shows strong performance on test set",
            "Monitor confidence distributions in production",
            "Consider retraining if accuracy drops below 85%",
            "Implement A/B testing before major model updates"
        ]

        output_path = os.path.join(reports_path, f"executive_summary_{timestamp}.pdf")
        report_generator.generate_executive_summary(
            model_name="Refusal Classifier",
            key_metrics=key_metrics,
            performance_chart_fig=perf_fig,
            recommendations=recommendations,
            output_path=output_path
        )
        plt.close('all')

    print(f"\n✅ Reports saved to: {reports_path}")


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

    # Ask about report generation
    print("\nReport Generation:")
    generate_report = input("  Generate PDF report? (y/n, default: n): ").strip().lower() == 'y'

    report_type = 'all'
    if generate_report:
        print("  Report types:")
        print("    1. Performance only")
        print("    2. Interpretability only")
        print("    3. Executive summary only")
        print("    4. All reports (default)")
        choice = input("  Select (1-4, default: 4): ").strip()
        report_type_map = {'1': 'performance', '2': 'interpretability', '3': 'executive', '4': 'all'}
        report_type = report_type_map.get(choice, 'all')

    # Confirm and run
    print("\n" + "-"*60)
    print("Ready to analyze:")
    print(f"  Refusal model: {refusal_path or 'default'}")
    print(f"  Jailbreak model: {jailbreak_path or 'default'}")
    print(f"  Test data: {test_data_path or 'default'}")
    print(f"  Generate report: {'Yes (' + report_type + ')' if generate_report else 'No'}")
    print("-"*60)

    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Analysis cancelled.")
        return

    # Run analysis
    runner = ExperimentRunner()

    # Run analysis with custom test data if provided
    # WHY: Allows validation on different test sets without retraining
    if test_data_path:
        print(f"\n✓ Using custom test data: {test_data_path}")

    analysis_results = runner.analyze_only(refusal_path, jailbreak_path, test_data_path)

    # Generate report if requested
    if generate_report and analysis_results:
        print("\n" + "="*60)
        print("📄 GENERATING PDF REPORT")
        print("="*60)
        print(f"\nReport type: {report_type}")
        print(f"Output location: {reports_path}")

        try:
            _generate_reports(analysis_results, report_type)
        except Exception as e:
            print(f"\n❌ Report generation failed: {e}")
            print("   Analysis results are still available in results/ directory")


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
        print(f"  Generate report: {'Yes (' + args.report_type + ')' if args.generate_report else 'No'}")

        # Notify if custom test data provided
        if args.test_data:
            print(f"\n✓ Using custom test data: {args.test_data}")

        # Run analysis
        runner = ExperimentRunner()
        analysis_results = runner.analyze_only(args.refusal_model, args.jailbreak_model, args.test_data)

        # Generate report if requested
        if args.generate_report and analysis_results:
            print("\n" + "="*60)
            print("📄 GENERATING PDF REPORT")
            print("="*60)
            print(f"\nReport type: {args.report_type}")
            print(f"Output location: {reports_path}")

            try:
                _generate_reports(analysis_results, args.report_type)
            except Exception as e:
                print(f"\n❌ Report generation failed: {e}")
                print("   Analysis results are still available in results/ directory")

    else:
        # Interactive mode
        interactive_mode()


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
