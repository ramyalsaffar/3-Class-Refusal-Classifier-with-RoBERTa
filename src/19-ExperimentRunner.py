# ExperimentRunner Module
#------------------------
# Manages different experiment execution modes.
# Adapted from Alignment Tax project.
# All imports are in 00-Imports.py
###############################################################################


class ExperimentRunner:
    """Manages different experiment execution modes."""

    def __init__(self):
        self.pipeline = None
        self.api_keys = None

    def _get_api_keys(self) -> Dict[str, str]:
        """
        Get API keys based on environment.

        Returns:
            Dictionary with keys: 'openai', 'anthropic', 'google'
        """
        if IS_AWS and AWS_AVAILABLE:
            # Use AWS Secrets Manager
            print("🔐 Retrieving API keys from AWS Secrets Manager...")
            try:
                secrets_handler = SecretsHandler(region=AWS_CONFIG['region'])

                return {
                    'openai': secrets_handler.get_api_key(AWS_CONFIG['secrets']['openai']),
                    'anthropic': secrets_handler.get_api_key(AWS_CONFIG['secrets']['anthropic']),
                    'google': secrets_handler.get_api_key(AWS_CONFIG['secrets']['google'])
                }
            except Exception as e:
                print(f"❌ Error retrieving secrets from AWS: {e}")
                print("Falling back to manual input...")

        # Local mode: Try .env first, then manual input
        print("🔑 API Key Configuration")
        print("-" * 40)

        # Try loading from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✓ Loaded .env file")
        except:
            pass

        # Get or prompt for OpenAI key
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("\nOpenAI API Key not found in .env")
            openai_key = getpass.getpass("Enter OpenAI API Key: ")

        # Get or prompt for Anthropic key
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_key:
            print("\nAnthropic API Key not found in .env")
            anthropic_key = getpass.getpass("Enter Anthropic API Key: ")

        # Get or prompt for Google key
        google_key = os.getenv('GOOGLE_API_KEY')
        if not google_key:
            print("\nGoogle API Key not found in .env")
            google_key = getpass.getpass("Enter Google API Key: ")

        print("\n✓ API keys configured")

        return {
            'openai': openai_key,
            'anthropic': anthropic_key,
            'google': google_key
        }

    def _print_experiment_header(self, mode_name: str, description: str = ""):
        """Print formatted experiment header."""
        print("\n" + "="*60)
        print(f"{mode_name.upper()}")
        print("="*60)
        if description:
            print(description)
            print("="*60)

    def quick_test(self):
        """Run quick test with reduced samples."""
        self._print_experiment_header(
            "Quick Test Mode",
            f"Testing with {EXPERIMENT_CONFIG['test_sample_size']} samples per category"
        )

        # Get API keys
        api_keys = self._get_api_keys()

        # TODO: Implement reduced dataset logic
        # For now, run full pipeline (can be optimized later)
        self.pipeline = RefusalPipeline(api_keys)

        print("\n⚠️  Note: Quick test mode runs full pipeline")
        print("Future enhancement: Implement sampling for faster testing")

        self.pipeline.run_full_pipeline()

    def full_experiment(self):
        """Run full experiment as configured."""
        self._print_experiment_header(
            "Full Experiment Mode",
            f"Running complete pipeline with {DATASET_CONFIG['total_prompts']} prompts"
        )

        # Get API keys
        api_keys = self._get_api_keys()

        # Run pipeline
        self.pipeline = RefusalPipeline(api_keys)
        self.pipeline.run_full_pipeline()

    def train_only(self):
        """Train only (assumes data already collected)."""
        self._print_experiment_header(
            "Train Only Mode",
            "Loading existing data and training classifier"
        )

        # Check if data exists
        labeled_data_path = os.path.join(data_processed_path, "labeled_responses.pkl")
        if not os.path.exists(labeled_data_path):
            print(f"❌ Error: Labeled data not found at {labeled_data_path}")
            print("Please run data collection first or use --full mode")
            return

        # Load data
        print(f"\nLoading data from {labeled_data_path}...")
        labeled_df = pd.read_pickle(labeled_data_path)
        print(f"✓ Loaded {len(labeled_df)} labeled samples")

        # Get API keys (only need OpenAI for potential adversarial testing)
        api_keys = {'openai': os.getenv('OPENAI_API_KEY') or getpass.getpass("OpenAI API Key (for analysis): ")}

        # Initialize pipeline
        self.pipeline = RefusalPipeline(api_keys)

        # Run training and analysis
        train_loader, val_loader, test_loader, test_df = self.pipeline.prepare_datasets(labeled_df)
        history = self.pipeline.train_classifier(train_loader, val_loader)
        analysis_results = self.pipeline.run_analyses(test_df)
        self.pipeline.generate_visualizations(history, analysis_results)

        print("\n✅ Training and analysis complete")

    def analyze_only(self, model_path: str = None):
        """Analysis only (load existing model)."""
        self._print_experiment_header(
            "Analysis Only Mode",
            "Loading trained model and running analysis"
        )

        # Check if model exists
        if model_path is None:
            model_path = os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_best.pt")

        if not os.path.exists(model_path):
            print(f"❌ Error: Model not found at {model_path}")
            print("Please provide model path or train first")
            return

        # Check if test data exists
        test_data_path = os.path.join(data_splits_path, "test.pkl")
        if not os.path.exists(test_data_path):
            print(f"❌ Error: Test data not found at {test_data_path}")
            print("Please run full pipeline first")
            return

        # Load test data
        print(f"\nLoading test data from {test_data_path}...")
        test_df = pd.read_pickle(test_data_path)
        print(f"✓ Loaded {len(test_df)} test samples")

        # Load model
        print(f"\nLoading model from {model_path}...")
        model = RefusalClassifier()
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        print(f"✓ Model loaded (Best Val F1: {checkpoint['best_val_f1']:.4f})")

        # Initialize tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

        # Get API keys for adversarial testing
        api_keys = {'openai': os.getenv('OPENAI_API_KEY') or getpass.getpass("OpenAI API Key (for adversarial): ")}

        # Run analyses
        print("\n" + "="*60)
        print("RUNNING ANALYSES")
        print("="*60)

        analysis_results = {}

        # Per-model analysis
        per_model_analyzer = PerModelAnalyzer(model, tokenizer, DEVICE)
        per_model_results = per_model_analyzer.analyze(test_df)
        per_model_analyzer.save_results(
            per_model_results,
            os.path.join(results_path, "per_model_analysis.json")
        )
        analysis_results['per_model'] = per_model_results

        # Confidence analysis
        confidence_analyzer = ConfidenceAnalyzer(model, tokenizer, DEVICE)
        conf_results, preds, labels, confidences = confidence_analyzer.analyze(test_df)
        confidence_analyzer.save_results(
            conf_results,
            os.path.join(results_path, "confidence_analysis.json")
        )
        analysis_results['confidence'] = conf_results
        analysis_results['predictions'] = {
            'preds': preds,
            'labels': labels,
            'confidences': confidences
        }

        # Adversarial testing
        adversarial_tester = AdversarialTester(model, tokenizer, DEVICE, api_keys['openai'])
        adv_results = adversarial_tester.test_robustness(test_df)
        adversarial_tester.save_results(
            adv_results,
            os.path.join(results_path, "adversarial_testing.json")
        )
        analysis_results['adversarial'] = adv_results

        # Attention visualization
        print("\n--- Attention Visualization ---")
        attention_viz = AttentionVisualizer(model, tokenizer, DEVICE)
        attention_results = attention_viz.analyze_samples(
            test_df,
            num_samples=INTERPRETABILITY_CONFIG['attention_samples_per_class']
        )
        analysis_results['attention'] = attention_results

        # SHAP analysis (if enabled)
        if INTERPRETABILITY_CONFIG['shap_enabled']:
            print("\n--- SHAP Analysis ---")
            try:
                shap_analyzer = ShapAnalyzer(model, tokenizer, DEVICE)
                shap_results = shap_analyzer.analyze_samples(
                    test_df,
                    num_samples=INTERPRETABILITY_CONFIG['shap_samples']
                )
                analysis_results['shap'] = shap_results
            except ImportError:
                print("⚠️  SHAP not installed - skipping SHAP analysis")
                print("   Install with: pip install shap")
                analysis_results['shap'] = None
            except Exception as e:
                print(f"⚠️  SHAP analysis failed: {e}")
                analysis_results['shap'] = None
        else:
            print("\n--- SHAP Analysis (Disabled) ---")
            analysis_results['shap'] = None

        # Generate visualizations
        visualizer = Visualizer()

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        visualizer.plot_confusion_matrix(
            cm,
            os.path.join(visualizations_path, "confusion_matrix.png")
        )

        # Per-class F1
        per_class_f1 = {}
        for class_name, class_data in analysis_results['per_model'].items():
            if class_name != 'analysis':
                for cls, f1 in class_data['f1_per_class'].items():
                    if cls not in per_class_f1:
                        per_class_f1[cls] = []
                    per_class_f1[cls].append(f1)

        avg_per_class_f1 = {cls: np.mean(scores) for cls, scores in per_class_f1.items()}
        visualizer.plot_per_class_f1(
            avg_per_class_f1,
            os.path.join(visualizations_path, "per_class_f1.png")
        )

        # Per-model F1
        visualizer.plot_per_model_f1(
            analysis_results['per_model'],
            os.path.join(visualizations_path, "per_model_f1.png")
        )

        # Adversarial robustness
        visualizer.plot_adversarial_robustness(
            analysis_results['adversarial'],
            os.path.join(visualizations_path, "adversarial_robustness.png")
        )

        # Confidence distributions
        visualizer.plot_confidence_distributions(
            labels,
            confidences,
            os.path.join(visualizations_path, "confidence_distributions.png")
        )

        print("\n✅ Analysis complete")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
