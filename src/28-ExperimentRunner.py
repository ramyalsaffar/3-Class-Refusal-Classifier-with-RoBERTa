# ExperimentRunner Module
#------------------------
# Manages different experiment execution modes.
# Adapted from Alignment Tax project.
# All imports are in 01-Imports.py
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
        """
        Run quick test with reduced samples.

        WHY: Allows rapid prototyping and testing without running full pipeline.
        Samples a subset of data at each stage to validate end-to-end workflow.
        """
        self._print_experiment_header(
            "Quick Test Mode",
            f"Testing with {EXPERIMENT_CONFIG['test_sample_size']} samples per category"
        )

        # Get API keys
        api_keys = self._get_api_keys()

        # Initialize pipeline with quick test mode
        self.pipeline = RefusalPipeline(api_keys)

        # Override pipeline behavior for quick test
        print("\n🚀 Quick test mode active - using reduced dataset")
        print(f"   Test sample size: {EXPERIMENT_CONFIG['test_sample_size']}")
        print(f"   Reduced total prompts: {EXPERIMENT_CONFIG['test_sample_size']}")

        # Temporarily override dataset config for quick testing
        # WHY: Reduce total prompts to speed up data collection and testing
        original_total_prompts = DATASET_CONFIG['total_prompts']
        DATASET_CONFIG['total_prompts'] = EXPERIMENT_CONFIG['test_sample_size']

        try:
            self.pipeline.run_full_pipeline()
        finally:
            # Restore original config
            # WHY: Prevent config pollution affecting other modes
            DATASET_CONFIG['total_prompts'] = original_total_prompts

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
            "Loading existing data and training BOTH classifiers"
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

        # Get API keys (only need OpenAI for adversarial testing)
        api_keys = {'openai': os.getenv('OPENAI_API_KEY') or getpass.getpass("OpenAI API Key (for analysis): ")}

        # Initialize pipeline
        self.pipeline = RefusalPipeline(api_keys)

        # Prepare datasets (returns dict with 'refusal' and 'jailbreak' keys)
        datasets = self.pipeline.prepare_datasets(labeled_df)

        # Train refusal classifier
        refusal_history = self.pipeline.train_refusal_classifier(
            datasets['refusal']['train_loader'],
            datasets['refusal']['val_loader']
        )

        # Train jailbreak detector
        jailbreak_history = self.pipeline.train_jailbreak_detector(
            datasets['jailbreak']['train_loader'],
            datasets['jailbreak']['val_loader']
        )

        # Run analyses (uses test_df from refusal datasets - same for both)
        analysis_results = self.pipeline.run_analyses(datasets['refusal']['test_df'])

        # Generate visualizations (using both histories)
        self.pipeline.generate_visualizations(refusal_history, jailbreak_history, analysis_results)

        print("\n✅ Training and analysis complete (BOTH classifiers trained)")

    def analyze_only(self, refusal_model_path: str = None, jailbreak_model_path: str = None, test_data_path: str = None):
        """
        Analysis only (load existing models for BOTH classifiers).

        Args:
            refusal_model_path: Path to trained refusal classifier (default: models/{experiment_name}_refusal_best.pt)
            jailbreak_model_path: Path to trained jailbreak detector (default: models/{experiment_name}_jailbreak_best.pt)
            test_data_path: Path to test data (default: data/splits/test.pkl)

        WHY: Allows rerunning analysis with different test sets or on different model checkpoints
        without retraining. Useful for validating model on new data or comparing model versions.
        """
        self._print_experiment_header(
            "Analysis Only Mode",
            "Loading trained models and running analysis (BOTH classifiers)"
        )

        # Determine model paths
        if refusal_model_path is None:
            refusal_model_path = os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt")
        if jailbreak_model_path is None:
            jailbreak_model_path = os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_jailbreak_best.pt")

        # Check if models exist
        if not os.path.exists(refusal_model_path):
            print(f"❌ Error: Refusal model not found at {refusal_model_path}")
            print("Please train models first or provide correct paths")
            return
        if not os.path.exists(jailbreak_model_path):
            print(f"❌ Error: Jailbreak model not found at {jailbreak_model_path}")
            print("Please train models first or provide correct paths")
            return

        # Determine test data path
        if test_data_path is None:
            test_data_path = os.path.join(data_splits_path, "test.pkl")

        # Check if test data exists
        if not os.path.exists(test_data_path):
            print(f"❌ Error: Test data not found at {test_data_path}")
            print("Please run full pipeline first or provide valid test data path")
            return

        # Load test data
        print(f"\nLoading test data from {test_data_path}...")
        test_df = pd.read_pickle(test_data_path)
        print(f"✓ Loaded {len(test_df)} test samples")

        # Validate test data has required columns
        # WHY: Ensure test data contains both refusal and jailbreak labels for dual-task analysis
        required_columns = ['response', 'refusal_label', 'jailbreak_label']
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            print(f"❌ Error: Test data missing required columns: {missing_columns}")
            print(f"   Required: {required_columns}")
            print(f"   Found: {list(test_df.columns)}")
            return

        # Initialize tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

        # Load refusal classifier
        print(f"\nLoading refusal classifier from {refusal_model_path}...")
        refusal_model = RefusalClassifier(num_classes=len(CLASS_NAMES))
        refusal_checkpoint = torch.load(refusal_model_path, map_location=DEVICE)
        refusal_model.load_state_dict(refusal_checkpoint['model_state_dict'])
        refusal_model = refusal_model.to(DEVICE)
        refusal_model.eval()  # WHY: Set to evaluation mode (disables dropout, sets BatchNorm to eval)
        print(f"✓ Refusal classifier loaded (Best Val F1: {refusal_checkpoint['best_val_f1']:.4f})")

        # Load jailbreak detector
        print(f"\nLoading jailbreak detector from {jailbreak_model_path}...")
        jailbreak_model = JailbreakDetector(num_classes=len(JAILBREAK_CLASS_NAMES))
        jailbreak_checkpoint = torch.load(jailbreak_model_path, map_location=DEVICE)
        jailbreak_model.load_state_dict(jailbreak_checkpoint['model_state_dict'])
        jailbreak_model = jailbreak_model.to(DEVICE)
        jailbreak_model.eval()  # WHY: Set to evaluation mode (disables dropout, sets BatchNorm to eval)
        print(f"✓ Jailbreak detector loaded (Best Val F1: {jailbreak_checkpoint['best_val_f1']:.4f})")

        # Get API keys for adversarial testing
        api_keys = {'openai': os.getenv('OPENAI_API_KEY') or getpass.getpass("OpenAI API Key (for adversarial): ")}

        # Run analyses
        print("\n" + "="*60)
        print("RUNNING ANALYSES (BOTH CLASSIFIERS)")
        print("="*60)

        analysis_results = {}

        # ═══════════════════════════════════════════════════════
        # REFUSAL CLASSIFIER ANALYSIS
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("REFUSAL CLASSIFIER ANALYSIS")
        print("="*60)

        # Per-model analysis
        print("\n--- Per-Model Analysis ---")
        per_model_analyzer = PerModelAnalyzer(refusal_model, tokenizer, DEVICE)
        per_model_results = per_model_analyzer.analyze(test_df)
        per_model_analyzer.save_results(
            per_model_results,
            os.path.join(results_path, "per_model_analysis.json")
        )
        analysis_results['per_model'] = per_model_results

        # Confidence analysis
        print("\n--- Confidence Analysis ---")
        confidence_analyzer = ConfidenceAnalyzer(refusal_model, tokenizer, DEVICE)
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

        # Power law analysis
        print("\n--- Power Law Analysis ---")
        power_law_analyzer = PowerLawAnalyzer(refusal_model, tokenizer, DEVICE, class_names=CLASS_NAMES)
        power_law_results = power_law_analyzer.analyze_all(
            test_df,
            np.array(preds),
            np.array(confidences),
            output_dir=visualizations_path
        )
        analysis_results['power_law'] = power_law_results

        # Adversarial testing
        print("\n--- Adversarial Testing ---")
        adversarial_tester = AdversarialTester(refusal_model, tokenizer, DEVICE, api_keys['openai'])
        adv_results = adversarial_tester.test_robustness(test_df)
        adversarial_tester.save_results(
            adv_results,
            os.path.join(results_path, "adversarial_testing.json")
        )
        analysis_results['adversarial'] = adv_results

        # Attention visualization
        print("\n--- Attention Visualization ---")
        attention_viz = AttentionVisualizer(refusal_model, tokenizer, DEVICE, class_names=CLASS_NAMES)
        attention_results = attention_viz.analyze_samples(
            test_df,
            num_samples=INTERPRETABILITY_CONFIG['attention_samples_per_class']
        )
        analysis_results['attention'] = attention_results

        # SHAP analysis (if enabled)
        if INTERPRETABILITY_CONFIG['shap_enabled']:
            print("\n--- SHAP Analysis ---")
            try:
                shap_analyzer = ShapAnalyzer(refusal_model, tokenizer, DEVICE, class_names=CLASS_NAMES)
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

        # ═══════════════════════════════════════════════════════
        # JAILBREAK DETECTOR ANALYSIS
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("JAILBREAK DETECTOR ANALYSIS")
        print("="*60)

        jailbreak_analyzer = JailbreakAnalysis(
            jailbreak_model,
            refusal_model,
            tokenizer,
            DEVICE
        )
        jailbreak_results = jailbreak_analyzer.analyze_full(test_df)
        jailbreak_analyzer.save_results(
            jailbreak_results,
            os.path.join(results_path, "jailbreak_analysis.json")
        )
        analysis_results['jailbreak'] = jailbreak_results

        # Power law analysis for jailbreak detector
        print("\n--- Power Law Analysis (Jailbreak) ---")
        jailbreak_class_names = ["Jailbreak Failed", "Jailbreak Succeeded"]
        jailbreak_power_law_analyzer = PowerLawAnalyzer(
            jailbreak_model, tokenizer, DEVICE, class_names=jailbreak_class_names
        )
        jailbreak_power_law_results = jailbreak_power_law_analyzer.analyze_all(
            test_df,
            jailbreak_results['predictions']['preds'],
            jailbreak_results['predictions']['confidences'],
            output_dir=visualizations_path
        )
        analysis_results['jailbreak_power_law'] = jailbreak_power_law_results

        # Refusal-Jailbreak Correlation Analysis
        print("\n--- Refusal-Jailbreak Correlation Analysis ---")
        correlation_analyzer = RefusalJailbreakCorrelationAnalyzer(
            refusal_preds=preds,
            jailbreak_preds=jailbreak_results['predictions']['preds'],
            refusal_labels=labels,
            jailbreak_labels=jailbreak_results['predictions']['labels'],
            texts=test_df['response'].tolist(),
            refusal_class_names=CLASS_NAMES,
            jailbreak_class_names=jailbreak_class_names
        )
        correlation_results = correlation_analyzer.analyze_full()
        correlation_analyzer.save_results(
            correlation_results,
            os.path.join(results_path, "correlation_analysis.pkl")
        )
        correlation_analyzer.visualize_correlation(output_dir=visualizations_path)
        analysis_results['correlation'] = correlation_results

        # Generate visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

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

        # Save structured analysis results for ReportGenerator
        print("\n" + "="*60)
        print("SAVING STRUCTURED ANALYSIS RESULTS")
        print("="*60)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'per_model': analysis_results['per_model'],
            'confidence': analysis_results['confidence'],
            'power_law': analysis_results['power_law'],
            'adversarial': analysis_results['adversarial'],
            'attention': analysis_results['attention'],
            'shap': analysis_results['shap'],
            'jailbreak': analysis_results['jailbreak'],
            'jailbreak_power_law': analysis_results['jailbreak_power_law'],
            'predictions': {
                'preds': [int(p) for p in preds],
                'labels': [int(l) for l in labels],
                'confidences': [float(c) for c in confidences]
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_test_samples': len(test_df),
                'refusal_class_names': CLASS_NAMES,
                'jailbreak_class_names': ["Jailbreak Failed", "Jailbreak Succeeded"]
            }
        }

        # Save as JSON
        analysis_results_path = os.path.join(results_path, "analysis_results_structured.json")
        with open(analysis_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✓ Saved structured results to: {analysis_results_path}")

        # Also save confusion matrix figure data for reports
        import matplotlib.pyplot as plt
        cm = confusion_matrix(labels, preds)
        cm_fig = plt.figure(figsize=(10, 8))
        visualizer.plot_confusion_matrix(cm, None)  # Plot but don't save yet
        plt.savefig(os.path.join(visualizations_path, "confusion_matrix_for_report.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()

        print("\n✅ Analysis complete (BOTH classifiers analyzed)")

        # Return results for potential use by ReportGenerator
        return serializable_results


    def train_with_cross_validation(self, k_folds: int = 5):
        """
        Train with cross-validation (Phase 2 feature).

        Runs:
        1. Hypothesis testing on dataset
        2. K-fold cross-validation
        3. Final model training on full train+val
        4. Error analysis on test set

        Args:
            k_folds: Number of cross-validation folds (default: 5)
        """
        self._print_experiment_header(
            "Cross-Validation Mode (Phase 2)",
            f"Training with {k_folds}-fold cross-validation + comprehensive error analysis"
        )

        # Check if labeled data exists
        labeled_data_path = os.path.join(data_processed_path, "labeled_responses.pkl")
        if not os.path.exists(labeled_data_path):
            print(f"❌ Error: Labeled data not found at {labeled_data_path}")
            print("Please run data collection first or use --full mode")
            return

        # Load data
        print(f"\nLoading data from {labeled_data_path}...")
        labeled_df = pd.read_pickle(labeled_data_path)
        print(f"✓ Loaded {len(labeled_df)} labeled samples")

        # Initialize tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

        # ═══════════════════════════════════════════════════════
        # REFUSAL CLASSIFIER: CV + HYPOTHESIS TESTING
        # ═══════════════════════════════════════════════════════

        print(f"\n{'#'*60}")
        print("REFUSAL CLASSIFIER: CROSS-VALIDATION")
        print(f"{'#'*60}\n")

        # Prepare refusal dataset
        refusal_texts = labeled_df['response'].tolist()
        refusal_labels = labeled_df['refusal_label'].tolist()  # Fixed: was 'refusal_class'
        refusal_dataset = ClassificationDataset(refusal_texts, refusal_labels, tokenizer)

        # Step 1: Hypothesis testing on refusal dataset
        print(f"\n{'='*60}")
        print("STEP 1: STATISTICAL HYPOTHESIS TESTING (REFUSAL)")
        print(f"{'='*60}\n")

        hypothesis_tester = HypothesisTester(class_names=CLASS_NAMES, alpha=0.05)
        refusal_stats = hypothesis_tester.analyze_dataset_statistics(
            refusal_dataset,
            task_type='refusal'
        )
        hypothesis_tester.save_results()
        hypothesis_tester.generate_statistical_report()

        # Step 2: Cross-validation on refusal classifier
        print(f"\n{'='*60}")
        print(f"STEP 2: {k_folds}-FOLD CROSS-VALIDATION (REFUSAL)")
        print(f"{'='*60}\n")

        refusal_cv_results = train_with_cross_validation(
            full_dataset=refusal_dataset,
            model_class=RefusalClassifier,
            k_folds=k_folds,
            test_split=0.2,
            class_names=CLASS_NAMES,
            save_final_model=True,
            final_model_path=os.path.join(
                models_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt"
            )
        )

        # ═══════════════════════════════════════════════════════
        # JAILBREAK DETECTOR: CV + HYPOTHESIS TESTING
        # ═══════════════════════════════════════════════════════

        print(f"\n{'#'*60}")
        print("JAILBREAK DETECTOR: CROSS-VALIDATION")
        print(f"{'#'*60}\n")

        # Prepare jailbreak dataset
        jailbreak_texts = labeled_df['response'].tolist()
        jailbreak_labels = labeled_df['jailbreak_label'].tolist()  # Fixed: was 'jailbreak_class'
        jailbreak_dataset = ClassificationDataset(jailbreak_texts, jailbreak_labels, tokenizer)

        # Step 1: Hypothesis testing on jailbreak dataset
        print(f"\n{'='*60}")
        print("STEP 1: STATISTICAL HYPOTHESIS TESTING (JAILBREAK)")
        print(f"{'='*60}\n")

        jailbreak_hypothesis_tester = HypothesisTester(class_names=JAILBREAK_CLASS_NAMES, alpha=0.05)
        jailbreak_stats = jailbreak_hypothesis_tester.analyze_dataset_statistics(
            jailbreak_dataset,
            task_type='jailbreak'
        )
        jailbreak_hypothesis_tester.save_results()
        jailbreak_hypothesis_tester.generate_statistical_report()

        # Step 2: Cross-validation on jailbreak detector
        print(f"\n{'='*60}")
        print(f"STEP 2: {k_folds}-FOLD CROSS-VALIDATION (JAILBREAK)")
        print(f"{'='*60}\n")

        jailbreak_cv_results = train_with_cross_validation(
            full_dataset=jailbreak_dataset,
            model_class=JailbreakDetector,
            k_folds=k_folds,
            test_split=0.2,
            class_names=JAILBREAK_CLASS_NAMES,
            save_final_model=True,
            final_model_path=os.path.join(
                models_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_jailbreak_best.pt"
            )
        )

        # ═══════════════════════════════════════════════════════
        # ERROR ANALYSIS (BOTH CLASSIFIERS)
        # ═══════════════════════════════════════════════════════

        print(f"\n{'#'*60}")
        print("COMPREHENSIVE ERROR ANALYSIS")
        print(f"{'#'*60}\n")

        # Load trained models
        refusal_model = RefusalClassifier(num_classes=len(CLASS_NAMES))
        refusal_checkpoint = torch.load(
            os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt"),
            map_location=DEVICE
        )
        refusal_model.load_state_dict(refusal_checkpoint['model_state_dict'])
        refusal_model = refusal_model.to(DEVICE)
        refusal_model.eval()  # WHY: Set to evaluation mode (disables dropout, sets BatchNorm to eval)

        jailbreak_model = JailbreakDetector(num_classes=len(JAILBREAK_CLASS_NAMES))
        jailbreak_checkpoint = torch.load(
            os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_jailbreak_best.pt"),
            map_location=DEVICE
        )
        jailbreak_model.load_state_dict(jailbreak_checkpoint['model_state_dict'])
        jailbreak_model = jailbreak_model.to(DEVICE)
        jailbreak_model.eval()  # WHY: Set to evaluation mode (disables dropout, sets BatchNorm to eval)

        # Create test datasets from CV results using saved test indices
        # WHY: Use actual test indices from stratified split, not last N samples
        refusal_test_idx = refusal_cv_results['split_info']['test_indices']
        jailbreak_test_idx = jailbreak_cv_results['split_info']['test_indices']

        # Reconstruct test datasets using proper indices
        refusal_test_texts = [refusal_texts[i] for i in refusal_test_idx]
        refusal_test_labels = refusal_cv_results['test_results']['labels']
        refusal_test_dataset = ClassificationDataset(refusal_test_texts, refusal_test_labels, tokenizer)

        jailbreak_test_texts = [jailbreak_texts[i] for i in jailbreak_test_idx]
        jailbreak_test_labels = jailbreak_cv_results['test_results']['labels']
        jailbreak_test_dataset = ClassificationDataset(jailbreak_test_texts, jailbreak_test_labels, tokenizer)

        # Run error analysis on refusal classifier
        print(f"\n{'='*60}")
        print("ERROR ANALYSIS: REFUSAL CLASSIFIER")
        print(f"{'='*60}\n")

        refusal_error_results = run_error_analysis(
            model=refusal_model,
            dataset=refusal_test_dataset,
            tokenizer=tokenizer,
            device=DEVICE,
            class_names=CLASS_NAMES,
            task_type='refusal'
        )

        # Run error analysis on jailbreak detector
        print(f"\n{'='*60}")
        print("ERROR ANALYSIS: JAILBREAK DETECTOR")
        print(f"{'='*60}\n")

        jailbreak_error_results = run_error_analysis(
            model=jailbreak_model,
            dataset=jailbreak_test_dataset,
            tokenizer=tokenizer,
            device=DEVICE,
            class_names=JAILBREAK_CLASS_NAMES,
            task_type='jailbreak'
        )

        # Print final summary
        print(f"\n{'#'*60}")
        print("PHASE 2 CROSS-VALIDATION COMPLETE")
        print(f"{'#'*60}\n")

        print("✅ Refusal Classifier:")
        print(f"   CV Accuracy: {refusal_cv_results['cv_results']['overall']['accuracy']['mean']:.4f} ± {refusal_cv_results['cv_results']['overall']['accuracy']['std']:.4f}")
        print(f"   Test Accuracy: {refusal_cv_results['test_results']['accuracy']:.4f}")
        print(f"   Model saved: {refusal_cv_results['final_model_path']}")

        print(f"\n✅ Jailbreak Detector:")
        print(f"   CV Accuracy: {jailbreak_cv_results['cv_results']['overall']['accuracy']['mean']:.4f} ± {jailbreak_cv_results['cv_results']['overall']['accuracy']['std']:.4f}")
        print(f"   Test Accuracy: {jailbreak_cv_results['test_results']['accuracy']:.4f}")
        print(f"   Model saved: {jailbreak_cv_results['final_model_path']}")

        print(f"\n✅ Error Analysis:")
        print(f"   Refusal failure cases: {len(refusal_error_results.get('failure_cases', []))}")
        print(f"   Jailbreak failure cases: {len(jailbreak_error_results.get('failure_cases', []))}")

        print(f"\n✅ All results saved to:")
        print(f"   - {results_path}")
        print(f"   - {visualizations_path}")
        print(f"   - {models_path}")

        print(f"\n{'#'*60}\n")

        return {
            'refusal_cv': refusal_cv_results,
            'jailbreak_cv': jailbreak_cv_results,
            'refusal_error': refusal_error_results,
            'jailbreak_error': jailbreak_error_results,
            'hypothesis_tests': {
                'refusal': refusal_stats,
                'jailbreak': jailbreak_stats
            }
        }


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
