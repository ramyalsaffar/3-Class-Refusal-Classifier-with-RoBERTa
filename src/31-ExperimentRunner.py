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

    def _read_api_keys_from_file(self) -> Dict[str, str]:
        """
        Read API keys from local file.

        File format (each line):
            OpenAI API Key: sk-...
            Anthropic API Key: sk-ant-...
            Google API Key: AI...

        Returns:
            Dictionary with keys: 'openai', 'anthropic', 'google'
            Returns None if file not found or parsing fails
        """
        try:
            if not os.path.exists(api_keys_file_path):
                print(f"⚠️  API keys file not found: {api_keys_file_path}")
                return None

            api_keys = {}
            with open(api_keys_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    # Strip ALL whitespace and remove non-printable characters
                    line = line.strip()
                    # Remove any BOM or hidden characters
                    line = ''.join(char for char in line if char.isprintable())

                    if not line:
                        continue

                    # Parse line: "OpenAI API Key: sk-..."
                    if ':' in line:
                        key_name, key_value = line.split(':', 1)
                        # Aggressively strip all whitespace
                        key_value = key_value.strip()

                        if 'OpenAI' in key_name:
                            api_keys['openai'] = key_value
                        elif 'Anthropic' in key_name:
                            api_keys['anthropic'] = key_value
                        elif 'Google' in key_name:
                            api_keys['google'] = key_value

            # Validate all keys are present
            if all(k in api_keys for k in ['openai', 'anthropic', 'google']):
                return api_keys
            else:
                missing = [k for k in ['openai', 'anthropic', 'google'] if k not in api_keys]
                print(f"⚠️  Missing API keys in file: {', '.join(missing)}")
                return None

        except Exception as e:
            print(f"⚠️  Error reading API keys file: {e}")
            return None

    def _get_api_keys(self) -> Dict[str, str]:
        """
        Get API keys based on environment.

        Priority order:
        1. AWS Secrets Manager (if in AWS)
        2. Local API Keys file (if exists)
        3. Environment variables (.env)
        4. Manual input (last resort)

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
                print("Falling back to local methods...")

        # Local mode: Try API keys file first
        print("🔑 API Key Configuration")
        print("-" * 40)

        # Try reading from local API keys file
        file_keys = self._read_api_keys_from_file()
        if file_keys:
            print("✓ Loaded API keys from local file")
            print(f"  - OpenAI: {'*' * 20}{file_keys['openai'][-4:]} (length: {len(file_keys['openai'])})")
            print(f"  - Anthropic: {'*' * 20}{file_keys['anthropic'][-4:]} (length: {len(file_keys['anthropic'])})")
            print(f"  - Google: {'*' * 20}{file_keys['google'][-4:]} (length: {len(file_keys['google'])})")

            # Debug: Check for whitespace issues
            if file_keys['openai'].strip() != file_keys['openai']:
                print("  ⚠️  WARNING: OpenAI key has leading/trailing whitespace!")
            if file_keys['anthropic'].strip() != file_keys['anthropic']:
                print("  ⚠️  WARNING: Anthropic key has leading/trailing whitespace!")
            if file_keys['google'].strip() != file_keys['google']:
                print("  ⚠️  WARNING: Google key has leading/trailing whitespace!")

            return file_keys

        # Try loading from .env file
        try:
            load_dotenv()
            print("✓ Checking .env file...")
        except (ImportError, FileNotFoundError):
            pass  # dotenv is optional

        # Get or prompt for OpenAI key
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("\nOpenAI API Key not found in file or .env")
            openai_key = getpass.getpass("Enter OpenAI API Key: ")

        # Get or prompt for Anthropic key
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_key:
            print("\nAnthropic API Key not found in file or .env")
            anthropic_key = getpass.getpass("Enter Anthropic API Key: ")

        # Get or prompt for Google key
        google_key = os.getenv('GOOGLE_API_KEY')
        if not google_key:
            print("\nGoogle API Key not found in file or .env")
            google_key = getpass.getpass("Enter Google API Key: ")

        print("\n✓ API keys configured")

        return {
            'openai': openai_key,
            'anthropic': anthropic_key,
            'google': google_key
        }

    def _check_and_prompt_for_resume(self) -> tuple:
        """
        Check for existing checkpoints/data and prompt user for execution strategy.

        Returns:
            Tuple of (resume_from_checkpoint: bool, start_step: int)
            - resume_from_checkpoint: Whether to resume from checkpoints (for Steps 2 & 4)
            - start_step: Which step to start from (1-9, where 1 = start from beginning)
        """
        # Check for checkpoints
        checkpoint_patterns = [
            'checkpoint_response_collection_*.pkl',
            'checkpoint_labeling_*.pkl',
            'checkpoint_wildjailbreak_loading_*.pkl'
        ]

        found_checkpoints = {}
        for pattern in checkpoint_patterns:
            checkpoints = glob.glob(os.path.join(data_checkpoints_path, pattern))
            if checkpoints:
                # Get most recent
                latest = max(checkpoints, key=os.path.getmtime)
                operation = pattern.split('_')[1]  # Extract operation name
                timestamp = os.path.basename(latest).split('_')[-1].replace('.pkl', '')
                age_hours = (time.time() - os.path.getmtime(latest)) / 3600

                found_checkpoints[operation] = {
                    'path': latest,
                    'timestamp': timestamp,
                    'age_hours': age_hours
                }

        # Check for intermediate data files
        temp_pipeline = RefusalPipeline(api_keys={'openai': '', 'anthropic': '', 'google': ''})
        available_data = temp_pipeline.detect_available_data()

        # If no checkpoints or data, start fresh
        if not found_checkpoints and not available_data:
            return (False, 1)

        # Display what's available
        print("\n" + "="*70)
        print("📋 EXISTING PIPELINE DATA DETECTED")
        print("="*70)

        # Show checkpoints
        if found_checkpoints:
            print("\n✓ Available Checkpoints (for in-progress operations):")
            for operation, info in found_checkpoints.items():
                print(f"  • {operation.title().replace('_', ' ')}: {info['age_hours']:.1f} hours old")

        # Show completed step data
        if available_data:
            print("\n✓ Completed Step Data:")
            step_names = {
                'responses': 'Step 2: Response Collection',
                'cleaned': 'Step 3: Data Cleaning',
                'labeled': 'Step 4: Data Labeling',
                'splits': 'Step 5: Dataset Preparation'
            }
            for key, data_info in available_data.items():
                step_name = step_names.get(key, key)
                print(f"  • {step_name}: {data_info['basename']} ({data_info['age_hours']:.1f} hours old)")

        print("\n" + "="*70)
        print("\n🎯 EXECUTION OPTIONS:")
        print("─"*70)
        print("1. Resume from latest checkpoints (continue in-progress operations)")
        print("   → Resumes Step 2 (responses) or Step 4 (labeling) if interrupted")
        print("\n2. Start from specific step (skip completed steps)")
        print("   → Load completed data and start from a later step")
        print("\n3. Start fresh (delete all checkpoints and ignore saved data)")
        print("   → Start pipeline from Step 1, delete all existing checkpoints")
        print("─"*70)

        # Get user choice
        while True:
            choice = input("\nSelect option (1/2/3): ").strip()

            if choice == '1':
                # Resume from checkpoints
                print("\n✓ Will resume from latest checkpoints")
                return (True, 1)

            elif choice == '2':
                # Start from specific step
                print("\n" + "─"*70)
                print("📍 SELECT STARTING STEP:")
                print("─"*70)

                # Build step list with availability indicators
                steps = [
                    ("1", "Generate Prompts", 'prompts' in available_data),
                    ("2", "Collect Responses", 'responses' in available_data),
                    ("3", "Clean Data", 'cleaned' in available_data),
                    ("4", "Label Data", 'labeled' in available_data),
                    ("5", "Prepare Datasets", 'splits' in available_data),
                    ("6", "Train Refusal Classifier", 'refusal_model' in available_data),
                    ("7", "Train Jailbreak Detector", 'jailbreak_model' in available_data),
                    ("8", "Run Analyses", 'analysis_results' in available_data),
                    ("9", "Generate Visualizations", 'visualizations' in available_data),
                    ("10", "Generate Reports", 'reports' in available_data)
                ]

                for num, name, has_data in steps:
                    indicator = "✓" if has_data else " "
                    print(f"  {num}. {name:30} [{indicator}]")

                print("\n  Legend: [✓] = Can skip (data available)")
                print("─"*70)

                while True:
                    step_input = input("\nStart from step (1-10): ").strip()
                    try:
                        start_step = int(step_input)
                        if 1 <= start_step <= 10:
                            # Validate that required data exists
                            try:
                                if start_step >= 3:
                                    temp_pipeline.load_data_for_step(start_step)
                                print(f"\n✓ Will start from Step {start_step}: {steps[start_step-1][1]}")
                                return (False, start_step)
                            except FileNotFoundError as e:
                                print(f"\n❌ Cannot start from Step {start_step}: {e}")
                                print("   Please select an earlier step or choose option 3 to start fresh.")
                                continue
                        else:
                            print("⚠️  Please enter a number between 1-10")
                    except ValueError:
                        print("⚠️  Please enter a valid number")

            elif choice == '3':
                # Start fresh
                print("\n⚠️  This will DELETE all checkpoints. Continue? (y/n): ", end='')
                confirm = input().strip().lower()
                if confirm in ['y', 'yes']:
                    print("\n✓ Starting fresh (deleting checkpoints...)")
                    # Cleanup all checkpoints - INCLUDING prompt_generation!
                    for operation in ['prompt_generation', 'response_collection', 'labeling', 'wildjailbreak_loading']:
                        checkpoint_manager = CheckpointManager(
                            checkpoint_dir=data_checkpoints_path,
                            operation_name=operation,
                            auto_cleanup=False
                        )
                        deleted = checkpoint_manager.delete_all_checkpoints(confirm=True)
                        if deleted > 0:
                            print(f"   ✓ Deleted {deleted} {operation} checkpoint(s)")
                    return (False, 1)
                else:
                    print("   Cancelled. Please select another option.")
                    continue

            else:
                print("⚠️  Please enter 1, 2, or 3")

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
            f"Testing with {EXPERIMENT_CONFIG['test_sample_size']} total prompts (distributed across categories)"
        )

        # Check for existing checkpoints and prompt user
        resume_from_checkpoint, start_step = self._check_and_prompt_for_resume()

        # Get API keys
        api_keys = self._get_api_keys()

        # Initialize pipeline with quick test mode and resume flag
        self.pipeline = RefusalPipeline(api_keys, resume_from_checkpoint=resume_from_checkpoint)

        # Override pipeline behavior for quick test
        print("\n🚀 Quick test mode active - using reduced dataset")
        print(f"   Test sample size: {EXPERIMENT_CONFIG['test_sample_size']}")
        print(f"   Reduced total prompts: {EXPERIMENT_CONFIG['test_sample_size']}")

        # Temporarily override dataset config for quick testing
        # WHY: Reduce total prompts to speed up data collection and testing
        original_total_prompts = DATASET_CONFIG['total_prompts']
        DATASET_CONFIG['total_prompts'] = EXPERIMENT_CONFIG['test_sample_size']

        try:
            # Prevent Mac from sleeping during execution
            with KeepAwake():
                if start_step == 1:
                    self.pipeline.run_full_pipeline()
                else:
                    self.pipeline.run_partial_pipeline(start_step=start_step)
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

        # Check for existing checkpoints and prompt user
        resume_from_checkpoint, start_step = self._check_and_prompt_for_resume()

        # Get API keys
        api_keys = self._get_api_keys()

        # Run pipeline with sleep prevention and resume flag
        self.pipeline = RefusalPipeline(api_keys, resume_from_checkpoint=resume_from_checkpoint)
        with KeepAwake():
            if start_step == 1:
                self.pipeline.run_full_pipeline()
            else:
                self.pipeline.run_partial_pipeline(start_step=start_step)

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
        required_columns = ['response', 'refusal_label', 'is_jailbreak_attempt', 'jailbreak_label']
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
        refusal_model = RefusalClassifier(num_classes=MODEL_CONFIG['num_classes'])
        refusal_checkpoint = torch.load(refusal_model_path, map_location=DEVICE)
        refusal_model.load_state_dict(refusal_checkpoint['model_state_dict'])
        refusal_model = refusal_model.to(DEVICE)
        refusal_model.eval()  # WHY: Set to evaluation mode (disables dropout, sets BatchNorm to eval)
        print(f"✓ Refusal classifier loaded (Best Val F1: {refusal_checkpoint['best_val_f1']:.4f})")

        # Load jailbreak detector
        print(f"\nLoading jailbreak detector from {jailbreak_model_path}...")
        jailbreak_model = JailbreakDetector(num_classes=JAILBREAK_CONFIG['num_classes'])
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
            os.path.join(analysis_results_path, "per_model_analysis.json")
        )
        analysis_results['per_model'] = per_model_results

        # Confidence analysis
        print("\n--- Confidence Analysis ---")
        confidence_analyzer = ConfidenceAnalyzer(refusal_model, tokenizer, DEVICE)
        conf_results, preds, labels, confidences = confidence_analyzer.analyze(test_df)
        confidence_analyzer.save_results(
            conf_results,
            os.path.join(analysis_results_path, "confidence_analysis.json")
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
            os.path.join(analysis_results_path, "adversarial_testing.json")
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
            os.path.join(analysis_results_path, "jailbreak_analysis.json")
        )
        analysis_results['jailbreak'] = jailbreak_results

        # Power law analysis for jailbreak detector
        print("\n--- Power Law Analysis (Jailbreak) ---")
        jailbreak_power_law_analyzer = PowerLawAnalyzer(
            jailbreak_model, tokenizer, DEVICE, class_names=JAILBREAK_CLASS_NAMES
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
            jailbreak_class_names=JAILBREAK_CLASS_NAMES
        )
        correlation_results = correlation_analyzer.run_full_analysis()
        correlation_analyzer.save_results(
            correlation_results,
            os.path.join(analysis_results_path, "correlation_analysis.pkl")
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
                'jailbreak_class_names': JAILBREAK_CLASS_NAMES
            }
        }

        # Save as JSON
        structured_results_path = os.path.join(analysis_results_path, "analysis_results_structured.json")
        with open(structured_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✓ Saved structured results to: {structured_results_path}")

        # Also save confusion matrix figure data for reports
        cm = confusion_matrix(labels, preds)
        cm_fig = plt.figure(figsize=(10, 8))
        visualizer.plot_confusion_matrix(cm, None)  # Plot but don't save yet
        plt.savefig(os.path.join(visualizations_path, "confusion_matrix_for_report.png"),
                   dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        print("\n✅ Analysis complete (BOTH classifiers analyzed)")

        # Return results for potential use by ReportGenerator
        return serializable_results


    def train_with_cross_validation(self, k_folds: int = None):
        """
        Train with cross-validation (Phase 2 feature).

        Runs:
        1. Hypothesis testing on dataset
        2. K-fold cross-validation
        3. Final model training on full train+val
        4. Error analysis on test set

        Args:
            k_folds: Number of cross-validation folds (default: from CROSS_VALIDATION_CONFIG)
        """
        # Use config value if not provided - NO HARDCODING!
        if k_folds is None:
            k_folds = CROSS_VALIDATION_CONFIG['default_folds']
        
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

        print()
        print_banner("REFUSAL CLASSIFIER: CROSS-VALIDATION", width=60, char='#')
        print()

        # Prepare refusal dataset
        refusal_texts = labeled_df['response'].tolist()
        refusal_labels = labeled_df['refusal_label'].tolist()  # Fixed: was 'refusal_class'
        refusal_dataset = ClassificationDataset(refusal_texts, refusal_labels, tokenizer)

        # Step 1: Hypothesis testing on refusal dataset
        print()
        print_banner("STEP 1: STATISTICAL HYPOTHESIS TESTING (REFUSAL)", width=60)
        print()

        hypothesis_tester = DatasetValidator(class_names=CLASS_NAMES)
        refusal_stats = hypothesis_tester.analyze_dataset_statistics(
            refusal_dataset,
            task_type='refusal'
        )
        hypothesis_tester.save_results()
        hypothesis_tester.generate_statistical_report()

        # Step 2: Cross-validation on refusal classifier
        print()
        print_banner(f"STEP 2: {k_folds}-FOLD CROSS-VALIDATION (REFUSAL)", width=60)
        print()

        refusal_cv_results = train_with_cross_validation(
            full_dataset=refusal_dataset,
            model_class=RefusalClassifier,
            k_folds=k_folds,
            test_split=DATASET_CONFIG['test_split'],
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

        print()
        print_banner("JAILBREAK DETECTOR: CROSS-VALIDATION", width=60, char='#')
        print()

        # Prepare jailbreak dataset
        jailbreak_texts = labeled_df['response'].tolist()
        jailbreak_labels = labeled_df['jailbreak_label'].tolist()  # NEW: Uses jailbreak_label (0=Failed, 1=Succeeded)
        jailbreak_dataset = ClassificationDataset(jailbreak_texts, jailbreak_labels, tokenizer)

        # Step 1: Hypothesis testing on jailbreak dataset
        print()
        print_banner("STEP 1: STATISTICAL HYPOTHESIS TESTING (JAILBREAK)", width=60)
        print()

        jailbreak_hypothesis_tester = DatasetValidator(class_names=JAILBREAK_CLASS_NAMES)
        jailbreak_stats = jailbreak_hypothesis_tester.analyze_dataset_statistics(
            jailbreak_dataset,
            task_type='jailbreak'
        )
        jailbreak_hypothesis_tester.save_results()
        jailbreak_hypothesis_tester.generate_statistical_report()

        # Step 2: Cross-validation on jailbreak detector
        print()
        print_banner(f"STEP 2: {k_folds}-FOLD CROSS-VALIDATION (JAILBREAK)", width=60)
        print()

        jailbreak_cv_results = train_with_cross_validation(
            full_dataset=jailbreak_dataset,
            model_class=JailbreakDetector,
            k_folds=k_folds,
            test_split=DATASET_CONFIG['test_split'],
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

        print()
        print_banner("COMPREHENSIVE ERROR ANALYSIS", width=60, char='#')
        print()

        # Load trained models
        refusal_model = RefusalClassifier(num_classes=MODEL_CONFIG['num_classes'])
        refusal_checkpoint = torch.load(
            os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt"),
            map_location=DEVICE
        )
        refusal_model.load_state_dict(refusal_checkpoint['model_state_dict'])
        refusal_model = refusal_model.to(DEVICE)
        refusal_model.eval()  # WHY: Set to evaluation mode (disables dropout, sets BatchNorm to eval)

        jailbreak_model = JailbreakDetector(num_classes=JAILBREAK_CONFIG['num_classes'])
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
        print()
        print_banner("ERROR ANALYSIS: REFUSAL CLASSIFIER", width=60)
        print()

        refusal_error_results = run_error_analysis(
            model=refusal_model,
            dataset=refusal_test_dataset,
            tokenizer=tokenizer,
            device=DEVICE,
            class_names=CLASS_NAMES,
            task_type='refusal'
        )

        # Run error analysis on jailbreak detector
        print()
        print_banner("ERROR ANALYSIS: JAILBREAK DETECTOR", width=60)
        print()

        jailbreak_error_results = run_error_analysis(
            model=jailbreak_model,
            dataset=jailbreak_test_dataset,
            tokenizer=tokenizer,
            device=DEVICE,
            class_names=JAILBREAK_CLASS_NAMES,
            task_type='jailbreak'
        )

        # Print final summary
        print()
        print_banner("PHASE 2 CROSS-VALIDATION COMPLETE", width=60, char='#')
        print()

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
        print(f"   - {analysis_results_path}")
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
