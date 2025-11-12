# RefusalPipeline Module
#-----------------------
# Main pipeline orchestrator for the complete refusal classification pipeline.
# Trains TWO independent classifiers: Refusal Classifier + Jailbreak Detector.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Uses safe_divide() from Utils for robust division
# - Uses get_timestamp() from Utils for consistent timestamps (already integrated)
# - Uses print_banner() from Utils for section headers
# - Improved comments and documentation
# All imports are in 01-Imports.py
###############################################################################


class RefusalPipeline:
    """
    Orchestrate the complete refusal classification pipeline with dual classifiers.
    
    This is the main pipeline that coordinates all steps from prompt generation
    through model training and evaluation. Trains two independent classifiers:
    
    1. **Refusal Classifier** (3-class): No Refusal, Hard Refusal, Soft Refusal
    2. **Jailbreak Detector** (2-class): Jailbreak Failed, Jailbreak Succeeded
    
    Pipeline Steps:
    1. Generate prompts (PromptGenerator)
    2. Collect LLM responses (ResponseCollector)  
    3. Clean data (DataCleaner)
    4. Label data with LLM judge (DataLabeler)
    5. Supplement jailbreak data if needed (WildJailbreak)
    6. Prepare datasets for both classifiers
    7. Train refusal classifier
    8. Train jailbreak detector
    9. Run analyses (cross-validation, per-model, confidence)
    10. Generate visualizations and reports
    
    All configuration comes from Config file (DATASET_CONFIG, TRAINING_CONFIG, etc.)
    """

    def __init__(self, api_keys: Dict, resume_from_checkpoint: bool = False):
        """
        Initialize pipeline.

        Args:
            api_keys: Dictionary with keys: 'openai', 'anthropic', 'google'
            resume_from_checkpoint: If True, resume from existing checkpoints
        """
        self.api_keys = api_keys
        self.resume_from_checkpoint = resume_from_checkpoint
        self.results = {}
        self.refusal_model = None
        self.jailbreak_model = None
        self.tokenizer = None

    def detect_available_data(self) -> Dict[str, Dict]:
        """
        Detect available intermediate data files and checkpoints.

        Returns:
            Dictionary mapping step names to available data/checkpoint info
        """
        available = {}

        # Step 1: Generated prompts
        prompt_files = glob.glob(os.path.join(data_raw_path, "prompts_*.json"))
        if prompt_files:
            latest = max(prompt_files, key=os.path.getmtime)
            available['prompts'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 1
            }

        # Step 2: Response collection
        response_files = glob.glob(os.path.join(data_responses_path, "responses_*.pkl"))
        if response_files:
            latest = max(response_files, key=os.path.getmtime)
            available['responses'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 2
            }

        # Step 3: Cleaned data
        cleaned_files = glob.glob(os.path.join(data_processed_path, "cleaned_responses_*.pkl"))
        if cleaned_files:
            latest = max(cleaned_files, key=os.path.getmtime)
            available['cleaned'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 3
            }

        # Step 4: Labeled data
        labeled_files = glob.glob(os.path.join(data_processed_path, "labeled_responses_*.pkl"))
        if labeled_files:
            latest = max(labeled_files, key=os.path.getmtime)
            available['labeled'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 4
            }

        # Step 5: Train/val/test splits
        train_files = glob.glob(os.path.join(data_splits_path, "train_*.pkl"))
        val_files = glob.glob(os.path.join(data_splits_path, "val_*.pkl"))
        test_files = glob.glob(os.path.join(data_splits_path, "test_*.pkl"))
        if train_files and val_files and test_files:
            latest_train = max(train_files, key=os.path.getmtime)
            available['splits'] = {
                'path': latest_train,  # Use train file as reference
                'basename': f"{len(train_files)} split files",
                'age_hours': (time.time() - os.path.getmtime(latest_train)) / 3600,
                'step': 5
            }

        return available

    def load_data_for_step(self, step: int) -> pd.DataFrame:
        """
        Load appropriate data for starting from a specific step.

        Args:
            step: Step number to start from (1-9)

        Returns:
            DataFrame with data needed for that step
        """
        if step <= 1:
            return None  # Start fresh

        available = self.detect_available_data()

        # For step 2 (collect responses), we need prompts - start fresh
        if step == 2:
            return None

        # For step 3 (clean data), load responses
        if step == 3:
            if 'responses' in available:
                print(f"📂 Loading responses from: {available['responses']['basename']}")
                return pd.read_pickle(available['responses']['path'])
            else:
                raise FileNotFoundError("No response data found. Please start from Step 2 (Collect Responses)")

        # For step 4 (label data), load cleaned responses
        if step == 4:
            if 'cleaned' in available:
                print(f"📂 Loading cleaned data from: {available['cleaned']['basename']}")
                return pd.read_pickle(available['cleaned']['path'])
            elif 'responses' in available:
                print(f"📂 No cleaned data found, loading responses from: {available['responses']['basename']}")
                print("   Will clean data before labeling...")
                return pd.read_pickle(available['responses']['path'])
            else:
                raise FileNotFoundError("No response or cleaned data found. Please start from Step 2")

        # For step 5+ (prepare datasets, training), load labeled data
        if step >= 5:
            if 'labeled' in available:
                print(f"📂 Loading labeled data from: {available['labeled']['basename']}")
                return pd.read_pickle(available['labeled']['path'])
            else:
                raise FileNotFoundError("No labeled data found. Please start from Step 4 (Label Data)")

        return None

    def run_partial_pipeline(self, start_step: int = 1):
        """
        Execute pipeline starting from a specific step.

        Args:
            start_step: Step number to start from (1-9)
                1: Generate Prompts
                2: Collect Responses
                3: Clean Data
                4: Label Data
                5: Prepare Datasets
                6: Train Refusal Classifier
                7: Train Jailbreak Detector
                8: Run Analyses
                9: Generate Visualizations
        """
        print(f"\n🔍 DEBUG [RefusalPipeline.run_partial_pipeline]: METHOD CALLED with start_step={start_step}")

        # Generate single timestamp for this run
        self.run_timestamp = get_timestamp('file')

        print_banner(f"REFUSAL CLASSIFIER - PARTIAL PIPELINE (START: STEP {start_step})", width=60)
        print(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
        print(f"Run Timestamp: {self.run_timestamp}")
        print(f"Classifier 1: Refusal Classification (3 classes)")
        print(f"Classifier 2: Jailbreak Detection (2 classes)")
        print("="*60 + "\n")

        # Load data if starting from later step
        prompts = None
        responses_df = None
        cleaned_df = None
        labeled_df = None
        datasets = None

        # Execute pipeline from specified step
        if start_step <= 1:
            prompts = self.generate_prompts()

        if start_step <= 2:
            if start_step == 2:
                prompts = self.generate_prompts()  # Need prompts for collection
            responses_df = self.collect_responses(prompts)
        elif start_step >= 3:
            responses_df = self.load_data_for_step(3)

        if start_step <= 3:
            cleaned_df = self.clean_data(responses_df)
        elif start_step >= 4:
            cleaned_df = self.load_data_for_step(4)

        if start_step <= 4:
            labeled_df = self.label_data(cleaned_df)
            labeled_df_augmented = self.prepare_jailbreak_training_data(labeled_df)
        elif start_step >= 5:
            labeled_df = self.load_data_for_step(5)
            labeled_df_augmented = self.prepare_jailbreak_training_data(labeled_df)

        if start_step <= 5:
            datasets = self.prepare_datasets(labeled_df_augmented)

        # Training and analysis always run if we reach this point
        if start_step <= 6:
            refusal_history = self.train_refusal_classifier(
                datasets['refusal']['train_loader'],
                datasets['refusal']['val_loader']
            )

        if start_step <= 7:
            jailbreak_history = self.train_jailbreak_detector(
                datasets['jailbreak']['train_loader'],
                datasets['jailbreak']['val_loader']
            )

        if start_step <= 8:
            analysis_results = self.run_analyses(datasets['refusal']['test_df'])

        if start_step <= 9:
            self.generate_visualizations(refusal_history, jailbreak_history, analysis_results)

        print("\n" + "="*60)
        print(f"✅ PARTIAL PIPELINE COMPLETE (Started from Step {start_step})")
        print("="*60)

    def run_full_pipeline(self):
        """Execute complete pipeline from start to finish."""
        print("\n🔍 DEBUG [RefusalPipeline.run_full_pipeline]: METHOD CALLED")

        # Generate single timestamp for this entire run
        self.run_timestamp = get_timestamp('file')

        print_banner("REFUSAL CLASSIFIER - FULL PIPELINE (DUAL CLASSIFIERS)", width=60)
        print(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
        print(f"Run Timestamp: {self.run_timestamp}")
        print(f"Classifier 1: Refusal Classification (3 classes)")
        print(f"Classifier 2: Jailbreak Detection (2 classes)")
        print("="*60 + "\n")

        # Step 1: Generate prompts
        prompts = self.generate_prompts()

        # Step 2: Collect responses
        responses_df = self.collect_responses(prompts)

        # Step 3: Clean data (remove invalid responses before labeling)
        cleaned_df = self.clean_data(responses_df)

        # Step 4: Label data (dual-task labeling - only clean data)
        labeled_df = self.label_data(cleaned_df)

        # Step 4.5: Prepare jailbreak training data (NEW - V09)
        # Supplements with WildJailbreak if insufficient real jailbreak succeeded samples
        labeled_df_augmented = self.prepare_jailbreak_training_data(labeled_df)

        # Step 5: Prepare datasets for BOTH classifiers
        datasets = self.prepare_datasets(labeled_df_augmented)

        # Step 6: Train refusal classifier
        refusal_history = self.train_refusal_classifier(
            datasets['refusal']['train_loader'],
            datasets['refusal']['val_loader']
        )

        # Step 7: Train jailbreak detector
        jailbreak_history = self.train_jailbreak_detector(
            datasets['jailbreak']['train_loader'],
            datasets['jailbreak']['val_loader']
        )

        # Step 8: Run analyses
        analysis_results = self.run_analyses(datasets['refusal']['test_df'])

        # Step 9: Generate visualizations
        self.generate_visualizations(refusal_history, jailbreak_history, analysis_results)

        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE (DUAL CLASSIFIERS TRAINED)")
        print("="*60)

    def generate_prompts(self) -> Dict[str, List[str]]:
        """Step 1: Generate prompts."""
        print_banner("STEP 1: GENERATING PROMPTS", width=60)

        generator = PromptGenerator(self.api_keys['openai'])
        prompts = generator.generate_all_prompts()
        generator.save_prompts(prompts, data_raw_path, timestamp=self.run_timestamp)

        return prompts

    def collect_responses(self, prompts: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Step 2: Collect LLM responses.

        UPDATED (Phase 2/3): Uses parallel processing if enabled in config.
        """
        print_banner("STEP 2: COLLECTING RESPONSES", width=60)

        collector = ResponseCollector(
            self.api_keys['anthropic'],
            self.api_keys['openai'],
            self.api_keys['google']
        )

        # Use parallel processing if async is enabled
        responses_df = collector.collect_all_responses(
            prompts,
            parallel=API_CONFIG.get('use_async', True),
            resume_from_checkpoint=self.resume_from_checkpoint
        )

        # Save responses to disk
        timestamp = self.run_timestamp
        responses_path = os.path.join(data_responses_path, f"responses_{timestamp}.pkl")
        responses_df.to_pickle(responses_path)
        print(f"💾 Responses saved: {responses_path}")

        return responses_df

    def label_data(self, responses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Label responses using LLM Judge (dual-task).

        Labels ONLY clean data (after cleaning), saving API costs.

        UPDATED (Phase 2/3): Uses parallel processing if enabled in config.
        """
        print_banner("STEP 4: LABELING DATA WITH LLM JUDGE (DUAL-TASK)", width=60)

        # Initialize labeler with OpenAI API key (for GPT-4o judge)
        labeler = DataLabeler(api_key=self.api_keys['openai'])

        # Use checkpointed version if async is enabled
        if API_CONFIG.get('use_async', True):
            labeled_df = labeler.label_dataset_with_checkpoints(
                responses_df,
                resume_from_checkpoint=self.resume_from_checkpoint
            )
            # Rename jailbreak_success to jailbreak_label for consistency with rest of pipeline
            if 'jailbreak_success' in labeled_df.columns:
                labeled_df = labeled_df.rename(columns={'jailbreak_success': 'jailbreak_label'})
            return labeled_df
        else:
            # Sequential labeling (original implementation)
            refusal_labels = []
            jailbreak_labels = []
            refusal_confidences = []
            jailbreak_confidences = []

            for idx, row in tqdm(responses_df.iterrows(), total=len(responses_df), desc="Dual-Task LLM Judge Labeling"):
                refusal_label, jailbreak_label, refusal_conf, jailbreak_conf = labeler.label_response(
                    response=row['response'],
                    prompt=row['prompt']
                )
                refusal_labels.append(refusal_label)
                jailbreak_labels.append(jailbreak_label)
                refusal_confidences.append(refusal_conf)
                jailbreak_confidences.append(jailbreak_conf)

            responses_df['refusal_label'] = refusal_labels
            responses_df['jailbreak_label'] = jailbreak_labels
            responses_df['refusal_confidence'] = refusal_confidences
            responses_df['jailbreak_confidence'] = jailbreak_confidences

        # Print refusal label distribution
        print(f"\n{'='*60}")
        print(f"REFUSAL LABELING SUMMARY")
        print(f"{'='*60}")
        for i in range(-1, 3):
            count = (responses_df['refusal_label'] == i).sum()
            pct = safe_divide(count, len(responses_df), default=0.0) * 100
            label_name = labeler.get_label_name(i)
            print(f"  {label_name}: {count} ({pct:.1f}%)")

        # Print jailbreak label distribution
        print(f"\n{'='*60}")
        print(f"JAILBREAK DETECTION SUMMARY")
        print(f"{'='*60}")
        for i in [0, 1]:
            count = (responses_df['jailbreak_label'] == i).sum()
            pct = safe_divide(count, len(responses_df), default=0.0) * 100
            label_name = labeler.get_jailbreak_label_name(i)
            print(f"  {label_name}: {count} ({pct:.1f}%)")

        # Print confidence statistics
        print(f"\n{'='*60}")
        print(f"CONFIDENCE STATISTICS")
        print(f"{'='*60}")
        valid_refusal = responses_df[responses_df['refusal_label'] != -1]
        valid_jailbreak = responses_df[responses_df['jailbreak_label'] != -1]

        # Use config threshold for low confidence
        low_conf_threshold = LABELING_CONFIG['low_confidence_threshold']

        if len(valid_refusal) > 0:
            avg_ref_conf = valid_refusal['refusal_confidence'].mean()
            low_conf_ref = (valid_refusal['refusal_confidence'] < low_conf_threshold).sum()
            print(f"  Refusal - Avg Confidence: {avg_ref_conf:.1f}%")
            print(f"  Refusal - Low confidence (<{low_conf_threshold}%): {low_conf_ref} ({low_conf_ref/len(valid_refusal)*100:.1f}%)")

        if len(valid_jailbreak) > 0:
            avg_jb_conf = valid_jailbreak['jailbreak_confidence'].mean()
            low_conf_jb = (valid_jailbreak['jailbreak_confidence'] < low_conf_threshold).sum()
            print(f"  Jailbreak - Avg Confidence: {avg_jb_conf:.1f}%")
            print(f"  Jailbreak - Low confidence (<{low_conf_threshold}%): {low_conf_jb} ({low_conf_jb/len(valid_jailbreak)*100:.1f}%)")

        # Analyze labeling quality
        print(f"\n{'='*60}")
        print("LABELING QUALITY ANALYSIS")
        print(f"{'='*60}")
        quality_analyzer = LabelingQualityAnalyzer(verbose=True)
        quality_results = quality_analyzer.analyze_full(responses_df)

        # Use run timestamp for consistency
        timestamp = self.run_timestamp

        # Save quality analysis results
        quality_analysis_path = os.path.join(analysis_results_path, f"labeling_quality_analysis_{timestamp}.json")
        quality_analyzer.save_results(quality_results, quality_analysis_path)

        # Export low-confidence samples for review
        if quality_results['low_confidence']['low_both_count'] > 0:
            flagged_samples_path = os.path.join(results_path, f"flagged_samples_for_review_{timestamp}.csv")
            quality_analyzer.export_flagged_samples(responses_df, flagged_samples_path, threshold=LABELING_CONFIG['low_confidence_threshold'])

        # Save labeled data
        labeled_path = os.path.join(data_processed_path, f"labeled_responses_{timestamp}.pkl")
        responses_df.to_pickle(labeled_path)
        print(f"\n✓ Saved labeled data to {labeled_path}")

        return responses_df

    def clean_data(self, responses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Clean and validate data quality BEFORE labeling.

        Removes duplicates, outliers, and invalid data before expensive labeling.
        This saves API costs by not labeling garbage data.

        Args:
            responses_df: Raw responses DataFrame (unlabeled)

        Returns:
            Cleaned DataFrame ready for labeling
        """
        print("\n" + "="*60)
        print("STEP 3: CLEANING DATA (BEFORE LABELING)")
        print("="*60)

        # Initialize cleaner
        cleaner = DataCleaner()

        # Get outlier report first
        report = cleaner.get_outlier_report(responses_df)

        if report['issues_found']:
            print(f"\n📋 Outlier Report:")
            print(f"   Total samples: {report['total_samples']}")
            for issue in report['issues_found']:
                print(f"   • {issue['type']}: {issue['count']} ({issue['percentage']:.2f}%)")
            print(f"   Recommendation: {report['recommendation']}")
        else:
            print(f"\n✅ No data quality issues detected!")
            print(f"   Total samples: {report['total_samples']}")

        # Clean the data
        strategy = DATA_CLEANING_CONFIG['default_strategy']
        cleaned_df = cleaner.clean_dataset(responses_df, strategy=strategy)

        # Save cleaned data
        timestamp = self.run_timestamp
        cleaned_path = os.path.join(data_processed_path, f"cleaned_responses_{timestamp}.pkl")
        cleaned_df.to_pickle(cleaned_path)
        print(f"\n✓ Saved cleaned data to {cleaned_path}")

        return cleaned_df

    def prepare_datasets(self, labeled_df: pd.DataFrame) -> Dict:
        """
        Step 5: Prepare train/val/test datasets for BOTH classifiers.

        Returns:
            Dictionary with structure:
            {
                'refusal': {train_loader, val_loader, test_loader, test_df},
                'jailbreak': {train_loader, val_loader, test_loader, test_df}
            }
        """
        print("\n" + "="*60)
        print("STEP 5: PREPARING DATASETS (DUAL CLASSIFIERS)")
        print("="*60)

        # Validate required columns exist before proceeding
        required_cols = ['prompt', 'response', 'refusal_label', 'jailbreak_label']
        missing_cols = [col for col in required_cols if col not in labeled_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for dataset preparation: {missing_cols}\n"
                f"This usually means labeling failed or was skipped. "
                f"Expected columns: {required_cols}"
            )

        # Filter out error labels (-1) from refusal labels
        error_count = (labeled_df['refusal_label'] == -1).sum()
        if error_count > 0:
            print(f"⚠️  Filtering out {error_count} error-labeled samples")
            labeled_df = labeled_df[labeled_df['refusal_label'] != -1].copy()
            print(f"✓ Remaining samples: {len(labeled_df)}")

        # Filter out NaN values in critical columns (COMPREHENSIVE FILTERING)
        # This prevents downstream errors in analyzers (PerModelAnalyzer, AttentionVisualizer, AdversarialTester)
        initial_count = len(labeled_df)
        labeled_df = labeled_df.dropna(subset=['response', 'prompt', 'model', 'refusal_label', 'jailbreak_label']).copy()
        nan_filtered = initial_count - len(labeled_df)

        if nan_filtered > 0:
            print(f"⚠️  Filtered out {nan_filtered} samples with NaN values in critical columns")
            print(f"✓ Remaining samples: {len(labeled_df)}")

        if len(labeled_df) == 0:
            raise ValueError("No valid samples remaining after filtering! Check data quality.")

        # Split data (same splits for both classifiers to maintain consistency)
        train_df, temp_df = train_test_split(
            labeled_df,
            test_size=(1 - DATASET_CONFIG['train_split']),
            random_state=DATASET_CONFIG['random_seed'],
            stratify=labeled_df['refusal_label']  # Stratify by refusal label
        )

        # Use safe_divide from Utils for robust calculation
        val_size = safe_divide(
            DATASET_CONFIG['val_split'], 
            DATASET_CONFIG['val_split'] + DATASET_CONFIG['test_split'],
            default=0.5
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=DATASET_CONFIG['random_seed'],
            stratify=temp_df['refusal_label']
        )

        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_df)} ({len(train_df)/len(labeled_df)*100:.1f}%)")
        print(f"  Val: {len(val_df)} ({len(val_df)/len(labeled_df)*100:.1f}%)")
        print(f"  Test: {len(test_df)} ({len(test_df)/len(labeled_df)*100:.1f}%)")

        # Save splits
        timestamp = self.run_timestamp
        train_df.to_pickle(os.path.join(data_splits_path, f"train_{timestamp}.pkl"))
        val_df.to_pickle(os.path.join(data_splits_path, f"val_{timestamp}.pkl"))
        test_df.to_pickle(os.path.join(data_splits_path, f"test_{timestamp}.pkl"))

        # Add 'label' column for backward compatibility with analysis modules
        # Analysis modules expect 'label' column, so copy 'refusal_label' to 'label'
        train_df['label'] = train_df['refusal_label']
        val_df['label'] = val_df['refusal_label']
        test_df['label'] = test_df['refusal_label']

        # Initialize tokenizer (shared by both classifiers)
        print("\nInitializing tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

        # Prepare datasets for REFUSAL CLASSIFIER
        print("\n--- Preparing Refusal Classifier Datasets ---")
        refusal_train_dataset = ClassificationDataset(
            train_df['response'].tolist(),
            train_df['refusal_label'].tolist(),
            self.tokenizer
        )

        refusal_val_dataset = ClassificationDataset(
            val_df['response'].tolist(),
            val_df['refusal_label'].tolist(),
            self.tokenizer
        )

        refusal_test_dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['refusal_label'].tolist(),
            self.tokenizer
        )

        refusal_train_loader, refusal_val_loader, refusal_test_loader = create_dataloaders(
            refusal_train_dataset, refusal_val_dataset, refusal_test_dataset
        )

        print(f"✓ Refusal classifier dataloaders created")
        print(f"  Train batches: {len(refusal_train_loader)}")
        print(f"  Val batches: {len(refusal_val_loader)}")
        print(f"  Test batches: {len(refusal_test_loader)}")

        # Prepare datasets for JAILBREAK DETECTOR
        print("\n--- Preparing Jailbreak Detector Datasets ---")
        jailbreak_train_dataset = ClassificationDataset(
            train_df['response'].tolist(),
            train_df['jailbreak_label'].tolist(),
            self.tokenizer
        )

        jailbreak_val_dataset = ClassificationDataset(
            val_df['response'].tolist(),
            val_df['jailbreak_label'].tolist(),
            self.tokenizer
        )

        jailbreak_test_dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['jailbreak_label'].tolist(),
            self.tokenizer
        )

        jailbreak_train_loader, jailbreak_val_loader, jailbreak_test_loader = create_dataloaders(
            jailbreak_train_dataset, jailbreak_val_dataset, jailbreak_test_dataset
        )

        print(f"✓ Jailbreak detector dataloaders created")
        print(f"  Train batches: {len(jailbreak_train_loader)}")
        print(f"  Val batches: {len(jailbreak_val_loader)}")
        print(f"  Test batches: {len(jailbreak_test_loader)}")

        return {
            'refusal': {
                'train_loader': refusal_train_loader,
                'val_loader': refusal_val_loader,
                'test_loader': refusal_test_loader,
                'test_df': test_df
            },
            'jailbreak': {
                'train_loader': jailbreak_train_loader,
                'val_loader': jailbreak_val_loader,
                'test_loader': jailbreak_test_loader,
                'test_df': test_df
            }
        }

    def train_refusal_classifier(self, train_loader, val_loader) -> Dict:
        """Step 6: Train RoBERTa refusal classifier (3 classes)."""
        print("\n" + "="*60)
        print("STEP 6: TRAINING REFUSAL CLASSIFIER (3 CLASSES)")
        print("="*60)

        # Initialize model
        # WHY: Use len(CLASS_NAMES) for generic design (works with any N-class classifier)
        self.refusal_model = RefusalClassifier(num_classes=len(CLASS_NAMES))
        self.refusal_model.freeze_roberta_layers()
        self.refusal_model = self.refusal_model.to(DEVICE)

        print(f"\nModel: {MODEL_CONFIG['model_name']}")
        print(f"Trainable parameters: {count_parameters(self.refusal_model):,}")

        # Calculate class weights
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['label'].tolist())

        class_counts = [train_labels.count(i) for i in range(self.refusal_model.num_classes)]
        print(f"\nClass distribution in training set:")
        for i, count in enumerate(class_counts):
            print(f"  Class {i} ({CLASS_NAMES[i]}): {count}")

        criterion = get_weighted_criterion(class_counts, DEVICE)

        # Optimizer and scheduler
        optimizer = AdamW(
            self.refusal_model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )

        num_training_steps = len(train_loader) * TRAINING_CONFIG['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=TRAINING_CONFIG['warmup_steps'],
            num_training_steps=num_training_steps
        )

        # Train
        trainer = Trainer(
            self.refusal_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            DEVICE
        )

        try:
            history = trainer.train(
                model_save_path=os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt")
            )
            print(f"\n✓ Refusal classifier training complete")
            return history
        except Exception as e:
            print(f"\n❌ Refusal classifier training failed: {e}")
            raise

    def prepare_jailbreak_training_data(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare jailbreak training data with WildJailbreak supplementation if needed.

        NEW METHOD (V09): Supplements training data when modern LLMs successfully
        defend against all jailbreak attempts, resulting in insufficient positive samples.

        Args:
            labeled_df: DataFrame with labeled data (includes jailbreak_label column)

        Returns:
            DataFrame with jailbreak training data (real + WildJailbreak if needed)
        """
        print(f"\n\n")
        print(f"{'#'*60}")
        print(f"{'#'*60}")
        print(f"##  STEP 4.5: WILDJAILBREAK SUPPLEMENTATION")
        print(f"{'#'*60}")
        print(f"{'#'*60}")
        print(f"\n🔧 DEBUG INFO:")
        print(f"   Input dataframe shape: {labeled_df.shape}")
        print(f"   WILDJAILBREAK_CONFIG['enabled'] = {WILDJAILBREAK_CONFIG['enabled']}")
        print(f"   Jailbreak succeeded count: {(labeled_df['jailbreak_label'] == 1).sum()}")

        if not WILDJAILBREAK_CONFIG['enabled']:
            print(f"\n📊 WildJailbreak supplementation disabled")
            # Add data_source column for consistency
            labeled_df['data_source'] = 'real'
            return labeled_df

        print(f"\n{'='*60}")
        print(f"📊 PREPARING JAILBREAK TRAINING DATA")
        print(f"{'='*60}")

        # Calculate threshold from config (no hardcoded values)
        threshold = int(DATASET_CONFIG['total_prompts'] * (WILDJAILBREAK_CONFIG['min_threshold_percentage'] / 100))

        # Count real jailbreak succeeded samples (jailbreak_label == 1)
        real_jailbreak_succeeded = labeled_df[labeled_df['jailbreak_label'] == 1]
        real_count = len(real_jailbreak_succeeded)

        print(f"\n  Threshold: {threshold} samples ({WILDJAILBREAK_CONFIG['min_threshold_percentage']}% of {DATASET_CONFIG['total_prompts']} prompts)")
        print(f"  Real jailbreak succeeded samples: {real_count}")

        # Add data_source column to real data
        labeled_df_copy = labeled_df.copy()
        labeled_df_copy['data_source'] = 'real'

        # Check if supplementation needed
        if real_count >= threshold:
            print(f"  ✓ Sufficient real data - no supplementation needed")
            print(f"  Real data: 100%")
            print(f"{'='*60}\n")

            # Save labeled data with data_source column (even without WildJailbreak)
            # This ensures labeled_responses file always reflects the data used in Step 5
            timestamp = self.run_timestamp
            labeled_path = os.path.join(data_processed_path, f"labeled_responses_{timestamp}.pkl")
            labeled_df_copy.to_pickle(labeled_path)
            print(f"✓ Saved labeled data to {labeled_path}\n")

            return labeled_df_copy

        # Need supplementation
        samples_needed = threshold - real_count
        print(f"  ⚠️  Insufficient data - need {samples_needed} more samples")
        print(f"\n  Loading WildJailbreak dataset...")

        try:
            # Check if datasets library is installed
            try:
                print(f"  ✓ 'datasets' library found (version {datasets.__version__})")
            except ImportError:
                raise ImportError(
                    "\n\n"
                    "="*60 + "\n"
                    "❌ CRITICAL ERROR: 'datasets' library not installed!\n"
                    "="*60 + "\n"
                    "WildJailbreak supplementation requires the HuggingFace datasets library.\n"
                    "\n"
                    "Install with:\n"
                    "  pip install datasets\n"
                    "\n"
                    "Or install all requirements:\n"
                    "  pip install -r requirements.txt\n"
                    "="*60
                )

            # Initialize WildJailbreak loader
            loader = WildJailbreakLoader(random_seed=WILDJAILBREAK_CONFIG['random_seed'])

            # Load and sample
            wildjailbreak_samples = loader.load_and_sample(n_samples=samples_needed)

            if len(wildjailbreak_samples) == 0:
                raise RuntimeError(
                    f"WildJailbreak loader returned 0 samples! "
                    f"Requested {samples_needed} samples but got none. "
                    f"Cannot proceed with jailbreak training."
                )

            # Apply quality filters
            wildjailbreak_filtered = wildjailbreak_samples[
                (wildjailbreak_samples['prompt'].str.len() >= WILDJAILBREAK_CONFIG['min_prompt_length']) &
                (wildjailbreak_samples['prompt'].str.len() <= WILDJAILBREAK_CONFIG['max_prompt_length']) &
                (wildjailbreak_samples['response'].str.len() >= WILDJAILBREAK_CONFIG['min_response_length']) &
                (wildjailbreak_samples['response'].str.len() <= WILDJAILBREAK_CONFIG['max_response_length'])
            ].copy()

            print(f"  ✓ Loaded {len(wildjailbreak_filtered)} WildJailbreak samples (after quality filters)")

            # Combine datasets
            combined_df = pd.concat([labeled_df_copy, wildjailbreak_filtered], ignore_index=True)

            # Calculate composition using safe_divide from Utils
            total_jailbreak_succeeded = len(combined_df[combined_df['jailbreak_label'] == 1])
            real_percentage = safe_divide(real_count, total_jailbreak_succeeded, default=0.0) * 100
            wildjailbreak_percentage = 100 - real_percentage

            print(f"\n  {'='*56}")
            print(f"  📈 JAILBREAK TRAINING DATA COMPOSITION")
            print(f"  {'='*56}")
            print(f"  Real data:        {real_count:4d} samples ({real_percentage:5.1f}%)")
            print(f"  WildJailbreak:    {len(wildjailbreak_filtered):4d} samples ({wildjailbreak_percentage:5.1f}%)")
            print(f"  Total succeeded:  {total_jailbreak_succeeded:4d} samples")
            print(f"  {'='*56}")

            # Warning if too much supplementation
            if wildjailbreak_percentage > WILDJAILBREAK_CONFIG['warn_threshold']:
                print(f"\n  ⚠️  WARNING: {wildjailbreak_percentage:.1f}% of data from WildJailbreak")
                print(f"  Consider:")
                print(f"    1. Generating more aggressive jailbreak prompts")
                print(f"    2. Using adversarial prompt engineering techniques")
                print(f"    3. Lowering min_threshold_percentage (currently {WILDJAILBREAK_CONFIG['min_threshold_percentage']}%)")

            print(f"{'='*60}\n")

            # Save augmented labeled data (with WildJailbreak samples)
            timestamp = self.run_timestamp
            labeled_augmented_path = os.path.join(data_processed_path, f"labeled_responses_{timestamp}.pkl")
            combined_df.to_pickle(labeled_augmented_path)
            print(f"✓ Saved augmented labeled data to {labeled_augmented_path}\n")

            return combined_df

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ CRITICAL ERROR: WildJailbreak supplementation failed!")
            print(f"{'='*60}")
            print(f"\nError: {e}")
            print(f"\nThis means jailbreak training CANNOT proceed because:")
            print(f"  - Real jailbreak succeeded samples: {real_count}")
            print(f"  - Threshold required: {threshold}")
            print(f"  - Shortfall: {samples_needed} samples")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"\n{'='*60}")
            print(f"SOLUTION: Fix the WildJailbreak loading error above")
            print(f"{'='*60}\n")

            # Return original data - training will be skipped due to zero samples
            return labeled_df_copy

    def train_jailbreak_detector(self, train_loader, val_loader) -> Dict:
        """Step 7: Train RoBERTa jailbreak detector (2 classes)."""
        print("\n" + "="*60)
        print("STEP 7: TRAINING JAILBREAK DETECTOR (2 CLASSES)")
        print("="*60)

        # Initialize model
        # WHY: Use len(JAILBREAK_CLASS_NAMES) for generic design (works with any N-class classifier)
        self.jailbreak_model = JailbreakDetector(num_classes=len(JAILBREAK_CLASS_NAMES))
        self.jailbreak_model.freeze_roberta_layers()
        self.jailbreak_model = self.jailbreak_model.to(DEVICE)

        print(f"\nModel: {MODEL_CONFIG['model_name']}")
        print(f"Trainable parameters: {count_parameters(self.jailbreak_model):,}")

        # Calculate class weights
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['label'].tolist())

        class_counts = [train_labels.count(i) for i in range(self.jailbreak_model.num_classes)]
        print(f"\nClass distribution in training set:")
        print(f"  Class 0 (Jailbreak Failed): {class_counts[0]}")
        print(f"  Class 1 (Jailbreak Succeeded): {class_counts[1]}")

        # CRITICAL: Cannot train with zero samples in any class
        zero_classes = [i for i, count in enumerate(class_counts) if count == 0]
        if zero_classes:
            print(f"\n{'='*60}")
            print(f"🛑 JAILBREAK DETECTOR TRAINING SKIPPED")
            print(f"{'='*60}")
            print(f"\nREASON: Zero samples in class(es): {zero_classes}")
            for i, count in enumerate(class_counts):
                status = "❌ ZERO" if i in zero_classes else f"✓ {count}"
                print(f"  Class {i} ({JAILBREAK_CLASS_NAMES[i]}): {status}")

            print(f"\nWildJailbreak supplementation should have fixed this!")
            print(f"Check the output above for WildJailbreak loading errors.")
            print(f"{'='*60}\n")

            return {'status': 'skipped', 'reason': 'zero_samples', 'class_counts': class_counts}

        # Proceed with training
        criterion = get_weighted_criterion(class_counts, DEVICE, class_names=JAILBREAK_CLASS_NAMES)

        # Optimizer and scheduler
        optimizer = AdamW(
            self.jailbreak_model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )

        num_training_steps = len(train_loader) * TRAINING_CONFIG['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=TRAINING_CONFIG['warmup_steps'],
            num_training_steps=num_training_steps
        )

        # Train
        trainer = Trainer(
            self.jailbreak_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            DEVICE
        )

        try:
            history = trainer.train(
                model_save_path=os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_jailbreak_best.pt")
            )
            print(f"\n✓ Jailbreak detector training complete")
            return history
        except Exception as e:
            print(f"\n❌ Jailbreak detector training failed: {e}")
            raise

    def run_analyses(self, test_df: pd.DataFrame) -> Dict:
        """Step 8: Run all analyses for BOTH classifiers."""
        print("\n" + "="*60)
        print("STEP 8: RUNNING ANALYSES (BOTH CLASSIFIERS)")
        print("="*60)

        # Validate models exist
        if self.refusal_model is None:
            raise RuntimeError("Refusal model not trained! Run train_refusal_classifier() first.")
        if self.jailbreak_model is None:
            raise RuntimeError("Jailbreak model not trained! Run train_jailbreak_detector() first.")

        analysis_results = {}

        # Use run timestamp for all analysis outputs
        timestamp = self.run_timestamp

        # ═══════════════════════════════════════════════════════
        # REFUSAL CLASSIFIER ANALYSIS
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("REFUSAL CLASSIFIER ANALYSIS")
        print("="*60)

        # Per-model analysis
        print("\n--- Per-Model Analysis ---")
        per_model_analyzer = PerModelAnalyzer(self.refusal_model, self.tokenizer, DEVICE)
        per_model_results = per_model_analyzer.analyze(test_df)
        per_model_analyzer.save_results(
            per_model_results,
            os.path.join(analysis_results_path, f"per_model_analysis_{timestamp}.json")
        )
        analysis_results['per_model'] = per_model_results

        # Confidence analysis
        print("\n--- Confidence Analysis ---")
        confidence_analyzer = ConfidenceAnalyzer(self.refusal_model, self.tokenizer, DEVICE)
        conf_results = confidence_analyzer.analyze(test_df)
        confidence_analyzer.save_results(
            conf_results,
            os.path.join(analysis_results_path, f"confidence_analysis_{timestamp}.json")
        )
        analysis_results['confidence'] = conf_results

        # Adversarial testing
        print("\n--- Adversarial Testing ---")
        adversarial_tester = AdversarialTester(
            self.refusal_model, self.tokenizer, DEVICE, self.api_keys['openai']
        )
        adv_results = adversarial_tester.test_robustness(test_df)
        adversarial_tester.save_results(
            adv_results,
            os.path.join(analysis_results_path, f"adversarial_testing_{timestamp}.json")
        )
        analysis_results['adversarial'] = adv_results

        # Attention visualization
        print("\n--- Attention Visualization ---")
        attention_viz = AttentionVisualizer(self.refusal_model, self.tokenizer, DEVICE, class_names=CLASS_NAMES)
        attention_results = attention_viz.analyze_samples(
            test_df,
            num_samples=INTERPRETABILITY_CONFIG['attention_samples_per_class']
        )
        analysis_results['attention'] = attention_results

        # SHAP analysis (if enabled)
        if INTERPRETABILITY_CONFIG['shap_enabled']:
            print("\n--- SHAP Analysis ---")
            try:
                shap_analyzer = ShapAnalyzer(self.refusal_model, self.tokenizer, DEVICE, class_names=CLASS_NAMES)
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

        # Power Law Analysis (Refusal Classifier)
        print("\n--- Power Law Analysis (Refusal Classifier) ---")
        power_law_analyzer = PowerLawAnalyzer(
            self.refusal_model, self.tokenizer, DEVICE, 
            class_names=CLASS_NAMES,
            model_type="Refusal Classifier"
        )
        power_law_results = power_law_analyzer.analyze_all(
            test_df,
            np.array(preds),
            np.array(confidences),
            output_dir=visualizations_path
        )
        analysis_results['power_law'] = power_law_results

        # ═══════════════════════════════════════════════════════
        # JAILBREAK DETECTOR ANALYSIS
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("JAILBREAK DETECTOR ANALYSIS")
        print("="*60)

        jailbreak_analyzer = JailbreakAnalysis(
            self.jailbreak_model,
            self.refusal_model,
            self.tokenizer,
            DEVICE
        )
        jailbreak_results = jailbreak_analyzer.analyze_full(test_df)
        jailbreak_analyzer.save_results(
            jailbreak_results,
            os.path.join(analysis_results_path, f"jailbreak_analysis_{timestamp}.json")
        )
        analysis_results['jailbreak'] = jailbreak_results

        # Power Law Analysis (Jailbreak Detector)
        print("\n--- Power Law Analysis (Jailbreak Detector) ---")
        jailbreak_power_law_analyzer = PowerLawAnalyzer(
            self.jailbreak_model, self.tokenizer, DEVICE, 
            class_names=JAILBREAK_CLASS_NAMES,
            model_type="Jailbreak Detector"
        )
        jailbreak_power_law_results = jailbreak_power_law_analyzer.analyze_all(
            test_df,
            jailbreak_results['predictions']['preds'],
            jailbreak_results['predictions']['confidences'],
            output_dir=power_law_viz_path
        )
        analysis_results['jailbreak_power_law'] = jailbreak_power_law_results

        # ═══════════════════════════════════════════════════════
        # REFUSAL-JAILBREAK CORRELATION ANALYSIS (Phase 2)
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("REFUSAL-JAILBREAK CORRELATION ANALYSIS")
        print("="*60)

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
        correlation_analyzer.save_analysis_results(
            os.path.join(analysis_results_path, f"correlation_analysis_{timestamp}.pkl")
        )
        correlation_analyzer.visualize_correlation(output_dir=correlation_viz_path)
        analysis_results['correlation'] = correlation_results

        # ═══════════════════════════════════════════════════════
        # ERROR ANALYSIS (Phase 2)
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("ERROR ANALYSIS: REFUSAL CLASSIFIER")
        print("="*60)

        # Create test dataset for error analysis
        refusal_test_dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['refusal_label'].tolist(),
            self.tokenizer
        )

        # Run comprehensive error analysis (7 modules)
        refusal_error_results = run_error_analysis(
            model=self.refusal_model,
            dataset=refusal_test_dataset,
            tokenizer=self.tokenizer,
            device=DEVICE,
            class_names=CLASS_NAMES,
            task_type='refusal'
        )
        analysis_results['error_analysis_refusal'] = refusal_error_results

        # Error analysis for jailbreak detector
        print("\n" + "="*60)
        print("ERROR ANALYSIS: JAILBREAK DETECTOR")
        print("="*60)

        jailbreak_test_dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['jailbreak_label'].tolist(),
            self.tokenizer
        )

        jailbreak_error_results = run_error_analysis(
            model=self.jailbreak_model,
            dataset=jailbreak_test_dataset,
            tokenizer=self.tokenizer,
            device=DEVICE,
            class_names=JAILBREAK_CLASS_NAMES,
            task_type='jailbreak'
        )
        analysis_results['error_analysis_jailbreak'] = jailbreak_error_results

        return analysis_results

    def generate_visualizations(self, refusal_history: Dict, jailbreak_history: Dict, analysis_results: Dict):
        """Step 9: Generate all visualizations for both classifiers."""
        print("\n" + "="*60)
        print("STEP 9: GENERATING VISUALIZATIONS")
        print("="*60)

        visualizer = Visualizer()

        # Training curves - Refusal Classifier
        visualizer.plot_training_curves(
            refusal_history,
            os.path.join(visualizations_path, "refusal_training_curves.png")
        )

        # Training curves - Jailbreak Detector
        visualizer.plot_training_curves(
            jailbreak_history,
            os.path.join(visualizations_path, "jailbreak_training_curves.png")
        )

        # Confusion matrix
        preds = analysis_results['predictions']['preds']
        labels = analysis_results['predictions']['labels']
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
        confidences = analysis_results['predictions']['confidences']
        visualizer.plot_confidence_distributions(
            labels,
            confidences,
            os.path.join(visualizations_path, "confidence_distributions.png")
        )

        print("\n✓ All visualizations generated")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: Refusal Pipeline Module
Created on October 28, 2025
@author: ramyalsaffar
"""
