# RefusalPipeline Module
#-----------------------
# Main pipeline orchestrator for the complete refusal classification pipeline.
# Trains TWO independent classifiers: Refusal Classifier + Jailbreak Detector.
# All imports are in 01-Imports.py
###############################################################################


class RefusalPipeline:
    """Orchestrates the complete refusal classification pipeline with dual classifiers."""

    def __init__(self, api_keys: Dict):
        """
        Initialize pipeline.

        Args:
            api_keys: Dictionary with keys: 'openai', 'anthropic', 'google'
        """
        self.api_keys = api_keys
        self.results = {}
        self.refusal_model = None
        self.jailbreak_model = None
        self.tokenizer = None

    def run_full_pipeline(self):
        """Execute complete pipeline from start to finish."""
        print("\n" + "="*60)
        print("REFUSAL CLASSIFIER - FULL PIPELINE (DUAL CLASSIFIERS)")
        print("="*60)
        print(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
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

        # Step 5: Prepare datasets for BOTH classifiers
        datasets = self.prepare_datasets(labeled_df)

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
        print("\n" + "="*60)
        print("STEP 1: GENERATING PROMPTS")
        print("="*60)

        generator = PromptGenerator(self.api_keys['openai'])
        prompts = generator.generate_all_prompts()
        generator.save_prompts(prompts, data_raw_path)

        return prompts

    def collect_responses(self, prompts: Dict[str, List[str]]) -> pd.DataFrame:
        """Step 2: Collect LLM responses."""
        print("\n" + "="*60)
        print("STEP 2: COLLECTING RESPONSES")
        print("="*60)

        collector = ResponseCollector(
            self.api_keys['anthropic'],
            self.api_keys['openai'],
            self.api_keys['google']
        )
        responses_df = collector.collect_all_responses(prompts)
        collector.save_responses(responses_df, data_responses_path)

        return responses_df

    def label_data(self, responses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Label responses using LLM Judge (dual-task).

        Labels ONLY clean data (after cleaning), saving API costs.
        """
        print("\n" + "="*60)
        print("STEP 4: LABELING DATA WITH LLM JUDGE (DUAL-TASK)")
        print("="*60)

        # Initialize labeler with OpenAI API key (for GPT-4 judge)
        labeler = DataLabeler(api_key=self.api_keys['openai'])

        # Label each response using the judge (returns labels + confidence scores)
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
            pct = count / len(responses_df) * 100
            label_name = labeler.get_label_name(i)
            print(f"  {label_name}: {count} ({pct:.1f}%)")

        # Print jailbreak label distribution
        print(f"\n{'='*60}")
        print(f"JAILBREAK DETECTION SUMMARY")
        print(f"{'='*60}")
        for i in [0, 1]:
            count = (responses_df['jailbreak_label'] == i).sum()
            pct = count / len(responses_df) * 100
            label_name = labeler.get_jailbreak_label_name(i)
            print(f"  {label_name}: {count} ({pct:.1f}%)")

        # Print confidence statistics
        print(f"\n{'='*60}")
        print(f"CONFIDENCE STATISTICS")
        print(f"{'='*60}")
        valid_refusal = responses_df[responses_df['refusal_label'] != -1]
        valid_jailbreak = responses_df[responses_df['jailbreak_label'] != -1]

        if len(valid_refusal) > 0:
            avg_ref_conf = valid_refusal['refusal_confidence'].mean()
            low_conf_ref = (valid_refusal['refusal_confidence'] < 60).sum()
            print(f"  Refusal - Avg Confidence: {avg_ref_conf:.1f}%")
            print(f"  Refusal - Low confidence (<60%): {low_conf_ref} ({low_conf_ref/len(valid_refusal)*100:.1f}%)")

        if len(valid_jailbreak) > 0:
            avg_jb_conf = valid_jailbreak['jailbreak_confidence'].mean()
            low_conf_jb = (valid_jailbreak['jailbreak_confidence'] < 60).sum()
            print(f"  Jailbreak - Avg Confidence: {avg_jb_conf:.1f}%")
            print(f"  Jailbreak - Low confidence (<60%): {low_conf_jb} ({low_conf_jb/len(valid_jailbreak)*100:.1f}%)")

        # Analyze labeling quality
        print(f"\n{'='*60}")
        print("LABELING QUALITY ANALYSIS")
        print(f"{'='*60}")
        quality_analyzer = LabelingQualityAnalyzer(verbose=True)
        quality_results = quality_analyzer.analyze_full(responses_df)

        # Save quality analysis results
        quality_analysis_path = os.path.join(results_path, "labeling_quality_analysis.json")
        quality_analyzer.save_results(quality_results, quality_analysis_path)

        # Export low-confidence samples for review
        if quality_results['low_confidence']['low_both_count'] > 0:
            flagged_samples_path = os.path.join(results_path, "flagged_samples_for_review.csv")
            quality_analyzer.export_flagged_samples(responses_df, flagged_samples_path, threshold=60)

        # Save labeled data
        labeled_path = os.path.join(data_processed_path, "labeled_responses.pkl")
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
        cleaner = DataCleaner(verbose=True)

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
        cleaned_path = os.path.join(data_processed_path, "cleaned_responses.pkl")
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

        # Filter out error labels (-1) from refusal labels
        error_count = (labeled_df['refusal_label'] == -1).sum()
        if error_count > 0:
            print(f"⚠️  Filtering out {error_count} error-labeled samples")
            labeled_df = labeled_df[labeled_df['refusal_label'] != -1].copy()
            print(f"✓ Remaining samples: {len(labeled_df)}")

        # Split data (same splits for both classifiers to maintain consistency)
        train_df, temp_df = train_test_split(
            labeled_df,
            test_size=(1 - DATASET_CONFIG['train_split']),
            random_state=DATASET_CONFIG['random_seed'],
            stratify=labeled_df['refusal_label']  # Stratify by refusal label
        )

        val_size = DATASET_CONFIG['val_split'] / (DATASET_CONFIG['val_split'] + DATASET_CONFIG['test_split'])
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
        train_df.to_pickle(os.path.join(data_splits_path, "train.pkl"))
        val_df.to_pickle(os.path.join(data_splits_path, "val.pkl"))
        test_df.to_pickle(os.path.join(data_splits_path, "test.pkl"))

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
        self.refusal_model = RefusalClassifier(num_classes=3)
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

    def train_jailbreak_detector(self, train_loader, val_loader) -> Dict:
        """Step 7: Train RoBERTa jailbreak detector (2 classes)."""
        print("\n" + "="*60)
        print("STEP 7: TRAINING JAILBREAK DETECTOR (2 CLASSES)")
        print("="*60)

        # Initialize model
        self.jailbreak_model = JailbreakDetector(num_classes=2)
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

        criterion = get_weighted_criterion(class_counts, DEVICE)

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
            os.path.join(results_path, "per_model_analysis.json")
        )
        analysis_results['per_model'] = per_model_results

        # Confidence analysis
        print("\n--- Confidence Analysis ---")
        confidence_analyzer = ConfidenceAnalyzer(self.refusal_model, self.tokenizer, DEVICE)
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
        print("\n--- Adversarial Testing ---")
        adversarial_tester = AdversarialTester(
            self.refusal_model, self.tokenizer, DEVICE, self.api_keys['openai']
        )
        adv_results = adversarial_tester.test_robustness(test_df)
        adversarial_tester.save_results(
            adv_results,
            os.path.join(results_path, "adversarial_testing.json")
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

        # Power Law Analysis
        print("\n--- Power Law Analysis ---")
        power_law_analyzer = PowerLawAnalyzer(self.refusal_model, self.tokenizer, DEVICE, class_names=CLASS_NAMES)
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
            os.path.join(results_path, "jailbreak_analysis.json")
        )
        analysis_results['jailbreak'] = jailbreak_results

        # Power Law Analysis (Jailbreak Detector)
        print("\n--- Power Law Analysis (Jailbreak) ---")
        jailbreak_class_names = ["Jailbreak Failed", "Jailbreak Succeeded"]
        jailbreak_power_law_analyzer = PowerLawAnalyzer(
            self.jailbreak_model, self.tokenizer, DEVICE, class_names=jailbreak_class_names
        )
        jailbreak_power_law_results = jailbreak_power_law_analyzer.analyze_all(
            test_df,
            jailbreak_results['predictions']['preds'],
            jailbreak_results['predictions']['confidences'],
            output_dir=visualizations_path
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
            jailbreak_class_names=["Jailbreak Failed", "Jailbreak Succeeded"]
        )
        correlation_results = correlation_analyzer.analyze_full()
        correlation_analyzer.save_results(
            correlation_results,
            os.path.join(results_path, "correlation_analysis.pkl")
        )
        correlation_analyzer.visualize_correlation(output_dir=visualizations_path)
        analysis_results['correlation'] = correlation_results

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
Created on October 28, 2025
@author: ramyalsaffar
"""
