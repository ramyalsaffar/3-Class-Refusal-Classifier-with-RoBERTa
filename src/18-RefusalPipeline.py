# RefusalPipeline Module
#-----------------------
# Main pipeline orchestrator for the complete refusal classification pipeline.
# All imports are in 00-Imports.py
###############################################################################


class RefusalPipeline:
    """Orchestrates the complete refusal classification pipeline."""

    def __init__(self, api_keys: Dict):
        """
        Initialize pipeline.

        Args:
            api_keys: Dictionary with keys: 'openai', 'anthropic', 'google'
        """
        self.api_keys = api_keys
        self.results = {}
        self.model = None
        self.tokenizer = None

    def run_full_pipeline(self):
        """Execute complete pipeline from start to finish."""
        print("\n" + "="*60)
        print("REFUSAL CLASSIFIER - FULL PIPELINE")
        print("="*60)
        print(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
        print("="*60 + "\n")

        # Step 1: Generate prompts
        prompts = self.generate_prompts()

        # Step 2: Collect responses
        responses_df = self.collect_responses(prompts)

        # Step 3: Label data
        labeled_df = self.label_data(responses_df)

        # Step 4: Prepare datasets
        train_loader, val_loader, test_loader, test_df = self.prepare_datasets(labeled_df)

        # Step 5: Train classifier
        history = self.train_classifier(train_loader, val_loader)

        # Step 6: Run analyses
        analysis_results = self.run_analyses(test_df)

        # Step 7: Generate visualizations
        self.generate_visualizations(history, analysis_results)

        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE")
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
        """Step 3: Label responses using LLM Judge."""
        print("\n" + "="*60)
        print("STEP 3: LABELING DATA WITH LLM JUDGE")
        print("="*60)

        # Initialize labeler with OpenAI API key (for GPT-4 judge)
        labeler = DataLabeler(api_key=self.api_keys['openai'])

        # Label each response using the judge
        labels = []
        confidences = []

        for idx, row in tqdm(responses_df.iterrows(), total=len(responses_df), desc="LLM Judge Labeling"):
            label, confidence = labeler.label_response(
                response=row['response'],
                prompt=row['prompt'],
                expected_label=row.get('expected_label', None)
            )
            labels.append(label)
            confidences.append(confidence)

        responses_df['label'] = labels
        responses_df['label_confidence'] = confidences

        # Print label distribution
        print(f"\nLabel distribution:")
        for i in range(-1, 3):
            count = (responses_df['label'] == i).sum()
            pct = count / len(responses_df) * 100
            label_name = labeler.get_label_name(i)
            print(f"  {label_name}: {count} ({pct:.1f}%)")

        # Save labeled data
        labeled_path = os.path.join(data_processed_path, "labeled_responses.pkl")
        responses_df.to_pickle(labeled_path)
        print(f"\n✓ Saved labeled data to {labeled_path}")

        return responses_df

    def prepare_datasets(self, labeled_df: pd.DataFrame):
        """Step 4: Prepare train/val/test datasets."""
        print("\n" + "="*60)
        print("STEP 4: PREPARING DATASETS")
        print("="*60)

        # Filter out error labels (-1)
        error_count = (labeled_df['label'] == -1).sum()
        if error_count > 0:
            print(f"⚠️  Filtering out {error_count} error-labeled samples")
            labeled_df = labeled_df[labeled_df['label'] != -1].copy()
            print(f"✓ Remaining samples: {len(labeled_df)}")

        # Split data
        train_df, temp_df = train_test_split(
            labeled_df,
            test_size=(1 - DATASET_CONFIG['train_split']),
            random_state=DATASET_CONFIG['random_seed'],
            stratify=labeled_df['label']
        )

        val_size = DATASET_CONFIG['val_split'] / (DATASET_CONFIG['val_split'] + DATASET_CONFIG['test_split'])
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=DATASET_CONFIG['random_seed'],
            stratify=temp_df['label']
        )

        print(f"Split sizes:")
        print(f"  Train: {len(train_df)} ({len(train_df)/len(labeled_df)*100:.1f}%)")
        print(f"  Val: {len(val_df)} ({len(val_df)/len(labeled_df)*100:.1f}%)")
        print(f"  Test: {len(test_df)} ({len(test_df)/len(labeled_df)*100:.1f}%)")

        # Save splits
        train_df.to_pickle(os.path.join(data_splits_path, "train.pkl"))
        val_df.to_pickle(os.path.join(data_splits_path, "val.pkl"))
        test_df.to_pickle(os.path.join(data_splits_path, "test.pkl"))

        # Create PyTorch datasets
        print("\nCreating PyTorch datasets...")
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

        train_dataset = RefusalDataset(
            train_df['response'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer
        )

        val_dataset = RefusalDataset(
            val_df['response'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer
        )

        test_dataset = RefusalDataset(
            test_df['response'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer
        )

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )

        print(f"✓ Dataloaders created")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader, test_df

    def train_classifier(self, train_loader, val_loader) -> Dict:
        """Step 5: Train RoBERTa classifier."""
        print("\n" + "="*60)
        print("STEP 5: TRAINING CLASSIFIER")
        print("="*60)

        # Initialize model
        self.model = RefusalClassifier()
        self.model.freeze_roberta_layers()
        self.model = self.model.to(DEVICE)

        print(f"\nModel: {MODEL_CONFIG['model_name']}")
        print(f"Trainable parameters: {count_parameters(self.model):,}")

        # Calculate class weights
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['label'].tolist())

        class_counts = [train_labels.count(i) for i in range(3)]
        print(f"\nClass distribution in training set:")
        for i, count in enumerate(class_counts):
            print(f"  Class {i} ({CLASS_NAMES[i]}): {count}")

        criterion = get_weighted_criterion(class_counts, DEVICE)

        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
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
            self.model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            DEVICE
        )

        history = trainer.train()

        return history

    def run_analyses(self, test_df: pd.DataFrame) -> Dict:
        """Step 6: Run all analyses."""
        print("\n" + "="*60)
        print("STEP 6: RUNNING ANALYSES")
        print("="*60)

        analysis_results = {}

        # Per-model analysis
        print("\n--- Per-Model Analysis ---")
        per_model_analyzer = PerModelAnalyzer(self.model, self.tokenizer, DEVICE)
        per_model_results = per_model_analyzer.analyze(test_df)
        per_model_analyzer.save_results(
            per_model_results,
            os.path.join(results_path, "per_model_analysis.json")
        )
        analysis_results['per_model'] = per_model_results

        # Confidence analysis
        print("\n--- Confidence Analysis ---")
        confidence_analyzer = ConfidenceAnalyzer(self.model, self.tokenizer, DEVICE)
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
            self.model, self.tokenizer, DEVICE, self.api_keys['openai']
        )
        adv_results = adversarial_tester.test_robustness(test_df)
        adversarial_tester.save_results(
            adv_results,
            os.path.join(results_path, "adversarial_testing.json")
        )
        analysis_results['adversarial'] = adv_results

        # Attention visualization
        print("\n--- Attention Visualization ---")
        attention_viz = AttentionVisualizer(self.model, self.tokenizer, DEVICE)
        attention_results = attention_viz.analyze_samples(
            test_df,
            num_samples=INTERPRETABILITY_CONFIG['attention_samples_per_class']
        )
        analysis_results['attention'] = attention_results

        # SHAP analysis (if enabled)
        if INTERPRETABILITY_CONFIG['shap_enabled']:
            print("\n--- SHAP Analysis ---")
            try:
                shap_analyzer = ShapAnalyzer(self.model, self.tokenizer, DEVICE)
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

        return analysis_results

    def generate_visualizations(self, history: Dict, analysis_results: Dict):
        """Step 7: Generate all visualizations."""
        print("\n" + "="*60)
        print("STEP 7: GENERATING VISUALIZATIONS")
        print("="*60)

        visualizer = Visualizer()

        # Training curves
        visualizer.plot_training_curves(
            history,
            os.path.join(visualizations_path, "training_curves.png")
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
