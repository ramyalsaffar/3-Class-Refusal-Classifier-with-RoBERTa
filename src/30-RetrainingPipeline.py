# RetrainingPipeline Module
#---------------------------
# Automated retraining with validation, A/B testing, and safe deployment.
# Triggered when monitoring detects performance degradation.
# NOTE: This is a standalone production script - core imports from 01-Imports.py
###############################################################################

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Automated retraining pipeline with A/B testing and validation."""

    def __init__(self, data_manager: 'DataManager'):
        """
        Initialize retraining pipeline.

        Args:
            data_manager: DataManager instance
        """
        self.data_manager = data_manager
        self.ab_test_stages = PRODUCTION_CONFIG['ab_test_stages']

    def run_retraining(self, reason: str = "performance_degradation") -> Dict:
        """
        Run complete retraining pipeline.

        Args:
            reason: Reason for retraining

        Returns:
            Dictionary with retraining results
        """
        print("\n" + "="*80)
        print(" "*28 + "RETRAINING PIPELINE")
        print("="*80)
        print(f"Reason: {reason}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Collect training data
        print("\n" + "="*80)
        print("STEP 1: COLLECTING TRAINING DATA")
        print("="*80)

        training_df = self.data_manager.get_retraining_data()

        if len(training_df) < 100:
            print(f"❌ Insufficient data for retraining: {len(training_df)} samples")
            print("   Need at least 100 samples")
            return {'status': 'failed', 'reason': 'insufficient_data'}

        # Step 2: Prepare datasets
        print("\n" + "="*80)
        print("STEP 2: PREPARING DATASETS")
        print("="*80)

        train_df, val_df, test_df = self._split_data(training_df)

        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")

        # Create dataloaders
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

        train_dataset = ClassificationDataset(
            train_df['response'].tolist(),
            train_df['label'].tolist(),
            tokenizer
        )
        val_dataset = ClassificationDataset(
            val_df['response'].tolist(),
            val_df['label'].tolist(),
            tokenizer
        )
        test_dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['label'].tolist(),
            tokenizer
        )

        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )

        # Step 3: Train new model
        print("\n" + "="*80)
        print("STEP 3: TRAINING NEW MODEL")
        print("="*80)
        logger.info("Starting model training")

        # GENERIC: Determine number of classes from data
        num_classes = len(train_df['label'].unique())
        logger.info(f"Training model with {num_classes} classes")

        model = RefusalClassifier(num_classes=num_classes).to(DEVICE)

        # Load current model weights as starting point (transfer learning)
        current_version = self.data_manager.get_active_model_version()
        if current_version:
            current_path = os.path.join(models_path, f"{current_version}.pt")
            if os.path.exists(current_path):
                print(f"✓ Loading current model {current_version} as starting point")
                logger.info(f"Loading checkpoint from {current_version}")
                checkpoint = torch.load(current_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])

        # Freeze fewer layers for fine-tuning (allow more adaptation)
        model.freeze_roberta_layers(num_layers_to_freeze=4)  # Less freezing than initial training

        # Training setup
        train_labels = [label for _, label in train_dataset]
        # GENERIC: Dynamic class count
        class_counts = [train_labels.count(i) for i in range(num_classes)]
        criterion = get_weighted_criterion(class_counts, DEVICE)

        optimizer = AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'] * 0.5,  # Lower LR for fine-tuning
            weight_decay=TRAINING_CONFIG['weight_decay']
        )

        num_training_steps = len(train_loader) * TRAINING_CONFIG['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=TRAINING_CONFIG['warmup_steps'],
            num_training_steps=num_training_steps
        )

        # Train with error handling
        try:
            trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE)
            history = trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'status': 'failed',
                'reason': 'training_failed',
                'error': str(e)
            }

        # Step 4: Validate new model
        print("\n" + "="*80)
        print("STEP 4: VALIDATING NEW MODEL")
        print("="*80)

        validation_passed, validation_metrics = self._validate_new_model(
            model, test_loader, tokenizer, test_df
        )

        if not validation_passed:
            print("❌ New model failed validation - aborting deployment")
            return {
                'status': 'failed',
                'reason': 'validation_failed',
                'metrics': validation_metrics
            }

        # Step 5: Save new model
        print("\n" + "="*80)
        print("STEP 5: SAVING NEW MODEL")
        print("="*80)

        new_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = os.path.join(models_path, f"{new_version}.pt")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'num_classes': num_classes,  # ADDED: Save num_classes for generic loading
            'training_history': history,
            'validation_metrics': validation_metrics,
            'training_date': datetime.now().isoformat(),
            'reason': reason
        }, model_path)
        logger.info(f"Model checkpoint saved to {model_path}")

        print(f"✓ Saved new model: {new_version}")

        # Register in database
        self.data_manager.register_model_version(
            version=new_version,
            f1_score=validation_metrics['f1_score'],
            is_active=False,  # Not active yet - needs A/B testing
            is_challenger=True,  # Mark as challenger
            metadata={
                'training_samples': len(train_df),
                'reason': reason,
                'parent_version': current_version
            }
        )

        # Step 6: Deploy with A/B testing
        print("\n" + "="*80)
        print("STEP 6: DEPLOYING WITH A/B TESTING")
        print("="*80)

        ab_test_result = self._run_ab_test(new_version, validation_metrics)

        return {
            'status': 'completed',
            'new_version': new_version,
            'validation_metrics': validation_metrics,
            'ab_test_result': ab_test_result
        }

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test.

        Args:
            df: Full dataset

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            random_state=DATASET_CONFIG['random_seed'],
            stratify=df['label']
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=DATASET_CONFIG['random_seed'],
            stratify=temp_df['label']
        )

        return train_df, val_df, test_df

    def _validate_new_model(self, model, test_loader, tokenizer, test_df) -> Tuple[bool, Dict]:
        """
        Validate new model meets minimum requirements.

        Args:
            model: New model to validate
            test_loader: Test data loader
            tokenizer: Tokenizer
            test_df: Test DataFrame

        Returns:
            Tuple of (passed, metrics)
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_confs = []

        print("\nRunning validation...")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label']

                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                confidence, preds = torch.max(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_confs.extend(confidence.cpu().numpy())

        # Calculate metrics
        f1 = f1_score(all_labels, all_preds, average='weighted')
        accuracy = accuracy_score(all_labels, all_preds)
        per_class_f1 = f1_score(all_labels, all_preds, average=None)

        # GENERIC: Dynamic class count
        num_classes = model.num_classes
        metrics = {
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'avg_confidence': float(np.mean(all_confs)),
            'per_class_f1': {CLASS_NAMES[i]: float(per_class_f1[i]) for i in range(num_classes)}
        }

        print(f"\nValidation Results:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Avg Confidence: {np.mean(all_confs):.4f}")

        # Validation checks
        min_f1 = PRODUCTION_CONFIG['validation_thresholds']['min_f1_score']
        min_confidence = PRODUCTION_CONFIG['validation_thresholds']['min_avg_confidence']

        passed = f1 >= min_f1 and np.mean(all_confs) >= min_confidence

        if passed:
            print(f"✅ Validation PASSED")
        else:
            print(f"❌ Validation FAILED")
            if f1 < min_f1:
                print(f"   F1 score {f1:.4f} below threshold {min_f1}")
            if np.mean(all_confs) < min_confidence:
                print(f"   Confidence {np.mean(all_confs):.4f} below threshold {min_confidence}")

        return passed, metrics

    def _run_ab_test(self, new_version: str, new_metrics: Dict) -> Dict:
        """
        Run A/B test with gradual rollout.

        Args:
            new_version: New model version
            new_metrics: Validation metrics for new model

        Returns:
            A/B test results
        """
        print(f"\nStarting A/B test for {new_version}")
        print(f"Gradual rollout stages: {self.ab_test_stages}")
        logger.info(f"Deploying {new_version} for A/B testing")

        # Set initial traffic percentage with error handling
        initial_traffic = self.ab_test_stages[0]

        try:
            cursor = self.data_manager.conn.cursor()
            cursor.execute("""
                UPDATE model_versions
                SET traffic_percentage = %s
                WHERE version = %s
            """, (initial_traffic, new_version))
            self.data_manager.conn.commit()
            cursor.close()
            logger.info(f"Set traffic percentage to {initial_traffic*100}%")
        except Exception as e:
            logger.error(f"Failed to set A/B test traffic: {e}")
            self.data_manager.conn.rollback()
            raise

        print(f"\n✓ Deployed challenger with {initial_traffic*100}% traffic")
        print(f"\n{'='*80}")
        print("A/B TEST ACTIVE")
        print("="*80)
        print(f"\nChallenger: {new_version} ({initial_traffic*100}% traffic)")
        print(f"Active: {self.data_manager.get_active_model_version()} ({(1-initial_traffic)*100}% traffic)")
        print(f"\nNext steps:")
        print(f"1. Monitor performance for 24-48 hours")
        print(f"2. If successful, manually increase traffic:")
        print(f"   - Use admin endpoint to promote or rollback")
        print(f"3. Gradual rollout: {' → '.join([f'{int(x*100)}%' for x in self.ab_test_stages])}")

        return {
            'status': 'ab_test_started',
            'challenger_version': new_version,
            'initial_traffic': initial_traffic,
            'stages': self.ab_test_stages
        }

    def manual_increase_traffic(self, challenger_version: str, new_traffic: float):
        """
        Manually increase traffic to challenger (gradual rollout).

        Args:
            challenger_version: Challenger version
            new_traffic: New traffic percentage (0.0-1.0)
        """
        if new_traffic > 1.0 or new_traffic < 0.0:
            raise ValueError("Traffic must be between 0.0 and 1.0")

        try:
            cursor = self.data_manager.conn.cursor()
            cursor.execute("""
                UPDATE model_versions
                SET traffic_percentage = %s
                WHERE version = %s AND is_challenger = TRUE
            """, (new_traffic, challenger_version))
            self.data_manager.conn.commit()
            cursor.close()
            logger.info(f"Increased traffic to {challenger_version}: {new_traffic*100}%")
            print(f"✓ Increased traffic to {challenger_version}: {new_traffic*100}%")
        except Exception as e:
            logger.error(f"Failed to increase traffic: {e}")
            self.data_manager.conn.rollback()
            raise

        if new_traffic >= 1.0:
            print("   Challenger is now receiving 100% of traffic")
            print("   Consider promoting to active model")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 30, 2025
@author: ramyalsaffar
"""
