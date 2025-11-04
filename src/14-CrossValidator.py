# Cross-Validator Module
#------------------------
# Implements 5-fold cross-validation for robust performance evaluation.
# Uses stratified k-fold to preserve class distribution across folds.
# Reports mean ± std metrics across folds plus final test set performance.
# All imports are in 01-Imports.py
###############################################################################


# =============================================================================
# CROSS-VALIDATION TRAINER
# =============================================================================

class CrossValidator:
    """
    K-Fold Cross-Validation for classification models.

    Implements stratified k-fold CV to ensure class balance across folds.
    Trains k models and reports aggregated metrics with confidence intervals.
    """

    def __init__(self,
                 model_class,
                 dataset,
                 k_folds: int = 5,
                 device: torch.device = None,
                 class_names: List[str] = None,
                 random_state: int = 42):
        """
        Initialize cross-validator.

        Args:
            model_class: Class to instantiate model (RefusalClassifier or JailbreakDetector)
            dataset: Full dataset (before train/test split)
            k_folds: Number of folds (default: 5)
            device: torch device
            class_names: List of class names for display
            random_state: Random seed for reproducibility
        """
        self.model_class = model_class
        self.dataset = dataset
        self.k_folds = k_folds
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(len(set(dataset.labels)))]
        self.random_state = random_state

        # Results storage
        self.fold_results = []
        self.cv_metrics = {
            'accuracy': [],
            'f1_macro': [],
            'precision_macro': [],
            'recall_macro': []
        }

        # Per-class metrics (for each fold)
        self.per_class_metrics = {
            'precision': [[] for _ in range(len(self.class_names))],
            'recall': [[] for _ in range(len(self.class_names))],
            'f1': [[] for _ in range(len(self.class_names))]
        }

        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SETUP")
        print(f"{'='*60}")
        print(f"K-Folds: {self.k_folds}")
        print(f"Total samples: {len(self.dataset)}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Device: {self.device}")
        print(f"Random state: {self.random_state}")
        print(f"{'='*60}\n")


    def run_cross_validation(self, save_fold_models: bool = False) -> Dict:
        """
        Run k-fold cross-validation.

        Args:
            save_fold_models: If True, save model for each fold

        Returns:
            Dictionary with CV results and statistics
        """
        # Create stratified k-fold splitter
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)

        # Get all labels for stratification
        all_labels = np.array([self.dataset.labels[i] for i in range(len(self.dataset))])

        print(f"\n{'='*60}")
        print(f"RUNNING {self.k_folds}-FOLD CROSS-VALIDATION")
        print(f"{'='*60}\n")

        # Iterate through folds
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), 1):
            print(f"\n{'-'*60}")
            print(f"FOLD {fold_idx}/{self.k_folds}")
            print(f"{'-'*60}")
            print(f"Train samples: {len(train_idx)}")
            print(f"Val samples: {len(val_idx)}")

            # Create fold datasets
            train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
            val_dataset = torch.utils.data.Subset(self.dataset, val_idx)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=TRAINING_CONFIG['batch_size'],
                shuffle=True,
                num_workers=TRAINING_CONFIG['num_workers']
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=TRAINING_CONFIG['batch_size'],
                shuffle=False,
                num_workers=TRAINING_CONFIG['num_workers']
            )

            # Calculate class counts for weighted loss
            train_labels = [all_labels[i] for i in train_idx]
            class_counts = [train_labels.count(i) for i in range(len(self.class_names))]

            print(f"\nClass distribution in fold {fold_idx}:")
            for i, (name, count) in enumerate(zip(self.class_names, class_counts)):
                print(f"  {name}: {count} ({count/len(train_idx)*100:.1f}%)")

            # Initialize model
            model = self.model_class(num_classes=len(self.class_names)).to(self.device)

            # Setup training components
            criterion = get_weighted_criterion(class_counts, self.device, self.class_names)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=TRAINING_CONFIG['learning_rate'],
                weight_decay=TRAINING_CONFIG['weight_decay']
            )

            # Learning rate scheduler
            num_training_steps = len(train_loader) * TRAINING_CONFIG['epochs']
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps
            )

            # Train fold
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device
            )

            # Train with early stopping
            fold_save_path = None
            if save_fold_models:
                fold_save_path = os.path.join(
                    models_path,
                    f"{EXPERIMENT_CONFIG['experiment_name']}_fold{fold_idx}.pt"
                )

            history = trainer.train(model_save_path=fold_save_path)

            # Evaluate fold
            model.eval()
            all_preds = []
            all_labels_val = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    logits = model(input_ids, attention_mask)
                    preds = torch.argmax(logits, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels_val.extend(labels.cpu().numpy())

            # Calculate metrics
            accuracy = accuracy_score(all_labels_val, all_preds)
            f1_macro = f1_score(all_labels_val, all_preds, average='macro')
            precision_macro = precision_score(all_labels_val, all_preds, average='macro')
            recall_macro = recall_score(all_labels_val, all_preds, average='macro')

            # Per-class metrics
            report = classification_report(
                all_labels_val,
                all_preds,
                target_names=self.class_names,
                output_dict=True
            )

            # Store overall metrics
            self.cv_metrics['accuracy'].append(accuracy)
            self.cv_metrics['f1_macro'].append(f1_macro)
            self.cv_metrics['precision_macro'].append(precision_macro)
            self.cv_metrics['recall_macro'].append(recall_macro)

            # Store per-class metrics
            for i, class_name in enumerate(self.class_names):
                self.per_class_metrics['precision'][i].append(report[class_name]['precision'])
                self.per_class_metrics['recall'][i].append(report[class_name]['recall'])
                self.per_class_metrics['f1'][i].append(report[class_name]['f1-score'])

            # Print fold results
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx} RESULTS")
            print(f"{'='*60}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 (macro): {f1_macro:.4f}")
            print(f"Precision (macro): {precision_macro:.4f}")
            print(f"Recall (macro): {recall_macro:.4f}")
            print(f"\nPer-class metrics:")
            for class_name in self.class_names:
                print(f"  {class_name}:")
                print(f"    Precision: {report[class_name]['precision']:.4f}")
                print(f"    Recall: {report[class_name]['recall']:.4f}")
                print(f"    F1: {report[class_name]['f1-score']:.4f}")
            print(f"{'='*60}\n")

            # Store fold result
            self.fold_results.append({
                'fold': fold_idx,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'per_class': report,
                'history': history
            })

        # Calculate statistics across folds
        cv_summary = self._calculate_cv_statistics()

        # Print CV summary
        self._print_cv_summary(cv_summary)

        return cv_summary


    def _calculate_cv_statistics(self) -> Dict:
        """
        Calculate mean and std across folds.

        Returns:
            Dictionary with CV statistics
        """
        stats = {
            'overall': {},
            'per_class': {}
        }

        # Overall metrics
        for metric_name, values in self.cv_metrics.items():
            stats['overall'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }

        # Per-class metrics
        for metric_type in ['precision', 'recall', 'f1']:
            stats['per_class'][metric_type] = {}
            for i, class_name in enumerate(self.class_names):
                values = self.per_class_metrics[metric_type][i]
                stats['per_class'][metric_type][class_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }

        # Add fold results
        stats['fold_results'] = self.fold_results

        return stats


    def _print_cv_summary(self, cv_summary: Dict):
        """Print cross-validation summary."""
        print(f"\n{'='*60}")
        print(f"{self.k_folds}-FOLD CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}\n")

        print("Overall Metrics (Mean ± Std):")
        print(f"{'-'*60}")
        for metric_name, stats in cv_summary['overall'].items():
            print(f"{metric_name.replace('_', ' ').title():20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"{'':20s}  [min: {stats['min']:.4f}, max: {stats['max']:.4f}]")

        print(f"\n{'='*60}")
        print("Per-Class Metrics (Mean ± Std):")
        print(f"{'='*60}\n")

        for class_name in self.class_names:
            print(f"{class_name}:")
            for metric_type in ['precision', 'recall', 'f1']:
                stats = cv_summary['per_class'][metric_type][class_name]
                print(f"  {metric_type.title():10s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print()

        print(f"{'='*60}\n")

        # Print fold-by-fold comparison
        print("Fold-by-Fold Comparison:")
        print(f"{'-'*60}")
        print(f"{'Fold':<8} {'Accuracy':<12} {'F1 (macro)':<12} {'Precision':<12} {'Recall':<12}")
        print(f"{'-'*60}")
        for fold_result in self.fold_results:
            print(f"{fold_result['fold']:<8} "
                  f"{fold_result['accuracy']:<12.4f} "
                  f"{fold_result['f1_macro']:<12.4f} "
                  f"{fold_result['precision_macro']:<12.4f} "
                  f"{fold_result['recall_macro']:<12.4f}")
        print(f"{'-'*60}\n")


    def save_cv_results(self, output_path: str = None):
        """
        Save cross-validation results to file.

        Args:
            output_path: Path to save results (default: results/cv_results.pkl)
        """
        if output_path is None:
            output_path = os.path.join(
                results_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_cv_results.pkl"
            )

        cv_summary = self._calculate_cv_statistics()

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(cv_summary, f)

        print(f"✓ Cross-validation results saved to: {output_path}")

        return output_path


# =============================================================================
# TRAIN/VAL/TEST SPLIT WITH CV
# =============================================================================

def train_with_cross_validation(
    full_dataset,
    model_class,
    k_folds: int = 5,
    test_split: float = 0.2,
    class_names: List[str] = None,
    save_final_model: bool = True,
    final_model_path: str = None
) -> Dict:
    """
    Complete training pipeline with cross-validation and held-out test set.

    Pipeline:
        1. Split data: 80% train+val, 20% test (held-out)
        2. Run k-fold CV on 80% train+val
        3. Train final model on full 80% with best hyperparameters
        4. Evaluate final model on held-out 20% test set

    Args:
        full_dataset: Complete dataset
        model_class: Model class to instantiate
        k_folds: Number of CV folds
        test_split: Fraction for held-out test set
        class_names: List of class names
        save_final_model: Whether to save final model
        final_model_path: Path to save final model

    Returns:
        Dictionary with CV results and final test performance
    """
    print(f"\n{'='*60}")
    print("TRAIN/VAL/TEST SPLIT WITH CROSS-VALIDATION")
    print(f"{'='*60}")
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Test split: {test_split*100:.0f}%")
    print(f"Train+Val split: {(1-test_split)*100:.0f}%")
    print(f"Cross-validation folds: {k_folds}")
    print(f"{'='*60}\n")

    # Get all indices and labels
    all_indices = list(range(len(full_dataset)))
    all_labels = [full_dataset.labels[i] for i in all_indices]

    # Stratified split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_split,
        stratify=all_labels,
        random_state=EXPERIMENT_CONFIG['random_seed']
    )

    print(f"Split sizes:")
    print(f"  Train+Val: {len(train_val_idx)} samples ({len(train_val_idx)/len(full_dataset)*100:.1f}%)")
    print(f"  Test (held-out): {len(test_idx)} samples ({len(test_idx)/len(full_dataset)*100:.1f}%)")

    # Create train+val dataset for CV
    train_val_dataset = torch.utils.data.Subset(full_dataset, train_val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    # STEP 1: Run cross-validation on train+val
    print(f"\n{'#'*60}")
    print("STEP 1: CROSS-VALIDATION ON TRAIN+VAL SET")
    print(f"{'#'*60}\n")

    cv = CrossValidator(
        model_class=model_class,
        dataset=train_val_dataset,
        k_folds=k_folds,
        class_names=class_names
    )

    cv_results = cv.run_cross_validation(save_fold_models=False)

    # Save CV results
    cv_results_path = cv.save_cv_results()

    # STEP 2: Train final model on full train+val set
    print(f"\n{'#'*60}")
    print("STEP 2: TRAIN FINAL MODEL ON FULL TRAIN+VAL SET")
    print(f"{'#'*60}\n")

    # Create final train loader (full train+val)
    final_train_loader = DataLoader(
        train_val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers']
    )

    # For validation during final training, use a small split from train+val
    # This is just for monitoring, not for hyperparameter selection
    final_train_idx, final_val_idx = train_test_split(
        list(range(len(train_val_dataset))),
        test_size=CROSS_VALIDATION_CONFIG['final_val_split'],
        stratify=[all_labels[train_val_idx[i]] for i in range(len(train_val_dataset))],
        random_state=EXPERIMENT_CONFIG['random_seed']
    )

    final_train_subset = torch.utils.data.Subset(train_val_dataset, final_train_idx)
    final_val_subset = torch.utils.data.Subset(train_val_dataset, final_val_idx)

    final_train_loader_split = DataLoader(final_train_subset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True, num_workers=TRAINING_CONFIG['num_workers'])
    final_val_loader_split = DataLoader(final_val_subset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False, num_workers=TRAINING_CONFIG['num_workers'])

    # Calculate class counts for final model
    final_labels = [all_labels[train_val_idx[i]] for i in final_train_idx]
    final_class_counts = [final_labels.count(i) for i in range(len(class_names))]

    # Initialize final model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model = model_class(num_classes=len(class_names)).to(device)

    # Setup training
    criterion = get_weighted_criterion(final_class_counts, device, class_names)
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    num_training_steps = len(final_train_loader_split) * TRAINING_CONFIG['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    # Train final model
    final_trainer = Trainer(
        model=final_model,
        train_loader=final_train_loader_split,
        val_loader=final_val_loader_split,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    if final_model_path is None:
        final_model_path = os.path.join(
            models_path,
            f"{EXPERIMENT_CONFIG['experiment_name']}_final.pt"
        )

    final_history = final_trainer.train(model_save_path=final_model_path if save_final_model else None)

    # STEP 3: Evaluate on held-out test set
    print(f"\n{'#'*60}")
    print("STEP 3: EVALUATE ON HELD-OUT TEST SET")
    print(f"{'#'*60}\n")

    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False, num_workers=TRAINING_CONFIG['num_workers'])

    final_model.eval()
    all_preds = []
    all_labels_test = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = final_model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())

    # Calculate test metrics
    test_accuracy = accuracy_score(all_labels_test, all_preds)
    test_f1 = f1_score(all_labels_test, all_preds, average='macro')
    test_report = classification_report(
        all_labels_test,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    # Print test results
    print(f"{'='*60}")
    print("FINAL TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    print(f"\nPer-class metrics:")
    for class_name in class_names:
        print(f"  {class_name}:")
        print(f"    Precision: {test_report[class_name]['precision']:.4f}")
        print(f"    Recall: {test_report[class_name]['recall']:.4f}")
        print(f"    F1: {test_report[class_name]['f1-score']:.4f}")
        print(f"    Support: {test_report[class_name]['support']:.0f}")
    print(f"{'='*60}\n")

    # Compare CV vs Test
    print(f"{'='*60}")
    print("CROSS-VALIDATION vs TEST SET COMPARISON")
    print(f"{'='*60}")
    print(f"CV Accuracy (mean ± std): {cv_results['overall']['accuracy']['mean']:.4f} ± {cv_results['overall']['accuracy']['std']:.4f}")
    print(f"Test Set Accuracy:        {test_accuracy:.4f}")
    print(f"\nCV F1 (mean ± std):       {cv_results['overall']['f1_macro']['mean']:.4f} ± {cv_results['overall']['f1_macro']['std']:.4f}")
    print(f"Test Set F1:              {test_f1:.4f}")
    print(f"{'='*60}\n")

    # Return complete results
    return {
        'cv_results': cv_results,
        'cv_results_path': cv_results_path,
        'final_model_path': final_model_path if save_final_model else None,
        'final_history': final_history,
        'test_results': {
            'accuracy': test_accuracy,
            'f1_macro': test_f1,
            'report': test_report,
            'predictions': all_preds,
            'labels': all_labels_test
        },
        'split_info': {
            'train_val_size': len(train_val_idx),
            'test_size': len(test_idx),
            'k_folds': k_folds,
            'test_indices': test_idx  # WHY: Save test indices for proper dataset reconstruction in error analysis
        }
    }


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 1, 2025
@author: ramyalsaffar
"""
