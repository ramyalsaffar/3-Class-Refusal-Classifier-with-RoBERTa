# Cross-Validator Module 
#------------------------
# Enhanced k-fold cross-validation for robust performance evaluation.
# Uses stratified k-fold to preserve class distribution across folds.
# Reports mean ± std metrics with confidence intervals and significance testing.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Statistical significance testing
# - Confidence interval calculation
# - Mixed precision support
# - Checkpoint support for resuming
# - Better visualization
# All imports are in 01-Imports.py
###############################################################################


class CrossValidator:
    """
    Enhanced K-Fold Cross-Validation for classification models.
    
    Features:
    - Stratified k-fold for class balance
    - Statistical significance testing
    - Confidence intervals
    - Per-class and overall metrics
    - Visualization support
    """
    
    def __init__(self,
                 model_class,
                 dataset,
                 k_folds: int = None,
                 device: torch.device = None,
                 class_names: List[str] = None,
                 random_state: int = None,
                 use_mixed_precision: bool = None):
        """
        Initialize enhanced cross-validator.
        
        Args:
            model_class: Class to instantiate model (RefusalClassifier or JailbreakClassifier)
            dataset: Full dataset (before train/test split)
            k_folds: Number of folds (uses config if None)
            device: torch device
            class_names: List of class names for display
            random_state: Random seed (uses config if None)
            use_mixed_precision: Use AMP for training
        """
        self.model_class = model_class
        self.dataset = dataset
        
        # Use config values - NO HARDCODING!
        self.k_folds = k_folds or CROSS_VAL_CONFIG.get('k_folds', 5)
        self.random_state = random_state or EXPERIMENT_CONFIG.get('random_seed', 42)
        self.use_mixed_precision = use_mixed_precision or TRAINING_CONFIG.get('mixed_precision', False)
        self.verbose = EXPERIMENT_CONFIG.get('verbose', True)
        
        # Device setup
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class names
        unique_labels = len(set(dataset.labels))
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(unique_labels)]
        
        # Results storage
        self.fold_results = []
        self.cv_metrics = {
            'accuracy': [],
            'f1_macro': [],
            'f1_weighted': [],
            'precision_macro': [],
            'recall_macro': [],
            'loss': []
        }
        
        # Per-class metrics
        self.per_class_metrics = {
            'precision': [[] for _ in range(len(self.class_names))],
            'recall': [[] for _ in range(len(self.class_names))],
            'f1': [[] for _ in range(len(self.class_names))]
        }
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            operation_name='cross_validation'
        )
        
        if self.verbose:
            self._print_cv_setup()
    
    def _print_cv_setup(self):
        """Print cross-validation setup information."""
        print_banner("CROSS-VALIDATION SETUP", char="=")
        print(f"  K-Folds: {self.k_folds}")
        print(f"  Total samples: {len(self.dataset):,}")
        print(f"  Classes: {len(self.class_names)}")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_mixed_precision}")
        print(f"  Random state: {self.random_state}")
        
        # Calculate and display class distribution
        label_counts = {}
        for label in self.dataset.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n  Overall class distribution:")
        for label, count in sorted(label_counts.items()):
            pct = safe_divide(count, len(self.dataset), 0) * 100
            if label < len(self.class_names):
                class_name = self.class_names[label]
                print(f"    {class_name}: {count:,} ({pct:.1f}%)")
        print("=" * 60)
    
    def run_cross_validation(self, 
                           save_fold_models: bool = False,
                           resume_from_fold: int = None) -> Dict:
        """
        Run enhanced k-fold cross-validation.
        
        Args:
            save_fold_models: Save model for each fold
            resume_from_fold: Resume from specific fold (for interrupted CV)
        
        Returns:
            Dictionary with CV results and statistics
        """
        # Create stratified k-fold splitter
        skf = StratifiedKFold(
            n_splits=self.k_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Get all labels for stratification
        all_labels = np.array([self.dataset.labels[i] for i in range(len(self.dataset))])
        
        # Check for checkpoint
        start_fold = 1
        if resume_from_fold:
            start_fold = resume_from_fold
            if self.verbose:
                print_banner("RESUMING CROSS-VALIDATION", char="!")
                print(f"  Starting from fold {start_fold}/{self.k_folds}")
                print("!" * 60)
        
        if self.verbose:
            print_banner(f"{self.k_folds}-FOLD CROSS-VALIDATION", char="=")
        
        # Track time
        cv_start_time = time.time()
        
        # Iterate through folds
        fold_iterator = enumerate(skf.split(np.zeros(len(all_labels)), all_labels), 1)
        for fold_idx, (train_idx, val_idx) in fold_iterator:
            # Skip if resuming
            if fold_idx < start_fold:
                continue
            
            fold_start_time = time.time()
            
            if self.verbose:
                print_banner(f"FOLD {fold_idx}/{self.k_folds}", char="─")
                print(f"  Train samples: {len(train_idx):,}")
                print(f"  Val samples: {len(val_idx):,}")
            
            # Create fold datasets
            train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
            val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=TRAINING_CONFIG['batch_size'],
                shuffle=True,
                num_workers=TRAINING_CONFIG['num_workers'],
                pin_memory=TRAINING_CONFIG.get('pin_memory', True)
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=TRAINING_CONFIG['batch_size'],
                shuffle=False,
                num_workers=TRAINING_CONFIG['num_workers'],
                pin_memory=TRAINING_CONFIG.get('pin_memory', True)
            )
            
            # Calculate class distribution for this fold
            train_labels = [all_labels[i] for i in train_idx]
            class_counts = [train_labels.count(i) for i in range(len(self.class_names))]
            
            if self.verbose:
                print(f"\n  Class distribution in fold {fold_idx}:")
                for i, (name, count) in enumerate(zip(self.class_names, class_counts)):
                    pct = safe_divide(count, len(train_idx), 0) * 100
                    print(f"    {name}: {count:,} ({pct:.1f}%)")
            
            # Train fold model
            fold_metrics = self._train_fold(
                fold_idx=fold_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                class_counts=class_counts,
                save_model=save_fold_models
            )
            
            # Store metrics
            self._store_fold_metrics(fold_idx, fold_metrics)
            
            # Save checkpoint after each fold
            self.checkpoint_manager.save_checkpoint(
                data={'fold_results': self.fold_results, 'cv_metrics': self.cv_metrics},
                last_index=fold_idx,
                metadata={'k_folds': self.k_folds}
            )
            
            # Print fold time
            fold_time = time.time() - fold_start_time
            if self.verbose:
                print(f"\n  Fold {fold_idx} completed in {format_time(fold_time)}")
                print("─" * 60)
        
        # Calculate statistics across folds
        cv_summary = self._calculate_cv_statistics()
        
        # Add total time
        cv_summary['total_time'] = time.time() - cv_start_time
        
        # Statistical significance testing
        cv_summary['significance'] = self._test_significance()
        
        # Print CV summary
        if self.verbose:
            self._print_cv_summary(cv_summary)
        
        return cv_summary
    
    def _train_fold(self, fold_idx: int, train_loader: DataLoader,
                   val_loader: DataLoader, class_counts: List[int],
                   save_model: bool) -> Dict:
        """Train a single fold and return metrics."""
        # Initialize model
        num_classes = len(self.class_names)
        model = self.model_class(num_classes=num_classes).to(self.device)
        
        # Setup training components
        criterion = get_weighted_criterion(
            class_counts,
            self.device,
            self.class_names,
            allow_zero=True
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # Learning rate scheduler
        num_training_steps = len(train_loader) * TRAINING_CONFIG['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=TRAINING_CONFIG.get('warmup_steps', 0),
            num_training_steps=num_training_steps
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            use_mixed_precision=self.use_mixed_precision
        )
        
        # Model save path
        model_save_path = None
        if save_model:
            model_save_path = os.path.join(
                models_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_fold{fold_idx}.pt"
            )
        
        # Train
        history = trainer.train(model_save_path=model_save_path)
        
        # Evaluate on validation set
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = safe_divide(total_loss, len(val_loader), 0)
        
        # Filter out error labels
        valid_mask = np.array(all_labels) != -1
        valid_preds = np.array(all_preds)[valid_mask]
        valid_labels = np.array(all_labels)[valid_mask]
        
        if len(valid_labels) > 0:
            report = classification_report(
                valid_labels,
                valid_preds,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            accuracy = accuracy_score(valid_labels, valid_preds)
            f1_macro = f1_score(valid_labels, valid_preds, average='macro', zero_division=0)
            f1_weighted = f1_score(valid_labels, valid_preds, average='weighted', zero_division=0)
            precision_macro = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
            recall_macro = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)
        else:
            # Handle edge case with no valid predictions
            report = {}
            accuracy = f1_macro = f1_weighted = precision_macro = recall_macro = 0.0
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'loss': avg_loss,
            'classification_report': report,
            'history': history
        }
    
    def _store_fold_metrics(self, fold_idx: int, metrics: Dict):
        """Store metrics from a fold."""
        # Store overall metrics
        self.cv_metrics['accuracy'].append(metrics['accuracy'])
        self.cv_metrics['f1_macro'].append(metrics['f1_macro'])
        self.cv_metrics['f1_weighted'].append(metrics['f1_weighted'])
        self.cv_metrics['precision_macro'].append(metrics['precision_macro'])
        self.cv_metrics['recall_macro'].append(metrics['recall_macro'])
        self.cv_metrics['loss'].append(metrics['loss'])
        
        # Store per-class metrics if report exists
        if metrics['classification_report']:
            for i, class_name in enumerate(self.class_names):
                if class_name in metrics['classification_report']:
                    self.per_class_metrics['precision'][i].append(
                        metrics['classification_report'][class_name]['precision']
                    )
                    self.per_class_metrics['recall'][i].append(
                        metrics['classification_report'][class_name]['recall']
                    )
                    self.per_class_metrics['f1'][i].append(
                        metrics['classification_report'][class_name]['f1-score']
                    )
        
        # Store complete fold result
        self.fold_results.append({
            'fold': fold_idx,
            **metrics
        })
        
        if self.verbose:
            print_banner(f"FOLD {fold_idx} RESULTS", char="─")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
            print(f"  Loss: {metrics['loss']:.4f}")
    
    def _calculate_cv_statistics(self) -> Dict:
        """Calculate mean, std, and confidence intervals across folds."""
        stats = {
            'overall': {},
            'per_class': {},
            'fold_results': self.fold_results
        }
        
        # Calculate overall metrics statistics
        for metric_name, values in self.cv_metrics.items():
            if values:  # Check if not empty
                values_array = np.array(values)
                
                # Calculate confidence interval
                confidence_level = HYPOTHESIS_TESTING_CONFIG.get('confidence_level', 0.95)
                degrees_freedom = len(values) - 1
                sample_mean = np.mean(values_array)
                sample_std = np.std(values_array, ddof=1)  # Use sample std
                
                # t-distribution for small samples
                t_score = scipy_stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
                margin_error = t_score * sample_std / np.sqrt(len(values))
                
                stats['overall'][metric_name] = {
                    'mean': sample_mean,
                    'std': sample_std,
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'confidence_interval': (sample_mean - margin_error, sample_mean + margin_error),
                    'values': values
                }
        
        # Calculate per-class statistics
        for metric_type in ['precision', 'recall', 'f1']:
            stats['per_class'][metric_type] = {}
            for i, class_name in enumerate(self.class_names):
                values = self.per_class_metrics[metric_type][i]
                if values:
                    values_array = np.array(values)
                    stats['per_class'][metric_type][class_name] = {
                        'mean': np.mean(values_array),
                        'std': np.std(values_array, ddof=1),
                        'min': np.min(values_array),
                        'max': np.max(values_array),
                        'values': values
                    }
        
        return stats
    
    def _test_significance(self) -> Dict:
        """Test statistical significance of results."""
        
        significance_results = {}
        
        # Use alpha from config - NO HARDCODING!
        alpha = HYPOTHESIS_TESTING_CONFIG.get('alpha', 0.05)
        
        # Test if performance is significantly different from random
        # For binary: random = 0.5, for 3-class: random = 0.33, etc.
        random_baseline = 1.0 / len(self.class_names)
        
        if self.cv_metrics['accuracy']:
            # One-sample t-test against random baseline
            t_stat, p_value = scipy_stats.ttest_1samp(
                self.cv_metrics['accuracy'],
                random_baseline
            )
            
            significance_results['vs_random'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'random_baseline': random_baseline,
                'alpha': alpha
            }
        
        # Test variance across folds (low variance = stable model)
        if self.cv_metrics['f1_macro']:
            cv_coefficient = np.std(self.cv_metrics['f1_macro']) / np.mean(self.cv_metrics['f1_macro'])
            significance_results['stability'] = {
                'cv_coefficient': cv_coefficient,
                'interpretation': 'Stable' if cv_coefficient < 0.1 else 'Variable'
            }
        
        return significance_results
    
    def _print_cv_summary(self, cv_summary: Dict):
        """Print enhanced cross-validation summary."""
        print_banner(f"{self.k_folds}-FOLD CV SUMMARY", char="=")
        
        # Overall metrics
        print("\nOverall Metrics (Mean ± Std [95% CI]):")
        print("─" * 60)
        for metric_name, stats in cv_summary['overall'].items():
            ci_lower, ci_upper = stats['confidence_interval']
            print(f"  {metric_name.replace('_', ' ').title():20s}: "
                  f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"[{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Per-class metrics
        print("\nPer-Class Metrics (Mean ± Std):")
        print("─" * 60)
        for class_name in self.class_names:
            print(f"  {class_name}:")
            for metric_type in ['precision', 'recall', 'f1']:
                if class_name in cv_summary['per_class'][metric_type]:
                    stats = cv_summary['per_class'][metric_type][class_name]
                    print(f"    {metric_type.title():10s}: "
                          f"{stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Statistical significance
        if 'significance' in cv_summary:
            print("\nStatistical Analysis:")
            print("─" * 60)
            
            if 'vs_random' in cv_summary['significance']:
                sig = cv_summary['significance']['vs_random']
                print(f"  Performance vs Random ({sig['random_baseline']:.2f}):")
                print(f"    p-value: {sig['p_value']:.4f}")
                print(f"    Result: {'✅ Significantly better' if sig['significant'] else '❌ Not significant'}")
            
            if 'stability' in cv_summary['significance']:
                stab = cv_summary['significance']['stability']
                print(f"  Model Stability:")
                print(f"    CV coefficient: {stab['cv_coefficient']:.3f}")
                print(f"    Assessment: {stab['interpretation']}")
        
        # Time
        if 'total_time' in cv_summary:
            print(f"\nTotal CV Time: {format_time(cv_summary['total_time'])}")
        
        print("=" * 60)
    
    def save_cv_results(self, output_path: str = None) -> str:
        """Save cross-validation results."""
        if output_path is None:
            output_path = os.path.join(
                results_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_cv_results.pkl"
            )
        
        ensure_dir_exists(os.path.dirname(output_path))
        
        cv_summary = self._calculate_cv_statistics()
        cv_summary['significance'] = self._test_significance()
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(cv_summary, f)
        
        # Also save as JSON for readability
        json_path = output_path.replace('.pkl', '.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_summary = self._make_json_serializable(cv_summary)
        
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ CV results saved to:")
            print(f"   Pickle: {output_path}")
            print(f"   JSON: {json_path}")
        
        return output_path
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
