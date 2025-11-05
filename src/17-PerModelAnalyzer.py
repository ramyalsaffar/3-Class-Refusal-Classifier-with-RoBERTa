# Per-Model Analysis Module
#--------------------------
# Analyze classifier performance per model.
# All imports are in 01-Imports.py
###############################################################################


class PerModelAnalyzer:
    """
    Analyze classifier performance per model.

    Generic analyzer that works for any classification model.
    """

    def __init__(self, model, tokenizer, device, class_names: List[str] = None):
        """
        Initialize per-model analyzer.

        Args:
            model: Classification model (RefusalClassifier or JailbreakDetector)
            tokenizer: RoBERTa tokenizer
            device: torch device
            class_names: List of class names (default: uses CLASS_NAMES from config)
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)

    def analyze(self, test_df: pd.DataFrame) -> Dict:
        """
        Analyze performance per model.

        Args:
            test_df: Test dataframe with 'model' column

        Returns:
            Dictionary with per-model results
        """
        # Filter out NaN model values before analysis
        models = test_df['model'].dropna().unique()

        if len(models) == 0:
            print(f"❌ ERROR: No valid model names found in test data")
            return {}

        results = {}

        print("\n" + "="*50)
        print("PER-MODEL ANALYSIS")
        print("="*50)

        for model_name in models:
            print(f"\nEvaluating on {model_name}...")
            model_df = test_df[test_df['model'] == model_name]

            # Create dataset and loader
            dataset = ClassificationDataset(
                model_df['response'].tolist(),
                model_df['label'].tolist(),
                self.tokenizer
            )
            loader = DataLoader(dataset, batch_size=API_CONFIG['inference_batch_size'], shuffle=False)

            # Evaluate
            preds, labels = self._evaluate(loader)

            # Calculate metrics
            # Explicitly specify labels to handle cases where not all classes are present
            label_indices = list(range(self.num_classes))
            f1_macro = f1_score(labels, preds, average='macro', labels=label_indices, zero_division=0)
            f1_per_class = f1_score(labels, preds, average=None, labels=label_indices, zero_division=0)
            report = classification_report(labels, preds, labels=label_indices, target_names=self.class_names, output_dict=True, zero_division=0)

            results[model_name] = {
                'f1_macro': float(f1_macro),
                'f1_per_class': {
                    self.class_names[i]: float(f1_per_class[i])
                    for i in range(len(self.class_names))
                },
                'accuracy': float(report['accuracy']),
                'num_samples': len(model_df)
            }

            print(f"  F1 Macro: {f1_macro:.3f}")
            print(f"  Accuracy: {report['accuracy']:.3f}")
            print(f"  Per-class F1: {[f'{f:.3f}' for f in f1_per_class]}")

        # Calculate variance
        f1_scores = [results[m]['f1_macro'] for m in models]
        results['analysis'] = {
            'f1_variance': float(pd.Series(f1_scores).var()),
            'f1_std': float(pd.Series(f1_scores).std()),
            'hardest_model': min(results.items(), key=lambda x: x[1]['f1_macro'] if x[0] != 'analysis' else float('inf'))[0],
            'easiest_model': max(results.items(), key=lambda x: x[1]['f1_macro'] if x[0] != 'analysis' else float('-inf'))[0]
        }

        print(f"\n{'='*50}")
        print(f"Cross-Model Variance: {results['analysis']['f1_variance']:.4f}")
        print(f"Hardest Model: {results['analysis']['hardest_model']}")
        print(f"Easiest Model: {results['analysis']['easiest_model']}")

        return results

    def _evaluate(self, loader):
        """Evaluate on dataloader."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                preds, _, _ = self.model.predict_with_confidence(input_ids, attention_mask)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_preds, all_labels

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved per-model results to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
