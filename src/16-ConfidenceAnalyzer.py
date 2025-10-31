# Confidence Analysis Module
#---------------------------
# Analyze prediction confidence scores.
# All imports are in 00-Imports.py
###############################################################################


class ConfidenceAnalyzer:
    """
    Analyze prediction confidence scores.

    Generic analyzer that works for any classification model.
    """

    def __init__(self, model, tokenizer, device, class_names: List[str] = None):
        """
        Initialize confidence analyzer.

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
        Analyze confidence distributions.

        Args:
            test_df: Test dataframe

        Returns:
            Dictionary with confidence metrics
        """
        print("\n" + "="*50)
        print("CONFIDENCE ANALYSIS")
        print("="*50)

        # Create dataset
        dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer
        )
        loader = DataLoader(dataset, batch_size=API_CONFIG['inference_batch_size'], shuffle=False)

        # Get predictions with confidence
        preds, labels, confidences, all_probs = self._evaluate(loader)

        # Calculate metrics
        correct = np.array(preds) == np.array(labels)

        results = {
            'overall': {
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'mean_confidence_correct': float(np.mean([c for c, cor in zip(confidences, correct) if cor])),
                'mean_confidence_incorrect': float(np.mean([c for c, cor in zip(confidences, correct) if not cor]))
            },
            'per_class': {}
        }

        # Per-class analysis (works for any number of classes)
        for class_idx in range(self.num_classes):
            class_mask = np.array(labels) == class_idx
            class_confidences = [c for c, m in zip(confidences, class_mask) if m]

            if len(class_confidences) > 0:
                results['per_class'][self.class_names[class_idx]] = {
                    'mean_confidence': float(np.mean(class_confidences)),
                    'std_confidence': float(np.std(class_confidences)),
                    'min_confidence': float(np.min(class_confidences)),
                    'max_confidence': float(np.max(class_confidences))
                }

        print(f"\nOverall Mean Confidence: {results['overall']['mean_confidence']:.3f}")
        print(f"Correct Predictions: {results['overall']['mean_confidence_correct']:.3f}")
        print(f"Incorrect Predictions: {results['overall']['mean_confidence_incorrect']:.3f}")

        print("\nPer-Class Mean Confidence:")
        for class_name, metrics in results['per_class'].items():
            print(f"  {class_name}: {metrics['mean_confidence']:.3f}")

        return results, preds, labels, confidences

    def _evaluate(self, loader):
        """Evaluate with confidence scores."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_confidences = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                preds, confidences, probs = self.model.predict_with_confidence(input_ids, attention_mask)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return all_preds, all_labels, all_confidences, all_probs

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved confidence metrics to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
