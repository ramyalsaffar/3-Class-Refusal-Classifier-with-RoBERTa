# Jailbreak Analysis Module
#--------------------------
# Specialized analysis for jailbreak detector with security-critical metrics.
# Wraps existing analyzers with jailbreak-specific interpretation and cross-analysis.
# All imports are in 00-Imports.py
###############################################################################


class JailbreakAnalysis:
    """
    Security-critical analysis for jailbreak detector.

    Key Differences from Refusal Analysis:
    - Extreme class imbalance (~95% Failed, ~5% Succeeded)
    - False Negatives are CATASTROPHIC (missed breaches)
    - Recall on "Succeeded" class is primary metric
    - Cross-analysis with refusal classifier
    """

    def __init__(self, jailbreak_model, refusal_model, tokenizer, device):
        """
        Initialize jailbreak analyzer.

        Args:
            jailbreak_model: Trained JailbreakDetector model
            refusal_model: Trained RefusalClassifier model (for cross-analysis)
            tokenizer: RoBERTa tokenizer
            device: torch device
        """
        self.jailbreak_model = jailbreak_model
        self.refusal_model = refusal_model
        self.tokenizer = tokenizer
        self.device = device

        # Reuse existing analyzers
        self.confidence_analyzer = ConfidenceAnalyzer(jailbreak_model, tokenizer, device)
        self.attention_viz = AttentionVisualizer(jailbreak_model, tokenizer, device)

    def analyze_full(self, test_df: pd.DataFrame) -> Dict:
        """
        Complete jailbreak detection analysis.

        Args:
            test_df: Test dataframe with columns:
                - 'response': LLM response text
                - 'jailbreak_label': Ground truth (0=Failed, 1=Succeeded)
                - 'refusal_label': Refusal classification (0/1/2)
                - 'model': Source model name
                - 'category': Prompt category

        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "="*60)
        print("JAILBREAK DETECTOR ANALYSIS")
        print("="*60)

        results = {}

        # 1. Security-Critical Metrics
        print("\n--- Security-Critical Metrics ---")
        results['security_metrics'] = self._calculate_security_metrics(test_df)
        self._print_security_metrics(results['security_metrics'])

        # 2. Identify False Negatives (CRITICAL!)
        print("\n--- False Negative Analysis ---")
        results['false_negatives'] = self._identify_false_negatives(test_df)
        print(f"Total False Negatives (missed jailbreaks): {len(results['false_negatives'])}")
        if len(results['false_negatives']) > 0:
            print("🚨 CRITICAL: These jailbreak successes were NOT detected!")
            print(f"   Review samples in: results/jailbreak_false_negatives.csv")

        # 3. Confidence Analysis
        print("\n--- Confidence Analysis ---")
        conf_results, preds, labels, confidences = self.confidence_analyzer.analyze(test_df)
        results['confidence'] = conf_results
        results['predictions'] = {'preds': preds, 'labels': labels, 'confidences': confidences}

        # 4. Per-Model Vulnerability
        print("\n--- Per-Model Vulnerability Analysis ---")
        results['vulnerability'] = self._analyze_vulnerability_per_model(test_df)
        self._print_vulnerability_analysis(results['vulnerability'])

        # 5. Attack Type Analysis
        print("\n--- Attack Type Success Rate ---")
        results['attack_types'] = self._analyze_attack_types(test_df)
        self._print_attack_analysis(results['attack_types'])

        # 6. Cross-Analysis with Refusal Classifier
        print("\n--- Cross-Analysis with Refusal Classifier ---")
        results['cross_analysis'] = self._cross_analyze_with_refusal(test_df)
        self._print_cross_analysis(results['cross_analysis'])

        # 7. Precision-Recall Curve (better than ROC for imbalanced data)
        print("\n--- Precision-Recall Analysis ---")
        results['pr_curve'] = self._calculate_pr_curve(preds, labels, confidences)

        # 8. Attention Analysis on Failures
        print("\n--- Attention Analysis on False Negatives ---")
        if len(results['false_negatives']) > 0:
            results['attention_fn'] = self._analyze_attention_on_failures(results['false_negatives'])

        print("\n" + "="*60)
        print("✅ JAILBREAK ANALYSIS COMPLETE")
        print("="*60)

        return results

    def _calculate_security_metrics(self, test_df: pd.DataFrame) -> Dict:
        """
        Calculate security-critical metrics.

        Primary Metrics:
        - Recall on "Succeeded" (class 1) - MOST IMPORTANT
        - False Negative Rate (FNR) - Must be minimized
        - True Negative Rate (TNR) - Correctly identifying safe responses
        - F1 Score (weighted for imbalance)
        """
        # Get predictions
        dataset = RefusalDataset(
            test_df['response'].tolist(),
            test_df['jailbreak_label'].tolist(),
            self.tokenizer
        )
        loader = DataLoader(dataset, batch_size=API_CONFIG['inference_batch_size'], shuffle=False)

        all_preds = []
        all_labels = []

        self.jailbreak_model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']

                logits = self.jailbreak_model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        cm = confusion_matrix(all_labels, all_preds)

        # For binary classification:
        # TN (True Negative): Predicted Failed, Actually Failed
        # FP (False Positive): Predicted Succeeded, Actually Failed
        # FN (False Negative): Predicted Failed, Actually Succeeded - CATASTROPHIC!
        # TP (True Positive): Predicted Succeeded, Actually Succeeded

        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]

        # Critical metrics
        recall_succeeded = recall_score(all_labels, all_preds, pos_label=1)  # Recall on class 1
        precision_succeeded = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)

        # Rates
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate

        # F1 Scores
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        # Overall accuracy (less important for imbalanced data)
        accuracy = accuracy_score(all_labels, all_preds)

        return {
            'confusion_matrix': cm,
            'recall_succeeded': recall_succeeded,  # PRIMARY METRIC
            'precision_succeeded': precision_succeeded,
            'false_negative_rate': fnr,  # CRITICAL - Must be low
            'true_negative_rate': tnr,
            'false_positive_rate': fpr,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': accuracy,
            'counts': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),  # These are breaches we missed!
                'true_positives': int(tp)
            }
        }

    def _print_security_metrics(self, metrics: Dict):
        """Print security metrics with alerts."""
        print(f"\nConfusion Matrix:")
        print(f"                  Predicted Failed  Predicted Succeeded")
        print(f"Actually Failed        {metrics['counts']['true_negatives']:6d}            {metrics['counts']['false_positives']:6d}")
        print(f"Actually Succeeded     {metrics['counts']['false_negatives']:6d}            {metrics['counts']['true_positives']:6d}")

        print(f"\n🎯 PRIMARY METRIC:")
        print(f"   Recall on Succeeded (class 1): {metrics['recall_succeeded']:.4f}")
        if metrics['recall_succeeded'] < 0.95:
            print(f"   🚨 WARNING: Recall below 95% target!")

        print(f"\n🚨 CRITICAL METRIC:")
        print(f"   False Negative Rate: {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']*100:.2f}%)")
        print(f"   Missed Jailbreaks: {metrics['counts']['false_negatives']}")
        if metrics['counts']['false_negatives'] > 0:
            print(f"   🚨 ALERT: {metrics['counts']['false_negatives']} jailbreak successes were NOT detected!")

        print(f"\n📊 Other Metrics:")
        print(f"   Precision on Succeeded: {metrics['precision_succeeded']:.4f}")
        print(f"   True Negative Rate (TNR): {metrics['true_negative_rate']:.4f}")
        print(f"   F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")

    def _identify_false_negatives(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify all false negatives (missed jailbreak successes).
        These are CRITICAL failures that must be manually reviewed.
        """
        # Get predictions
        dataset = RefusalDataset(
            test_df['response'].tolist(),
            test_df['jailbreak_label'].tolist(),
            self.tokenizer
        )
        loader = DataLoader(dataset, batch_size=API_CONFIG['inference_batch_size'], shuffle=False)

        all_preds = []
        all_confidences = []

        self.jailbreak_model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = self.jailbreak_model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                confidences = torch.max(probs, dim=1)[0].cpu().numpy()

                all_preds.extend(preds)
                all_confidences.extend(confidences)

        test_df = test_df.copy()
        test_df['jailbreak_pred'] = all_preds
        test_df['jailbreak_confidence'] = all_confidences

        # False Negatives: Actually Succeeded (1) but predicted Failed (0)
        false_negatives = test_df[
            (test_df['jailbreak_label'] == 1) & (test_df['jailbreak_pred'] == 0)
        ].copy()

        # Save for review
        if len(false_negatives) > 0:
            fn_path = os.path.join(results_path, "jailbreak_false_negatives.csv")
            false_negatives[['prompt', 'response', 'model', 'category', 'jailbreak_confidence']].to_csv(fn_path, index=False)

        return false_negatives

    def _analyze_vulnerability_per_model(self, test_df: pd.DataFrame) -> Dict:
        """
        Analyze which source models are most vulnerable to jailbreaks.
        """
        vulnerability = {}

        for model_name in test_df['model'].unique():
            model_df = test_df[test_df['model'] == model_name]

            # How many jailbreak attempts were there?
            total_samples = len(model_df)
            jailbreak_successes = (model_df['jailbreak_label'] == 1).sum()
            success_rate = jailbreak_successes / total_samples if total_samples > 0 else 0

            vulnerability[model_name] = {
                'total_samples': total_samples,
                'jailbreak_successes': int(jailbreak_successes),
                'success_rate': success_rate
            }

        return vulnerability

    def _print_vulnerability_analysis(self, vulnerability: Dict):
        """Print vulnerability analysis by model."""
        print("\nVulnerability by Source Model:")
        for model, stats in vulnerability.items():
            print(f"  {model}:")
            print(f"    Jailbreak Successes: {stats['jailbreak_successes']}/{stats['total_samples']} ({stats['success_rate']*100:.2f}%)")

    def _analyze_attack_types(self, test_df: pd.DataFrame) -> Dict:
        """
        Analyze success rate by attack type (based on category).
        """
        attack_analysis = {}

        # Map categories to attack types
        jailbreak_categories = ['jailbreaks']  # Could expand if we track attack types

        for category in test_df['category'].unique():
            cat_df = test_df[test_df['category'] == category]

            total = len(cat_df)
            successes = (cat_df['jailbreak_label'] == 1).sum()
            success_rate = successes / total if total > 0 else 0

            attack_analysis[category] = {
                'total_samples': total,
                'jailbreak_successes': int(successes),
                'success_rate': success_rate
            }

        return attack_analysis

    def _print_attack_analysis(self, attack_analysis: Dict):
        """Print attack type analysis."""
        print("\nJailbreak Success Rate by Category:")
        for category, stats in sorted(attack_analysis.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            if stats['jailbreak_successes'] > 0:
                print(f"  {category}:")
                print(f"    Successes: {stats['jailbreak_successes']}/{stats['total_samples']} ({stats['success_rate']*100:.2f}%)")

    def _cross_analyze_with_refusal(self, test_df: pd.DataFrame) -> Dict:
        """
        Cross-analyze jailbreak detection with refusal classification.

        Critical Question: Do jailbreak successes bypass refusal detection?
        """
        # Get predictions from both models
        jailbreak_preds = self._get_jailbreak_predictions(test_df)
        refusal_preds = self._get_refusal_predictions(test_df)

        # Cross-tabulation
        cross_tab = pd.crosstab(
            refusal_preds,
            jailbreak_preds,
            rownames=['Refusal'],
            colnames=['Jailbreak']
        )

        # Critical case: Jailbreak Succeeded + No Refusal
        dangerous = (refusal_preds == 0) & (jailbreak_preds == 1)
        dangerous_count = dangerous.sum()

        # Also check: Jailbreak Succeeded + Soft Refusal (partial bypass)
        partial_bypass = (refusal_preds == 2) & (jailbreak_preds == 1)
        partial_count = partial_bypass.sum()

        return {
            'cross_tab': cross_tab,
            'dangerous_combinations': dangerous_count,
            'partial_bypass': partial_count,
            'dangerous_samples': test_df[dangerous].copy() if dangerous_count > 0 else pd.DataFrame()
        }

    def _print_cross_analysis(self, cross_analysis: Dict):
        """Print cross-analysis results."""
        print("\nCross-Tabulation (Refusal vs Jailbreak):")
        print(cross_analysis['cross_tab'])

        if cross_analysis['dangerous_combinations'] > 0:
            print(f"\n🚨 CRITICAL: {cross_analysis['dangerous_combinations']} samples where:")
            print(f"   - Jailbreak SUCCEEDED")
            print(f"   - Refusal classifier says NO REFUSAL")
            print(f"   → Model completely bypassed!")

        if cross_analysis['partial_bypass'] > 0:
            print(f"\n⚠️  WARNING: {cross_analysis['partial_bypass']} samples where:")
            print(f"   - Jailbreak SUCCEEDED")
            print(f"   - Refusal classifier says SOFT REFUSAL")
            print(f"   → Partial bypass detected!")

    def _get_jailbreak_predictions(self, test_df: pd.DataFrame) -> np.ndarray:
        """Get jailbreak detector predictions."""
        dataset = RefusalDataset(
            test_df['response'].tolist(),
            test_df['jailbreak_label'].tolist(),
            self.tokenizer
        )
        loader = DataLoader(dataset, batch_size=API_CONFIG['inference_batch_size'], shuffle=False)

        all_preds = []
        self.jailbreak_model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = self.jailbreak_model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        return np.array(all_preds)

    def _get_refusal_predictions(self, test_df: pd.DataFrame) -> np.ndarray:
        """Get refusal classifier predictions."""
        dataset = RefusalDataset(
            test_df['response'].tolist(),
            test_df['refusal_label'].tolist(),
            self.tokenizer
        )
        loader = DataLoader(dataset, batch_size=API_CONFIG['inference_batch_size'], shuffle=False)

        all_preds = []
        self.refusal_model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = self.refusal_model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        return np.array(all_preds)

    def _calculate_pr_curve(self, preds, labels, confidences) -> Dict:
        """
        Calculate Precision-Recall curve.
        Better than ROC for highly imbalanced data.
        """
        # Use confidence as score
        precision, recall, thresholds = precision_recall_curve(labels, confidences)
        avg_precision = average_precision_score(labels, confidences)

        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': avg_precision
        }

    def _analyze_attention_on_failures(self, false_negatives: pd.DataFrame) -> Dict:
        """
        Analyze attention patterns on false negatives.
        Helps understand why we missed these jailbreaks.
        """
        if len(false_negatives) == 0:
            return {}

        # Analyze first 5 false negatives
        num_analyze = min(5, len(false_negatives))
        attention_results = []

        for idx in range(num_analyze):
            text = false_negatives.iloc[idx]['response']
            attention_data = self.attention_viz.get_attention_weights(text)
            attention_results.append(attention_data)

        return {'analyzed_samples': num_analyze, 'attention_data': attention_results}

    def save_results(self, results: Dict, output_path: str):
        """Save analysis results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'false_negatives':
                serializable_results[key] = f"{len(value)} samples (see jailbreak_false_negatives.csv)"
            elif key == 'predictions':
                serializable_results[key] = {
                    'num_samples': len(value['preds'])
                }
            elif key == 'cross_analysis':
                serializable_results[key] = {
                    'dangerous_combinations': int(value['dangerous_combinations']),
                    'partial_bypass': int(value['partial_bypass'])
                }
            elif isinstance(value, dict):
                serializable_results[key] = value

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"✓ Saved jailbreak analysis to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 30, 2025
@author: ramyalsaffar
"""
