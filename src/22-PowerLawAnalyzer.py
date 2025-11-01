# Power Law Analyzer Module
#--------------------------
# Analyzes power law distributions in classifier performance:
# 1. Error Concentration (Pareto): Do 20% of categories cause 80% of errors?
# 2. Confidence Distribution: Do confidence scores follow power law?
# 3. Attention Distribution: Do few tokens dominate attention?
# All imports are in 01-Imports.py
###############################################################################


class PowerLawAnalyzer:
    """
    Analyzes power law distributions in classifier behavior.

    Power Laws in ML:
    - Error Concentration: Few categories/models cause most errors (Pareto Principle)
    - Confidence Distribution: Scores often follow power law
    - Attention Weights: Few tokens receive most attention (Zipfian)

    These analyses help identify:
    - Where to focus improvement efforts (concentrated errors)
    - Model calibration issues (confidence power laws)
    - Feature importance patterns (attention distribution)
    """

    def __init__(self, model, tokenizer, device, class_names: List[str] = None):
        """
        Initialize power law analyzer.

        Args:
            model: Classification model
            tokenizer: RoBERTa tokenizer
            device: torch device
            class_names: List of class names
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)

    def analyze_all(self, test_df: pd.DataFrame, predictions: np.ndarray,
                    confidences: np.ndarray, output_dir: str = None) -> Dict:
        """
        Run all power law analyses.

        Args:
            test_df: Test dataframe with 'category', 'model', 'response', 'label' columns
            predictions: Model predictions
            confidences: Prediction confidences
            output_dir: Directory to save visualizations (default: visualizations_path)

        Returns:
            Dictionary with all power law analysis results
        """
        if output_dir is None:
            output_dir = visualizations_path

        print("\n" + "="*60)
        print("POWER LAW ANALYSIS")
        print("="*60)

        results = {}

        # 1. Error Concentration Analysis (Pareto)
        print("\n--- Error Concentration Analysis (Pareto Principle) ---")
        results['error_concentration'] = self._analyze_error_concentration(
            test_df, predictions, output_dir
        )
        self._print_pareto_results(results['error_concentration'])

        # 2. Confidence Distribution Analysis
        print("\n--- Confidence Distribution (Power Law Check) ---")
        results['confidence_distribution'] = self._analyze_confidence_distribution(
            confidences, predictions, test_df['label'].values, output_dir
        )
        self._print_confidence_analysis(results['confidence_distribution'])

        # 3. Attention Power Law Analysis
        print("\n--- Attention Distribution (Token Importance) ---")
        results['attention_power_law'] = self._analyze_attention_power_law(
            test_df, output_dir
        )
        self._print_attention_analysis(results['attention_power_law'])

        print("\n" + "="*60)
        print("✅ POWER LAW ANALYSIS COMPLETE")
        print("="*60)

        return results

    def _analyze_error_concentration(self, test_df: pd.DataFrame,
                                    predictions: np.ndarray,
                                    output_dir: str) -> Dict:
        """
        Analyze error concentration (Pareto Principle).

        Question: Do 20% of categories/models cause 80% of errors?

        Returns:
            Dictionary with Pareto analysis results
        """
        labels = test_df['label'].values
        errors = predictions != labels

        results = {
            'by_category': self._pareto_analysis_by_group(
                test_df, errors, 'category', 'Category'
            ),
            'by_model': self._pareto_analysis_by_group(
                test_df, errors, 'model', 'Model'
            ) if 'model' in test_df.columns else None,
            'by_class': self._pareto_analysis_by_class(
                labels, predictions
            )
        }

        # Create Pareto visualizations
        self._plot_pareto_chart(
            results['by_category'],
            os.path.join(output_dir, "pareto_errors_by_category.png"),
            "Error Concentration by Category (Pareto Analysis)"
        )

        if results['by_model'] is not None:
            self._plot_pareto_chart(
                results['by_model'],
                os.path.join(output_dir, "pareto_errors_by_model.png"),
                "Error Concentration by Model (Pareto Analysis)"
            )

        return results

    def _pareto_analysis_by_group(self, test_df: pd.DataFrame, errors: np.ndarray,
                                  group_col: str, group_name: str) -> Dict:
        """
        Perform Pareto analysis by grouping column.

        Returns:
            Dict with error counts, cumulative percentages, Pareto metrics
        """
        if group_col not in test_df.columns:
            return None

        # Count errors per group
        df_with_errors = test_df.copy()
        df_with_errors['error'] = errors

        error_counts = df_with_errors.groupby(group_col)['error'].agg(['sum', 'count'])
        error_counts['error_rate'] = error_counts['sum'] / error_counts['count']
        error_counts = error_counts.sort_values('sum', ascending=False)

        # Calculate cumulative percentages
        total_errors = error_counts['sum'].sum()
        error_counts['cumulative_errors'] = error_counts['sum'].cumsum()
        # FIX: Protect against division by zero when model has perfect performance
        if total_errors > 0:
            error_counts['cumulative_pct'] = (error_counts['cumulative_errors'] / total_errors * 100)
        else:
            error_counts['cumulative_pct'] = 0

        # Find what % of groups cause 80% of errors
        groups_80pct = (error_counts['cumulative_pct'] <= 80).sum()
        total_groups = len(error_counts)
        groups_80pct_ratio = groups_80pct / total_groups * 100 if total_groups > 0 else 0

        return {
            'group_name': group_name,
            'error_counts': error_counts,
            'total_groups': total_groups,
            'groups_causing_80pct_errors': groups_80pct,
            'groups_80pct_ratio': groups_80pct_ratio,
            'pareto_holds': groups_80pct_ratio <= 30  # Strict Pareto is ~20%, allow up to 30%
        }

    def _pareto_analysis_by_class(self, labels: np.ndarray,
                                  predictions: np.ndarray) -> Dict:
        """
        Analyze which classes contribute most to errors.

        Returns:
            Dict with per-class error analysis
        """
        class_errors = {}
        total_errors = 0

        for class_idx in range(self.num_classes):
            class_mask = labels == class_idx
            if class_mask.sum() == 0:
                continue

            class_error_count = ((predictions != labels) & class_mask).sum()
            total_errors += class_error_count

            class_errors[self.class_names[class_idx]] = {
                'error_count': int(class_error_count),
                'total_samples': int(class_mask.sum()),
                'error_rate': float(class_error_count / class_mask.sum())
            }

        # Sort by error count
        sorted_classes = sorted(
            class_errors.items(),
            key=lambda x: x[1]['error_count'],
            reverse=True
        )

        # Calculate cumulative
        cumulative_errors = 0
        for class_name, stats in sorted_classes:
            cumulative_errors += stats['error_count']
            stats['cumulative_pct'] = cumulative_errors / total_errors * 100 if total_errors > 0 else 0

        return {
            'class_errors': dict(sorted_classes),
            'total_errors': int(total_errors)
        }

    def _analyze_confidence_distribution(self, confidences: np.ndarray,
                                        predictions: np.ndarray,
                                        labels: np.ndarray,
                                        output_dir: str) -> Dict:
        """
        Analyze confidence score distribution for power law.

        Power law: P(x) ∝ x^(-α)
        On log-log plot, this appears as straight line with slope -α

        Also checks calibration: are high-confidence predictions actually correct?

        Returns:
            Dict with power law fit, calibration metrics
        """
        errors = predictions != labels
        correct_confidences = confidences[~errors]
        error_confidences = confidences[errors]

        # Bin confidences and count
        bins = np.linspace(0, 1, 21)  # 20 bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        hist_all, _ = np.histogram(confidences, bins=bins)
        hist_correct, _ = np.histogram(correct_confidences, bins=bins)
        hist_errors, _ = np.histogram(error_confidences, bins=bins)

        # Fit power law to high-confidence region (0.5-1.0)
        high_conf_mask = bin_centers >= 0.5
        x_fit = bin_centers[high_conf_mask]
        y_fit = hist_all[high_conf_mask]

        # Log-log fit (avoid log(0))
        valid_mask = y_fit > 0
        if valid_mask.sum() > 2:
            log_x = np.log(x_fit[valid_mask])
            log_y = np.log(y_fit[valid_mask])
            slope, intercept = np.polyfit(log_x, log_y, 1)
            power_law_exponent = -slope
        else:
            power_law_exponent = None
            slope = None
            intercept = None

        # Calibration analysis: confidence bins vs accuracy
        calibration = []
        for i in range(len(bins) - 1):
            bin_mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if bin_mask.sum() > 0:
                bin_accuracy = (predictions[bin_mask] == labels[bin_mask]).mean()
                bin_conf = confidences[bin_mask].mean()
                calibration.append({
                    'confidence_range': f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                    'mean_confidence': float(bin_conf),
                    'accuracy': float(bin_accuracy),
                    'count': int(bin_mask.sum()),
                    'calibration_error': float(abs(bin_conf - bin_accuracy))
                })

        # Expected Calibration Error (ECE)
        ece = np.mean([c['calibration_error'] for c in calibration])

        # Plot confidence distribution
        self._plot_confidence_distribution(
            confidences, correct_confidences, error_confidences,
            os.path.join(output_dir, "confidence_distribution.png")
        )

        # Plot calibration curve
        self._plot_calibration_curve(
            calibration,
            os.path.join(output_dir, "confidence_calibration.png")
        )

        return {
            'power_law_exponent': power_law_exponent,
            'power_law_slope': slope,
            'power_law_intercept': intercept,
            'expected_calibration_error': float(ece),
            'calibration_bins': calibration,
            'stats': {
                'mean_confidence_all': float(confidences.mean()),
                'mean_confidence_correct': float(correct_confidences.mean()),
                'mean_confidence_errors': float(error_confidences.mean()),
                'overconfidence_on_errors': float(error_confidences.mean() > 0.5)
            }
        }

    def _analyze_attention_power_law(self, test_df: pd.DataFrame,
                                    output_dir: str) -> Dict:
        """
        Analyze attention weight distribution for power law.

        Question: Do few tokens receive most attention? (Zipfian distribution)

        Samples a subset of test data to analyze attention patterns.

        Returns:
            Dict with attention power law metrics
        """
        # Sample texts for analysis (attention is expensive)
        sample_size = min(ANALYSIS_CONFIG.get('attention_sample_size', 100), len(test_df))
        sample_df = test_df.sample(n=sample_size, random_state=42)

        all_token_attentions = []
        top_k_concentrations = []

        self.model.eval()
        with torch.no_grad():
            for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analyzing attention"):
                text = row['response']

                # Tokenize
                encoding = self.tokenizer(
                    text,
                    max_length=MODEL_CONFIG['max_length'],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                # Get attention weights from model
                if hasattr(self.model, 'roberta'):
                    outputs = self.model.roberta(
                        input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
                    attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)

                    # Average across layers and heads, get attention to CLS token
                    # Shape: (layers, batch, heads, seq, seq) -> (seq,)
                    avg_attention = torch.stack([a[0].mean(0) for a in attentions]).mean(0)
                    cls_attention = avg_attention[0, :]  # Attention from CLS to all tokens

                    # Get valid tokens only
                    valid_mask = attention_mask[0].cpu().numpy().astype(bool)
                    valid_attention = cls_attention.cpu().numpy()[valid_mask]

                    all_token_attentions.extend(valid_attention.tolist())

                    # Top-k concentration: what % of attention goes to top 20% tokens?
                    k = max(1, len(valid_attention) // 5)  # Top 20%
                    top_k_attn = np.partition(valid_attention, -k)[-k:].sum()
                    top_k_concentrations.append(float(top_k_attn))

        if len(all_token_attentions) == 0:
            return {'error': 'No attention weights extracted'}

        # Analyze distribution
        all_token_attentions = np.array(all_token_attentions)

        # Fit power law (Zipf: frequency ∝ rank^(-s))
        # Sort attentions in descending order
        sorted_attentions = np.sort(all_token_attentions)[::-1]
        ranks = np.arange(1, len(sorted_attentions) + 1)

        # Log-log fit
        log_ranks = np.log(ranks[sorted_attentions > 0])
        log_attentions = np.log(sorted_attentions[sorted_attentions > 0])

        if len(log_ranks) > 10:
            slope, intercept = np.polyfit(log_ranks[:1000], log_attentions[:1000], 1)  # Fit first 1000
            zipf_exponent = -slope
        else:
            zipf_exponent = None
            slope = None

        # Plot attention distribution
        self._plot_attention_distribution(
            sorted_attentions, ranks,
            os.path.join(output_dir, "attention_power_law.png")
        )

        return {
            'zipf_exponent': zipf_exponent,
            'zipf_slope': slope,
            'mean_top20_concentration': float(np.mean(top_k_concentrations)),
            'attention_stats': {
                'mean_attention': float(all_token_attentions.mean()),
                'std_attention': float(all_token_attentions.std()),
                'max_attention': float(all_token_attentions.max()),
                'min_attention': float(all_token_attentions.min()),
                'total_tokens_analyzed': len(all_token_attentions)
            }
        }

    def _plot_pareto_chart(self, pareto_data: Dict, output_path: str, title: str):
        """Plot Pareto chart (bar + cumulative line)."""
        if pareto_data is None:
            return

        error_counts = pareto_data['error_counts']

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar chart for error counts
        x = np.arange(len(error_counts))
        ax1.bar(x, error_counts['sum'], color='steelblue', alpha=0.7)
        ax1.set_xlabel(pareto_data['group_name'])
        ax1.set_ylabel('Error Count', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(error_counts.index, rotation=45, ha='right')

        # Cumulative percentage line
        ax2 = ax1.twinx()
        ax2.plot(x, error_counts['cumulative_pct'], color='red', marker='o', linewidth=2)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.set_ylabel('Cumulative % of Errors', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 105])
        ax2.legend()

        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved Pareto chart: {output_path}")

    def _plot_confidence_distribution(self, all_conf, correct_conf, error_conf, output_path: str):
        """Plot confidence distribution for correct vs incorrect predictions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green', density=True)
        axes[0].hist(error_conf, bins=20, alpha=0.7, label='Errors', color='red', density=True)
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Confidence Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Log-log plot (power law check)
        bins = np.linspace(0, 1, 21)
        hist_all, _ = np.histogram(all_conf, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        valid_mask = (hist_all > 0) & (bin_centers > 0)
        axes[1].loglog(bin_centers[valid_mask], hist_all[valid_mask], 'o-', label='All predictions')
        axes[1].set_xlabel('Confidence (log scale)')
        axes[1].set_ylabel('Frequency (log scale)')
        axes[1].set_title('Power Law Check (Log-Log Plot)')
        axes[1].legend()
        axes[1].grid(alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved confidence distribution: {output_path}")

    def _plot_calibration_curve(self, calibration: List[Dict], output_path: str):
        """Plot calibration curve (confidence vs accuracy)."""
        confidences = [c['mean_confidence'] for c in calibration]
        accuracies = [c['accuracy'] for c in calibration]

        plt.figure(figsize=(8, 8))
        plt.plot(confidences, accuracies, 'o-', linewidth=2, markersize=8, label='Model')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        plt.xlabel('Mean Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved calibration curve: {output_path}")

    def _plot_attention_distribution(self, sorted_attentions: np.ndarray,
                                    ranks: np.ndarray, output_path: str):
        """Plot attention distribution on log-log scale (Zipf's law check)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Regular plot
        axes[0].plot(ranks[:500], sorted_attentions[:500], 'o-', alpha=0.6)
        axes[0].set_xlabel('Token Rank')
        axes[0].set_ylabel('Attention Weight')
        axes[0].set_title('Attention Distribution (Top 500 Tokens)')
        axes[0].grid(alpha=0.3)

        # Log-log plot (Zipf's law)
        valid_mask = sorted_attentions > 0
        axes[1].loglog(ranks[valid_mask][:1000], sorted_attentions[valid_mask][:1000], 'o', alpha=0.5)
        axes[1].set_xlabel('Token Rank (log scale)')
        axes[1].set_ylabel('Attention Weight (log scale)')
        axes[1].set_title('Zipf\'s Law Check (Log-Log Plot)')
        axes[1].grid(alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved attention distribution: {output_path}")

    def _print_pareto_results(self, results: Dict):
        """Print Pareto analysis results."""
        print("\n📊 Pareto Principle Analysis:")

        # By category
        if results['by_category']:
            cat = results['by_category']
            print(f"\n   By Category:")
            print(f"   - Total categories: {cat['total_groups']}")
            print(f"   - Categories causing 80% of errors: {cat['groups_causing_80pct_errors']} ({cat['groups_80pct_ratio']:.1f}%)")
            if cat['pareto_holds']:
                print(f"   ✅ Pareto Principle HOLDS: {cat['groups_80pct_ratio']:.1f}% of categories cause 80% of errors")
            else:
                print(f"   ⚠️  Pareto Principle does NOT hold: {cat['groups_80pct_ratio']:.1f}% > 30%")

        # By model
        if results['by_model']:
            model = results['by_model']
            print(f"\n   By Model:")
            print(f"   - Total models: {model['total_groups']}")
            print(f"   - Models causing 80% of errors: {model['groups_causing_80pct_errors']} ({model['groups_80pct_ratio']:.1f}%)")

        # By class
        if results['by_class']:
            print(f"\n   By Class (top 3 error producers):")
            for i, (class_name, stats) in enumerate(list(results['by_class']['class_errors'].items())[:3]):
                print(f"   {i+1}. {class_name}: {stats['error_count']} errors ({stats['cumulative_pct']:.1f}% cumulative)")

    def _print_confidence_analysis(self, results: Dict):
        """Print confidence distribution analysis."""
        print("\n📈 Confidence Distribution Analysis:")

        if results['power_law_exponent'] is not None:
            print(f"   Power law exponent (α): {results['power_law_exponent']:.3f}")
            if 1.0 <= results['power_law_exponent'] <= 3.0:
                print(f"   ✅ Confidence follows power law distribution")
            else:
                print(f"   ⚠️  Unusual power law exponent")

        print(f"\n   Calibration:")
        print(f"   - Expected Calibration Error (ECE): {results['expected_calibration_error']:.4f}")
        if results['expected_calibration_error'] < 0.05:
            print(f"   ✅ Well calibrated (ECE < 0.05)")
        elif results['expected_calibration_error'] < 0.10:
            print(f"   ⚠️  Moderate calibration (ECE < 0.10)")
        else:
            print(f"   🚨 Poor calibration (ECE ≥ 0.10)")

        stats = results['stats']
        print(f"\n   Confidence Statistics:")
        print(f"   - Mean confidence (all): {stats['mean_confidence_all']:.3f}")
        print(f"   - Mean confidence (correct): {stats['mean_confidence_correct']:.3f}")
        print(f"   - Mean confidence (errors): {stats['mean_confidence_errors']:.3f}")
        if stats['mean_confidence_errors'] > 0.7:
            print(f"   🚨 WARNING: High confidence on errors (overconfidence)!")

    def _print_attention_analysis(self, results: Dict):
        """Print attention power law analysis."""
        if 'error' in results:
            print(f"   ⚠️  {results['error']}")
            return

        print("\n🔍 Attention Distribution Analysis:")

        if results['zipf_exponent'] is not None:
            print(f"   Zipf exponent (s): {results['zipf_exponent']:.3f}")
            if 0.8 <= results['zipf_exponent'] <= 1.5:
                print(f"   ✅ Attention follows Zipfian distribution (typical for language)")
            else:
                print(f"   ⚠️  Unusual Zipf exponent")

        print(f"\n   Attention Concentration:")
        print(f"   - Top 20% tokens receive: {results['mean_top20_concentration']*100:.1f}% of attention")
        if results['mean_top20_concentration'] > 0.6:
            print(f"   ✅ Attention appropriately concentrated on key tokens")
        else:
            print(f"   ⚠️  Attention may be too diffuse")

        stats = results['attention_stats']
        print(f"\n   Attention Statistics:")
        print(f"   - Total tokens analyzed: {stats['total_tokens_analyzed']}")
        print(f"   - Mean attention: {stats['mean_attention']:.6f}")
        print(f"   - Max attention: {stats['max_attention']:.6f}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 31, 2025
@author: ramyalsaffar
"""
