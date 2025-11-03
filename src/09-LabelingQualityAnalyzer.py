# Labeling Quality Analyzer Module
#-----------------------------------
# Analyzes judge labeling quality and confidence scores.
# Helps identify low-quality labels and systematic issues.
# All imports are in 01-Imports.py
###############################################################################


class LabelingQualityAnalyzer:
    """Analyze judge labeling quality using confidence scores."""

    def __init__(self, verbose: bool = True):
        """
        Initialize analyzer.

        Args:
            verbose: Print detailed output
        """
        self.verbose = verbose
        self.class_names = CLASS_NAMES

    def analyze_full(self, labeled_df: pd.DataFrame) -> Dict:
        """
        Run complete labeling quality analysis.

        Args:
            labeled_df: DataFrame with columns:
                - refusal_label, refusal_confidence
                - jailbreak_label, jailbreak_confidence
                - model (LLM that generated response)
                - category (prompt category)

        Returns:
            Dictionary with all analysis results
        """
        if self.verbose:
            print("\n" + "="*60)
            print("LABELING QUALITY ANALYSIS")
            print("="*60)

        results = {}

        # Overall confidence statistics
        results['overall'] = self._analyze_overall_confidence(labeled_df)

        # Per-model analysis
        if 'model' in labeled_df.columns:
            results['per_model'] = self._analyze_per_model_confidence(labeled_df)

        # Per-category analysis
        if 'category' in labeled_df.columns:
            results['per_category'] = self._analyze_per_category_confidence(labeled_df)

        # Low confidence flags
        results['low_confidence'] = self._flag_low_confidence(labeled_df)

        # Task agreement analysis
        results['task_agreement'] = self._analyze_task_agreement(labeled_df)

        return results

    def _analyze_overall_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze overall confidence statistics."""
        if self.verbose:
            print("\n--- Overall Confidence Statistics ---")

        # Filter out error labels
        valid_refusal = df[df['refusal_label'] != -1]
        valid_jailbreak = df[df['jailbreak_label'] != -1]

        threshold = LABELING_CONFIG['low_confidence_threshold']

        results = {
            'refusal': {
                'mean': float(valid_refusal['refusal_confidence'].mean()),
                'std': float(valid_refusal['refusal_confidence'].std()),
                'median': float(valid_refusal['refusal_confidence'].median()),
                'min': float(valid_refusal['refusal_confidence'].min()),
                'max': float(valid_refusal['refusal_confidence'].max()),
                'count_low': int((valid_refusal['refusal_confidence'] < threshold).sum()),
                'percent_low': float((valid_refusal['refusal_confidence'] < threshold).sum() / len(valid_refusal) * 100)
            },
            'jailbreak': {
                'mean': float(valid_jailbreak['jailbreak_confidence'].mean()),
                'std': float(valid_jailbreak['jailbreak_confidence'].std()),
                'median': float(valid_jailbreak['jailbreak_confidence'].median()),
                'min': float(valid_jailbreak['jailbreak_confidence'].min()),
                'max': float(valid_jailbreak['jailbreak_confidence'].max()),
                'count_low': int((valid_jailbreak['jailbreak_confidence'] < threshold).sum()),
                'percent_low': float((valid_jailbreak['jailbreak_confidence'] < threshold).sum() / len(valid_jailbreak) * 100)
            }
        }

        if self.verbose:
            print(f"  Refusal - Mean: {results['refusal']['mean']:.1f}%, "
                  f"Median: {results['refusal']['median']:.1f}%, "
                  f"Low (<{threshold}%): {results['refusal']['count_low']} ({results['refusal']['percent_low']:.1f}%)")
            print(f"  Jailbreak - Mean: {results['jailbreak']['mean']:.1f}%, "
                  f"Median: {results['jailbreak']['median']:.1f}%, "
                  f"Low (<{threshold}%): {results['jailbreak']['count_low']} ({results['jailbreak']['percent_low']:.1f}%)")

        return results

    def _analyze_per_model_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze confidence by LLM model."""
        if self.verbose:
            print("\n--- Per-Model Confidence ---")

        threshold = LABELING_CONFIG['low_confidence_threshold']
        results = {}
        valid_df = df[df['refusal_label'] != -1]

        for model in valid_df['model'].unique():
            model_df = valid_df[valid_df['model'] == model]

            results[model] = {
                'refusal': {
                    'mean': float(model_df['refusal_confidence'].mean()),
                    'count_low': int((model_df['refusal_confidence'] < threshold).sum()),
                    'percent_low': float((model_df['refusal_confidence'] < threshold).sum() / len(model_df) * 100)
                },
                'jailbreak': {
                    'mean': float(model_df['jailbreak_confidence'].mean()),
                    'count_low': int((model_df['jailbreak_confidence'] < threshold).sum()),
                    'percent_low': float((model_df['jailbreak_confidence'] < threshold).sum() / len(model_df) * 100)
                }
            }

            if self.verbose:
                print(f"\n  {model}:")
                print(f"    Refusal - Mean: {results[model]['refusal']['mean']:.1f}%, "
                      f"Low: {results[model]['refusal']['count_low']} ({results[model]['refusal']['percent_low']:.1f}%)")
                print(f"    Jailbreak - Mean: {results[model]['jailbreak']['mean']:.1f}%, "
                      f"Low: {results[model]['jailbreak']['count_low']} ({results[model]['jailbreak']['percent_low']:.1f}%)")

        return results

    def _analyze_per_category_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze confidence by prompt category."""
        if self.verbose:
            print("\n--- Per-Category Confidence ---")

        threshold = LABELING_CONFIG['low_confidence_threshold']
        results = {}
        valid_df = df[df['refusal_label'] != -1]

        for category in valid_df['category'].unique():
            cat_df = valid_df[valid_df['category'] == category]

            results[category] = {
                'refusal': {
                    'mean': float(cat_df['refusal_confidence'].mean()),
                    'count_low': int((cat_df['refusal_confidence'] < threshold).sum()),
                    'percent_low': float((cat_df['refusal_confidence'] < threshold).sum() / len(cat_df) * 100)
                },
                'jailbreak': {
                    'mean': float(cat_df['jailbreak_confidence'].mean()),
                    'count_low': int((cat_df['jailbreak_confidence'] < threshold).sum()),
                    'percent_low': float((cat_df['jailbreak_confidence'] < threshold).sum() / len(cat_df) * 100)
                }
            }

        # Print top 5 categories with lowest confidence
        if self.verbose:
            sorted_cats = sorted(results.items(),
                               key=lambda x: x[1]['refusal']['mean'])[:5]
            print("\n  Top 5 categories with lowest refusal confidence:")
            for cat, metrics in sorted_cats:
                print(f"    {cat}: {metrics['refusal']['mean']:.1f}%")

        return results

    def _flag_low_confidence(self, df: pd.DataFrame, threshold: int = None) -> Dict:
        """Flag samples with low confidence for review."""
        threshold = threshold or LABELING_CONFIG['low_confidence_threshold']
        if self.verbose:
            print(f"\n--- Low Confidence Flags (threshold: {threshold}%) ---")

        valid_df = df[df['refusal_label'] != -1]

        # Find low confidence samples
        low_refusal = valid_df[valid_df['refusal_confidence'] < threshold]
        low_jailbreak = valid_df[valid_df['jailbreak_confidence'] < threshold]
        low_both = valid_df[(valid_df['refusal_confidence'] < threshold) &
                           (valid_df['jailbreak_confidence'] < threshold)]

        results = {
            'threshold': threshold,
            'low_refusal_count': len(low_refusal),
            'low_jailbreak_count': len(low_jailbreak),
            'low_both_count': len(low_both),
            'low_refusal_percent': float(len(low_refusal) / len(valid_df) * 100),
            'low_jailbreak_percent': float(len(low_jailbreak) / len(valid_df) * 100),
            'low_both_percent': float(len(low_both) / len(valid_df) * 100)
        }

        if self.verbose:
            print(f"  Low refusal confidence: {results['low_refusal_count']} ({results['low_refusal_percent']:.1f}%)")
            print(f"  Low jailbreak confidence: {results['low_jailbreak_count']} ({results['low_jailbreak_percent']:.1f}%)")
            print(f"  Low confidence in BOTH: {results['low_both_count']} ({results['low_both_percent']:.1f}%)")

        # Save flagged samples for review
        results['flagged_samples'] = []
        for idx, row in low_both.iterrows():
            results['flagged_samples'].append({
                'index': int(idx),
                'model': row.get('model', 'unknown'),
                'category': row.get('category', 'unknown'),
                'refusal_label': int(row['refusal_label']),
                'refusal_confidence': float(row['refusal_confidence']),
                'jailbreak_label': int(row['jailbreak_label']),
                'jailbreak_confidence': float(row['jailbreak_confidence']),
                'prompt_preview': row['prompt'][:100] if 'prompt' in row else None,
                'response_preview': row['response'][:100] if 'response' in row else None
            })

        if self.verbose and len(results['flagged_samples']) > 0:
            print(f"\n  Flagged {len(results['flagged_samples'])} samples for manual review")

        return results

    def _analyze_task_agreement(self, df: pd.DataFrame) -> Dict:
        """Analyze agreement between refusal and jailbreak labeling."""
        if self.verbose:
            print("\n--- Task Agreement Analysis ---")

        valid_df = df[(df['refusal_label'] != -1) & (df['jailbreak_label'] != -1)]

        # Cross-tabulation
        results = {
            'correlation': float(valid_df['refusal_confidence'].corr(valid_df['jailbreak_confidence'])),
            'refusal_no_jb_success': len(valid_df[(valid_df['refusal_label'] == 0) &
                                                   (valid_df['jailbreak_label'] == 1)]),
            'refusal_yes_jb_failed': len(valid_df[(valid_df['refusal_label'].isin([1, 2])) &
                                                   (valid_df['jailbreak_label'] == 0)])
        }

        if self.verbose:
            print(f"  Confidence correlation: {results['correlation']:.3f}")
            print(f"  No refusal but jailbreak succeeded: {results['refusal_no_jb_success']} "
                  f"(unusual - should investigate)")
            print(f"  Refusal but jailbreak failed: {results['refusal_yes_jb_failed']}")

        return results

    def export_flagged_samples(self, labeled_df: pd.DataFrame, output_path: str,
                              threshold: int = None):
        """
        Export low-confidence samples for manual review.

        Args:
            labeled_df: Labeled DataFrame
            output_path: Path to save CSV
            threshold: Confidence threshold (uses config if None)
        """
        threshold = threshold or LABELING_CONFIG['low_confidence_threshold']
        valid_df = labeled_df[labeled_df['refusal_label'] != -1]
        low_both = valid_df[(valid_df['refusal_confidence'] < threshold) &
                           (valid_df['jailbreak_confidence'] < threshold)]

        # Select relevant columns
        export_cols = ['prompt', 'response', 'model', 'category',
                      'refusal_label', 'refusal_confidence',
                      'jailbreak_label', 'jailbreak_confidence']
        export_cols = [col for col in export_cols if col in low_both.columns]

        low_both[export_cols].to_csv(output_path, index=False)

        if self.verbose:
            print(f"\n✓ Exported {len(low_both)} low-confidence samples to {output_path}")

    def save_results(self, results: Dict, output_path: str):
        """Save analysis results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\n✓ Saved labeling quality analysis to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 31, 2025
@author: ramyalsaffar
"""
