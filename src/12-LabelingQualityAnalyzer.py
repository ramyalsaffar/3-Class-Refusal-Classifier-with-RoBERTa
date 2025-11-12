# Labeling Quality Analyzer Module 
#----------------------------------
# Analyzes judge labeling quality and confidence scores.
# Helps identify low-quality labels and systematic issues.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Handles new is_jailbreak_attempt field
# - Better statistics with safe_divide
# - Visualization support
# - Export capabilities enhanced
# All imports are in 01-Imports.py
###############################################################################


class LabelingQualityAnalyzer:
    """Analyze judge labeling quality using confidence scores."""
    
    def __init__(self, verbose: bool = None):
        """
        Initialize analyzer.
        
        Args:
            verbose: Print detailed output (default: from EXPERIMENT_CONFIG)
        """
        # Use config values - NO HARDCODING!
        self.verbose = verbose if verbose is not None else EXPERIMENT_CONFIG.get('verbose', True)
        self.class_names = CLASS_NAMES
        self.threshold = LABELING_CONFIG['low_confidence_threshold']
        
        # Statistics tracking
        self.stats = {
            'total_analyzed': 0,
            'error_labels': 0,
            'low_confidence_samples': 0,
            'analysis_timestamp': None
        }

    def analyze_full(self, labeled_df: pd.DataFrame) -> Dict:
        """
        Run complete labeling quality analysis.
        
        Args:
            labeled_df: DataFrame with columns:
                - refusal_label, refusal_confidence
                - is_jailbreak_attempt, jailbreak_label, jailbreak_confidence
                - model (LLM that generated response)
                - category (prompt category)
        
        Returns:
            Dictionary with all analysis results
        """
        if self.verbose:
            print_banner("LABELING QUALITY ANALYSIS", char="=")
        
        self.stats['total_analyzed'] = len(labeled_df)
        self.stats['analysis_timestamp'] = get_timestamp('display')
        
        results = {
            'timestamp': self.stats['analysis_timestamp'],
            'total_samples': len(labeled_df)
        }
        
        # Overall confidence statistics
        results['overall'] = self._analyze_overall_confidence(labeled_df)
        
        # Jailbreak attempt analysis (NEW)
        if 'is_jailbreak_attempt' in labeled_df.columns:
            results['jailbreak_attempts'] = self._analyze_jailbreak_attempts(labeled_df)
        
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
        
        # Label distribution
        results['label_distribution'] = self._analyze_label_distribution(labeled_df)
        
        if self.verbose:
            self._print_summary(results)
        
        return results

    def _analyze_overall_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze overall confidence statistics."""
        if self.verbose:
            print_banner("Overall Confidence Statistics", char="─")
        
        # Filter out error labels
        valid_refusal = df[df['refusal_label'] != -1]
        valid_jailbreak = df[df['jailbreak_label'] != -1]

        # Calculate statistics safely
        results = {
            'refusal': self._calculate_confidence_stats(
                valid_refusal['refusal_confidence'],
                'Refusal'
            ),
            'jailbreak': self._calculate_confidence_stats(
                valid_jailbreak['jailbreak_confidence'],
                'Jailbreak'
            )
        }

        # Track error rates
        refusal_error_rate = safe_divide(
            (df['refusal_label'] == -1).sum(),
            len(df), 0
        ) * 100
        jailbreak_error_rate = safe_divide(
            (df['jailbreak_label'] == -1).sum(),
            len(df), 0
        ) * 100
        
        results['error_rates'] = {
            'refusal': refusal_error_rate,
            'jailbreak': jailbreak_error_rate
        }
        
        if self.verbose:
            print(f"  Refusal - Mean: {results['refusal']['mean']:.1f}%, "
                  f"Median: {results['refusal']['median']:.1f}%, "
                  f"Low (<{self.threshold}%): {results['refusal']['count_low']} "
                  f"({results['refusal']['percent_low']:.1f}%)")
            print(f"  Jailbreak - Mean: {results['jailbreak']['mean']:.1f}%, "
                  f"Median: {results['jailbreak']['median']:.1f}%, "
                  f"Low (<{self.threshold}%): {results['jailbreak']['count_low']} "
                  f"({results['jailbreak']['percent_low']:.1f}%)")
            print(f"  Error rates - Refusal: {refusal_error_rate:.1f}%, "
                  f"Jailbreak: {jailbreak_error_rate:.1f}%")
        
        return results

    def _calculate_confidence_stats(self, confidence_series: pd.Series, 
                                   label_name: str) -> Dict:
        """Calculate confidence statistics safely."""
        if len(confidence_series) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'min': 0.0, 'max': 0.0, 'count_low': 0, 'percent_low': 0.0
            }
        
        count_low = (confidence_series < self.threshold).sum()
        percent_low = safe_divide(count_low, len(confidence_series), 0) * 100
        
        return {
            'mean': float(confidence_series.mean()),
            'std': float(confidence_series.std()),
            'median': float(confidence_series.median()),
            'min': float(confidence_series.min()),
            'max': float(confidence_series.max()),
            'count_low': int(count_low),
            'percent_low': float(percent_low)
        }

    def _analyze_jailbreak_attempts(self, df: pd.DataFrame) -> Dict:
        """Analyze jailbreak attempt detection (NEW)."""
        if self.verbose:
            print_banner("Jailbreak Attempt Analysis", char="─")
        
        total = len(df)
        attempts = df['is_jailbreak_attempt'].sum()
        attempt_rate = safe_divide(attempts, total, 0) * 100
        
        # Among attempts, how many succeeded?
        attempts_df = df[df['is_jailbreak_attempt'] == 1]
        if len(attempts_df) > 0:
            successes = (attempts_df['jailbreak_label'] == 1).sum()
            success_rate = safe_divide(successes, len(attempts_df), 0) * 100
        else:
            successes = 0
            success_rate = 0.0
        
        results = {
            'total_attempts': int(attempts),
            'attempt_rate': float(attempt_rate),
            'total_successes': int(successes),
            'success_rate': float(success_rate),
            'defense_rate': float(100 - success_rate)
        }
        
        if self.verbose:
            print(f"  Jailbreak attempts: {attempts:,}/{total:,} ({attempt_rate:.1f}%)")
            if attempts > 0:
                print(f"  Success rate: {successes:,}/{attempts:,} ({success_rate:.1f}%)")
                print(f"  Defense rate: {100 - success_rate:.1f}%")
        
        return results

    def _analyze_per_model_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze confidence by LLM model."""
        if self.verbose:
            print_banner("Per-Model Confidence", char="─")
        
        results = {}
        valid_df = df[df['refusal_label'] != -1]
        
        for model in valid_df['model'].unique():
            model_df = valid_df[valid_df['model'] == model]
            
            refusal_low = (model_df['refusal_confidence'] < self.threshold).sum()
            jailbreak_low = (model_df['jailbreak_confidence'] < self.threshold).sum()
            
            results[model] = {
                'sample_count': len(model_df),
                'refusal': {
                    'mean': float(model_df['refusal_confidence'].mean()),
                    'count_low': int(refusal_low),
                    'percent_low': safe_divide(refusal_low, len(model_df), 0) * 100
                },
                'jailbreak': {
                    'mean': float(model_df['jailbreak_confidence'].mean()),
                    'count_low': int(jailbreak_low),
                    'percent_low': safe_divide(jailbreak_low, len(model_df), 0) * 100
                }
            }
            
            # Add jailbreak attempt stats if available
            if 'is_jailbreak_attempt' in model_df.columns:
                attempts = model_df['is_jailbreak_attempt'].sum()
                results[model]['jailbreak_attempts'] = int(attempts)
                results[model]['attempt_rate'] = safe_divide(attempts, len(model_df), 0) * 100
            
            if self.verbose:
                display_name = get_model_display_name(model) if 'get_model_display_name' in globals() else model
                print(f"\n  {display_name}:")
                print(f"    Samples: {len(model_df):,}")
                print(f"    Refusal - Mean: {results[model]['refusal']['mean']:.1f}%, "
                      f"Low: {refusal_low} ({results[model]['refusal']['percent_low']:.1f}%)")
                print(f"    Jailbreak - Mean: {results[model]['jailbreak']['mean']:.1f}%, "
                      f"Low: {jailbreak_low} ({results[model]['jailbreak']['percent_low']:.1f}%)")
        
        return results

    def _analyze_per_category_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze confidence by prompt category."""
        if self.verbose:
            print_banner("Per-Category Confidence", char="─")
        
        results = {}
        valid_df = df[df['refusal_label'] != -1]
        
        if 'category' not in valid_df.columns:
            if self.verbose:
                print("  No category column found")
            return results
        
        for category in valid_df['category'].unique():
            cat_df = valid_df[valid_df['category'] == category]
            
            refusal_low = (cat_df['refusal_confidence'] < self.threshold).sum()
            jailbreak_low = (cat_df['jailbreak_confidence'] < self.threshold).sum()
            
            results[category] = {
                'sample_count': len(cat_df),
                'refusal': {
                    'mean': float(cat_df['refusal_confidence'].mean()),
                    'count_low': int(refusal_low),
                    'percent_low': safe_divide(refusal_low, len(cat_df), 0) * 100
                },
                'jailbreak': {
                    'mean': float(cat_df['jailbreak_confidence'].mean()),
                    'count_low': int(jailbreak_low),
                    'percent_low': safe_divide(jailbreak_low, len(cat_df), 0) * 100
                }
            }
        
        # Print top 5 categories with lowest confidence
        if self.verbose and results:
            sorted_cats = sorted(results.items(),
                               key=lambda x: x[1]['refusal']['mean'])[:5]
            print("\n  Top 5 categories with lowest refusal confidence:")
            for cat, metrics in sorted_cats:
                print(f"    {cat}: {metrics['refusal']['mean']:.1f}% "
                      f"(n={metrics['sample_count']})")
        
        return results

    def _flag_low_confidence(self, df: pd.DataFrame) -> Dict:
        """Flag samples with low confidence for review."""
        if self.verbose:
            print_banner(f"Low Confidence Flags (threshold: {self.threshold}%)", char="─")
        
        valid_df = df[df['refusal_label'] != -1]
        
        # Find low confidence samples
        low_refusal = valid_df[valid_df['refusal_confidence'] < self.threshold]
        low_jailbreak = valid_df[valid_df['jailbreak_confidence'] < self.threshold]
        low_both = valid_df[(valid_df['refusal_confidence'] < self.threshold) &
                           (valid_df['jailbreak_confidence'] < self.threshold)]
        
        self.stats['low_confidence_samples'] = len(low_both)
        
        results = {
            'threshold': self.threshold,
            'low_refusal_count': len(low_refusal),
            'low_jailbreak_count': len(low_jailbreak),
            'low_both_count': len(low_both),
            'low_refusal_percent': safe_divide(len(low_refusal), len(valid_df), 0) * 100,
            'low_jailbreak_percent': safe_divide(len(low_jailbreak), len(valid_df), 0) * 100,
            'low_both_percent': safe_divide(len(low_both), len(valid_df), 0) * 100
        }
        
        if self.verbose:
            print(f"  Low refusal confidence: {results['low_refusal_count']:,} "
                  f"({results['low_refusal_percent']:.1f}%)")
            print(f"  Low jailbreak confidence: {results['low_jailbreak_count']:,} "
                  f"({results['low_jailbreak_percent']:.1f}%)")
            print(f"  Low confidence in BOTH: {results['low_both_count']:,} "
                  f"({results['low_both_percent']:.1f}%)")
        
        # Save flagged samples for review
        results['flagged_samples'] = []
        for idx, row in low_both.head(100).iterrows():  # Limit to 100 for JSON size
            results['flagged_samples'].append({
                'index': int(idx),
                'model': row.get('model', 'unknown'),
                'category': row.get('category', 'unknown'),
                'refusal_label': int(row['refusal_label']),
                'refusal_confidence': float(row['refusal_confidence']),
                'jailbreak_label': int(row.get('jailbreak_label', -1)),
                'jailbreak_confidence': float(row['jailbreak_confidence']),
                'is_jailbreak_attempt': int(row.get('is_jailbreak_attempt', 0))
            })
        
        if self.verbose and results['flagged_samples']:
            print(f"  Flagged {len(results['flagged_samples'])} samples for review")
        
        return results

    def _analyze_task_agreement(self, df: pd.DataFrame) -> Dict:
        """Analyze agreement between refusal and jailbreak labeling."""
        if self.verbose:
            print_banner("Task Agreement Analysis", char="─")

        valid_df = df[(df['refusal_label'] != -1) & (df['jailbreak_label'] != -1)]

        # Confidence correlation
        correlation = valid_df['refusal_confidence'].corr(valid_df['jailbreak_confidence'])


        alpha = HYPOTHESIS_TESTING_CONFIG.get('alpha', 0.05)
        n = len(valid_df)
        if n > 3:  # Need at least 4 samples
            # Test if correlation is significantly different from 0
            t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            correlation_significant = p_value < alpha
        else:
            p_value = 1.0
            correlation_significant = False




        # Logical consistency checks
        # No refusal (0) but jailbreak succeeded (1) - expected for successful jailbreaks
        no_refusal_jb_success = len(valid_df[
            (valid_df['refusal_label'] == 0) &
            (valid_df['jailbreak_label'] == 1)
        ])

        # Refusal (1 or 2) but jailbreak succeeded (1) - inconsistent!
        refusal_but_jb_success = len(valid_df[
            (valid_df['refusal_label'].isin([1, 2])) &
            (valid_df['jailbreak_label'] == 1)
        ])
        
        results = {
            'confidence_correlation': float(correlation) if not pd.isna(correlation) else 0.0,
            'correlation_p_value': float(p_value),                      
            'correlation_significant': bool(correlation_significant),
            'no_refusal_jb_success': int(no_refusal_jb_success),
            'refusal_but_jb_success': int(refusal_but_jb_success),  # This is problematic
            'inconsistency_rate': safe_divide(refusal_but_jb_success, len(valid_df), 0) * 100
        }
        
        if self.verbose:
            print(f"  Confidence correlation: {results['confidence_correlation']:.3f} "
                  f"(p={p_value:.4f}, {'significant' if correlation_significant else 'not significant'})")
            print(f"  No refusal + jailbreak success: {no_refusal_jb_success:,} (expected)")
            print(f"  Refusal + jailbreak success: {refusal_but_jb_success:,} "
                  f"({results['inconsistency_rate']:.1f}% - INCONSISTENT!)")
        
        return results

    def _analyze_label_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze label distribution."""
        results = {}
        
        # Refusal distribution
        if 'refusal_label' in df.columns:
            refusal_dist = df['refusal_label'].value_counts().to_dict()
            results['refusal'] = {
                (self.class_names[k] if 0 <= k < len(self.class_names) else f'Label_{k}'): v
                for k, v in refusal_dist.items()
            }
        
        # Jailbreak distribution
        if 'jailbreak_label' in df.columns:
            jb_dist = df['jailbreak_label'].value_counts().to_dict()
            results['jailbreak'] = {
                'Failed': jb_dist.get(0, 0),
                'Succeeded': jb_dist.get(1, 0),
                'Error': jb_dist.get(-1, 0)
            }
        
        # Jailbreak attempt distribution
        if 'is_jailbreak_attempt' in df.columns:
            attempt_dist = df['is_jailbreak_attempt'].value_counts().to_dict()
            results['jailbreak_attempts'] = {
                'Not Attempt': attempt_dist.get(0, 0),
                'Is Attempt': attempt_dist.get(1, 0)
            }
        
        return results

    def _print_summary(self, results: Dict):
        """Print analysis summary."""
        print_banner("ANALYSIS SUMMARY", char="=")
        print(f"Total samples analyzed: {results['total_samples']:,}")
        print(f"Timestamp: {results['timestamp']}")
        
        if 'jailbreak_attempts' in results:
            print(f"\nJailbreak Statistics:")
            print(f"  Attempt rate: {results['jailbreak_attempts']['attempt_rate']:.1f}%")
            print(f"  Success rate: {results['jailbreak_attempts']['success_rate']:.1f}%")
        
        if 'low_confidence' in results:
            print(f"\nLow Confidence Samples:")
            print(f"  Both tasks: {results['low_confidence']['low_both_count']:,} "
                  f"({results['low_confidence']['low_both_percent']:.1f}%)")
        
        if 'task_agreement' in results:
            print(f"\nTask Agreement:")
            print(f"  Inconsistency rate: {results['task_agreement']['inconsistency_rate']:.1f}%")

    def export_flagged_samples(self, labeled_df: pd.DataFrame, output_path: str):
        """Export low-confidence samples for manual review."""
        ensure_dir_exists(os.path.dirname(output_path))
        
        valid_df = labeled_df[labeled_df['refusal_label'] != -1]
        low_both = valid_df[
            (valid_df['refusal_confidence'] < self.threshold) &
            (valid_df['jailbreak_confidence'] < self.threshold)
        ]
        
        # Select relevant columns
        export_cols = [
            'prompt', 'response', 'model', 'category',
            'refusal_label', 'refusal_confidence',
            'is_jailbreak_attempt', 'jailbreak_label', 'jailbreak_confidence'
        ]
        export_cols = [col for col in export_cols if col in low_both.columns]
        
        low_both[export_cols].to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"\n✅ Exported {len(low_both):,} low-confidence samples to {output_path}")

    def save_results(self, results: Dict, output_path: str):
        """Save analysis results to JSON."""
        ensure_dir_exists(os.path.dirname(output_path))

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Saved labeling quality analysis to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 31, 2025
@author: ramyalsaffar
"""
