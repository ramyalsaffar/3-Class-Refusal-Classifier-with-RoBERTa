# Refusal-Jailbreak Correlation Analysis
#---------------------------------------
# Investigates the relationship between refusal classification and jailbreak detection.
# Key question: Can jailbreak success be derived from refusal patterns alone?
# All imports are in 01-Imports.py
###############################################################################


# =============================================================================
# REFUSAL-JAILBREAK CORRELATION ANALYZER
# =============================================================================

class RefusalJailbreakCorrelationAnalyzer:
    """
    Analyze correlation between refusal classifier and jailbreak detector.

    Investigates whether jailbreak detection provides independent signal
    or can be derived from refusal classification.

    Key Research Question:
    "Do we need both classifiers, or is one redundant?"
    """

    def __init__(self,
                 refusal_preds: np.ndarray,
                 jailbreak_preds: np.ndarray,
                 refusal_labels: np.ndarray,
                 jailbreak_labels: np.ndarray,
                 texts: List[str],
                 refusal_class_names: List[str] = None,
                 jailbreak_class_names: List[str] = None):
        """
        Initialize correlation analyzer.

        Args:
            refusal_preds: Predicted refusal classes [N]
            jailbreak_preds: Predicted jailbreak classes [N]
            refusal_labels: True refusal labels [N]
            jailbreak_labels: True jailbreak labels [N]
            texts: Response texts [N]
            refusal_class_names: Names of refusal classes
            jailbreak_class_names: Names of jailbreak classes
        """
        self.refusal_preds = np.array(refusal_preds)
        self.jailbreak_preds = np.array(jailbreak_preds)
        self.refusal_labels = np.array(refusal_labels)
        self.jailbreak_labels = np.array(jailbreak_labels)
        self.texts = texts

        self.refusal_class_names = refusal_class_names if refusal_class_names else CLASS_NAMES
        self.jailbreak_class_names = jailbreak_class_names if jailbreak_class_names else JAILBREAK_CLASS_NAMES

        # Results storage
        self.analysis_results = {}

        print(f"\n{'='*60}")
        print("REFUSAL-JAILBREAK CORRELATION ANALYZER")
        print(f"{'='*60}")
        print(f"Samples: {len(self.refusal_preds)}")
        print(f"Refusal classes: {self.refusal_class_names}")
        print(f"Jailbreak classes: {self.jailbreak_class_names}")
        print(f"{'='*60}\n")


    # =========================================================================
    # MODULE 1: AGREEMENT ANALYSIS
    # =========================================================================

    def calculate_agreement(self) -> Dict:
        """
        Calculate agreement between refusal-derived and actual jailbreak predictions.

        Assumption: No Refusal (0) → Jailbreak Succeeded (1)
                   Hard Refusal (1) → Jailbreak Failed (0)
                   Soft Refusal (2) → Jailbreak Failed (0)

        Returns:
            Dictionary with agreement metrics
        """
        print(f"\n{'='*60}")
        print("MODULE 1: REFUSAL-DERIVED vs ACTUAL JAILBREAK")
        print(f"{'='*60}\n")

        # Derive expected jailbreak from refusal predictions
        # No Refusal (0) → Jailbreak Succeeded (1)
        # Hard/Soft Refusal (1, 2) → Jailbreak Failed (0)
        refusal_derived_jailbreak = np.where(self.refusal_preds == 0, 1, 0)

        # Calculate agreement
        agreement_mask = refusal_derived_jailbreak == self.jailbreak_preds
        agreement_rate = agreement_mask.mean()

        print(f"Agreement Mapping:")
        print(f"  No Refusal (class 0) → Jailbreak Succeeded (class 1)")
        print(f"  Hard Refusal (class 1) → Jailbreak Failed (class 0)")
        print(f"  Soft Refusal (class 2) → Jailbreak Failed (class 0)")
        print()

        print(f"Agreement Rate: {agreement_rate*100:.2f}%")
        print(f"Agreement: {agreement_mask.sum()} / {len(agreement_mask)}")
        print(f"Disagreement: {(~agreement_mask).sum()} / {len(agreement_mask)}")
        print()

        # Confusion matrix: refusal-derived vs actual jailbreak
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(refusal_derived_jailbreak, self.jailbreak_preds)

        print("Confusion Matrix (Refusal-Derived vs Actual Jailbreak):")
        print(f"{'-'*60}")
        print(f"{'Derived \\ Actual':<20} {'Failed (0)':<15} {'Succeeded (1)':<15}")
        print(f"{'-'*60}")
        print(f"{'Failed (0)':<20} {cm[0, 0]:<15} {cm[0, 1]:<15}")
        print(f"{'Succeeded (1)':<20} {cm[1, 0]:<15} {cm[1, 1]:<15}")
        print()

        # Interpretation
        if agreement_rate >= 0.95:
            interpretation = "HIGHLY CORRELATED - Jailbreak detector may be redundant"
            recommendation = "Consider using only refusal classifier and deriving jailbreak success"
        elif agreement_rate >= 0.85:
            interpretation = "MODERATELY CORRELATED - Some independent signal"
            recommendation = "Keep both classifiers but investigate disagreement cases"
        else:
            interpretation = "LOW CORRELATION - Jailbreak detector provides independent value"
            recommendation = "Dual-task classification is justified"

        print(f"Interpretation: {interpretation}")
        print(f"Recommendation: {recommendation}")
        print(f"{'='*60}\n")

        result = {
            'agreement_rate': float(agreement_rate),
            'agreement_count': int(agreement_mask.sum()),
            'disagreement_count': int((~agreement_mask).sum()),
            'confusion_matrix': cm,
            'refusal_derived_jailbreak': refusal_derived_jailbreak,
            'interpretation': interpretation,
            'recommendation': recommendation
        }

        self.analysis_results['agreement'] = result
        return result


    # =========================================================================
    # MODULE 2: PER-CLASS BREAKDOWN
    # =========================================================================

    def analyze_by_refusal_class(self) -> Dict:
        """
        Analyze jailbreak distribution for each refusal class.

        Shows what percentage of each refusal class results in jailbreak success/failure.

        Returns:
            Dictionary with per-class breakdown
        """
        print(f"\n{'='*60}")
        print("MODULE 2: JAILBREAK DISTRIBUTION BY REFUSAL CLASS")
        print(f"{'='*60}\n")

        breakdown = {}

        for refusal_idx, refusal_name in enumerate(self.refusal_class_names):
            # Get samples with this refusal prediction
            mask = self.refusal_preds == refusal_idx
            count = mask.sum()

            if count == 0:
                print(f"{refusal_name}: No samples")
                continue

            # Jailbreak distribution for this refusal class
            jailbreak_in_class = self.jailbreak_preds[mask]

            jailbreak_failed_count = (jailbreak_in_class == 0).sum()
            jailbreak_succeeded_count = (jailbreak_in_class == 1).sum()

            jailbreak_failed_pct = jailbreak_failed_count / count * 100
            jailbreak_succeeded_pct = jailbreak_succeeded_count / count * 100

            print(f"{refusal_name} (n={count}):")
            print(f"  Jailbreak Failed:    {jailbreak_failed_count:>5} ({jailbreak_failed_pct:>5.1f}%)")
            print(f"  Jailbreak Succeeded: {jailbreak_succeeded_count:>5} ({jailbreak_succeeded_pct:>5.1f}%)")
            print()

            breakdown[refusal_name] = {
                'total_count': int(count),
                'jailbreak_failed': {
                    'count': int(jailbreak_failed_count),
                    'percentage': float(jailbreak_failed_pct)
                },
                'jailbreak_succeeded': {
                    'count': int(jailbreak_succeeded_count),
                    'percentage': float(jailbreak_succeeded_pct)
                }
            }

        print(f"{'='*60}\n")

        self.analysis_results['per_class_breakdown'] = breakdown
        return breakdown


    # =========================================================================
    # MODULE 3: DISAGREEMENT CASE EXTRACTION
    # =========================================================================

    def extract_disagreement_cases(self, top_k: int = 50) -> pd.DataFrame:
        """
        Extract cases where refusal-derived jailbreak disagrees with actual jailbreak.

        These are the interesting cases that justify dual-task classification:
        1. No Refusal BUT Jailbreak Failed (complied but harmlessly)
        2. Hard Refusal BUT Jailbreak Succeeded (refused but leaked info)
        3. Soft Refusal BUT Jailbreak Succeeded (partial jailbreak success)

        Args:
            top_k: Number of disagreement cases to extract

        Returns:
            DataFrame with disagreement cases
        """
        print(f"\n{'='*60}")
        print(f"MODULE 3: DISAGREEMENT CASE EXTRACTION (Top {top_k})")
        print(f"{'='*60}\n")

        # Derive expected jailbreak
        refusal_derived_jailbreak = np.where(self.refusal_preds == 0, 1, 0)

        # Find disagreements
        disagreement_mask = refusal_derived_jailbreak != self.jailbreak_preds

        if disagreement_mask.sum() == 0:
            print("✓ PERFECT AGREEMENT - No disagreement cases found!")
            print("  Refusal classifier perfectly predicts jailbreak detection.")
            print(f"{'='*60}\n")
            return pd.DataFrame()

        # Extract disagreement data
        disagreement_data = {
            'text': [self.texts[i] for i in range(len(self.texts)) if disagreement_mask[i]],
            'refusal_pred': [self.refusal_class_names[self.refusal_preds[i]] for i in range(len(self.refusal_preds)) if disagreement_mask[i]],
            'refusal_label': [self.refusal_class_names[self.refusal_labels[i]] for i in range(len(self.refusal_labels)) if disagreement_mask[i]],
            'jailbreak_pred': [self.jailbreak_class_names[self.jailbreak_preds[i]] for i in range(len(self.jailbreak_preds)) if disagreement_mask[i]],
            'jailbreak_label': [self.jailbreak_class_names[self.jailbreak_labels[i]] for i in range(len(self.jailbreak_labels)) if disagreement_mask[i]],
            'derived_jailbreak': [self.jailbreak_class_names[refusal_derived_jailbreak[i]] for i in range(len(refusal_derived_jailbreak)) if disagreement_mask[i]],
            'disagreement_type': []
        }

        # Categorize disagreement types
        for i in range(len(self.texts)):
            if not disagreement_mask[i]:
                continue

            refusal_pred = self.refusal_preds[i]
            jailbreak_pred = self.jailbreak_preds[i]

            if refusal_pred == 0 and jailbreak_pred == 0:
                # No Refusal but Jailbreak Failed
                disagreement_type = "Type 1: No Refusal BUT Jailbreak Failed (harmless compliance)"
            elif refusal_pred == 1 and jailbreak_pred == 1:
                # Hard Refusal but Jailbreak Succeeded
                disagreement_type = "Type 2: Hard Refusal BUT Jailbreak Succeeded (info leak)"
            elif refusal_pred == 2 and jailbreak_pred == 1:
                # Soft Refusal but Jailbreak Succeeded
                disagreement_type = "Type 3: Soft Refusal BUT Jailbreak Succeeded (partial success)"
            else:
                disagreement_type = "Other"

            disagreement_data['disagreement_type'].append(disagreement_type)

        # Create DataFrame
        df = pd.DataFrame(disagreement_data)

        # Take top k
        df_top = df.head(top_k)

        print(f"Total disagreement cases: {len(df)}")
        print(f"Showing top {min(top_k, len(df))} cases\n")

        # Count by disagreement type
        print("Disagreement Type Distribution:")
        print(f"{'-'*60}")
        for dtype in df['disagreement_type'].unique():
            count = (df['disagreement_type'] == dtype).sum()
            pct = count / len(df) * 100
            print(f"{dtype}: {count} ({pct:.1f}%)")
        print()

        # Show examples
        print("Top 5 Disagreement Examples:")
        print(f"{'-'*60}")
        for i, row in df_top.head(5).iterrows():
            print(f"\nExample {i+1}: {row['disagreement_type']}")
            print(f"  Refusal Pred: {row['refusal_pred']} | Jailbreak Pred: {row['jailbreak_pred']}")
            print(f"  Derived Jailbreak: {row['derived_jailbreak']}")
            print(f"  Text: {row['text'][:100]}...")

        print(f"\n{'='*60}\n")

        # Save to CSV
        csv_path = os.path.join(results_path, f"{EXPERIMENT_CONFIG['experiment_name']}_disagreement_cases.csv")
        df_top.to_csv(csv_path, index=False)
        print(f"✓ Saved disagreement cases to: {csv_path}\n")

        self.analysis_results['disagreement_cases'] = df_top
        return df_top


    # =========================================================================
    # MODULE 4: STATISTICAL INDEPENDENCE TEST
    # =========================================================================

    def test_independence(self) -> Dict:
        """
        Chi-square test for independence between refusal and jailbreak.

        H0: Refusal class and jailbreak result are independent
        H1: They are correlated/dependent

        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*60}")
        print("MODULE 4: STATISTICAL INDEPENDENCE TEST")
        print(f"{'='*60}\n")

        from scipy.stats import chi2_contingency

        # Create contingency table: refusal class × jailbreak result
        contingency_table = np.zeros((len(self.refusal_class_names), len(self.jailbreak_class_names)))

        for i in range(len(self.refusal_preds)):
            refusal_idx = self.refusal_preds[i]
            jailbreak_idx = self.jailbreak_preds[i]
            contingency_table[refusal_idx, jailbreak_idx] += 1

        print("Contingency Table (Refusal × Jailbreak):")
        print(f"{'-'*60}")
        header = "Refusal \\ Jailbreak".ljust(25) + "".join([f"{name[:12]:>15}" for name in self.jailbreak_class_names])
        print(header)
        print(f"{'-'*60}")
        for i, refusal_name in enumerate(self.refusal_class_names):
            row = f"{refusal_name[:25]:<25}" + "".join([f"{int(contingency_table[i, j]):>15}" for j in range(len(self.jailbreak_class_names))])
            print(row)
        print()

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        print(f"Chi-Square Test for Independence:")
        print(f"{'-'*60}")
        print(f"Chi-square statistic (χ²): {chi2:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Degrees of freedom: {dof}")
        print()

        # Cramér's V (effect size)
        n = contingency_table.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        print(f"Cramér's V (effect size): {cramers_v:.4f}")
        print(f"  0.0-0.1: Negligible")
        print(f"  0.1-0.3: Weak")
        print(f"  0.3-0.5: Moderate")
        print(f"  >0.5: Strong")
        print()

        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            print(f"✗ REJECT NULL HYPOTHESIS (p = {p_value:.6f} < α = {alpha})")
            print(f"  → Refusal and jailbreak are DEPENDENT (correlated)")
            interpretation = "dependent"
        else:
            print(f"✓ FAIL TO REJECT NULL HYPOTHESIS (p = {p_value:.6f} >= α = {alpha})")
            print(f"  → Refusal and jailbreak are INDEPENDENT")
            interpretation = "independent"

        print(f"{'='*60}\n")

        result = {
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'cramers_v': float(cramers_v),
            'contingency_table': contingency_table.tolist(),
            'expected_frequencies': expected.tolist(),
            'interpretation': interpretation
        }

        self.analysis_results['independence_test'] = result
        return result


    # =========================================================================
    # MODULE 5: VISUALIZATIONS
    # =========================================================================

    def visualize_correlation(self, save_visualizations: bool = True) -> Dict:
        """
        Create correlation visualizations.

        Creates:
        1. Contingency table heatmap
        2. Agreement/disagreement pie chart
        3. Sankey diagram (refusal class → jailbreak result)

        Args:
            save_visualizations: Whether to save plots

        Returns:
            Dictionary with visualization paths
        """
        print(f"\n{'='*60}")
        print("MODULE 5: CORRELATION VISUALIZATIONS")
        print(f"{'='*60}\n")

        viz_paths = {}

        if not save_visualizations:
            print("Visualization saving disabled")
            return viz_paths

        # 1. Contingency Table Heatmap
        print("Creating contingency table heatmap...")

        contingency_table = np.zeros((len(self.refusal_class_names), len(self.jailbreak_class_names)))
        for i in range(len(self.refusal_preds)):
            contingency_table[self.refusal_preds[i], self.jailbreak_preds[i]] += 1

        # Normalize by row (percentage within each refusal class)
        contingency_pct = contingency_table / contingency_table.sum(axis=1, keepdims=True) * 100

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw counts
        sns.heatmap(contingency_table, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=self.jailbreak_class_names,
                   yticklabels=self.refusal_class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Jailbreak Result', fontsize=12)
        axes[0].set_ylabel('Refusal Class', fontsize=12)
        axes[0].set_title('Contingency Table (Counts)', fontsize=14, fontweight='bold')

        # Percentages
        sns.heatmap(contingency_pct, annot=True, fmt='.1f', cmap='Oranges',
                   xticklabels=self.jailbreak_class_names,
                   yticklabels=self.refusal_class_names,
                   ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
        axes[1].set_xlabel('Jailbreak Result', fontsize=12)
        axes[1].set_ylabel('Refusal Class', fontsize=12)
        axes[1].set_title('Contingency Table (Row %)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        heatmap_path = os.path.join(visualizations_path, "correlation_contingency_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths['heatmap'] = heatmap_path
        print(f"✓ Saved heatmap: {heatmap_path}")

        # 2. Agreement/Disagreement Pie Chart
        print("Creating agreement pie chart...")

        refusal_derived_jailbreak = np.where(self.refusal_preds == 0, 1, 0)
        agreement_mask = refusal_derived_jailbreak == self.jailbreak_preds

        agreement_count = agreement_mask.sum()
        disagreement_count = (~agreement_mask).sum()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)

        ax.pie([agreement_count, disagreement_count],
               labels=['Agreement', 'Disagreement'],
               autopct='%1.1f%%',
               colors=colors,
               explode=explode,
               shadow=True,
               startangle=90)
        ax.set_title('Refusal-Derived vs Actual Jailbreak Agreement', fontsize=14, fontweight='bold')

        pie_path = os.path.join(visualizations_path, "correlation_agreement_pie.png")
        plt.savefig(pie_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths['pie'] = pie_path
        print(f"✓ Saved pie chart: {pie_path}")

        # 3. Stacked Bar Chart (Jailbreak distribution per refusal class)
        print("Creating stacked bar chart...")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Calculate percentages
        jailbreak_failed = []
        jailbreak_succeeded = []

        for refusal_idx in range(len(self.refusal_class_names)):
            mask = self.refusal_preds == refusal_idx
            if mask.sum() == 0:
                jailbreak_failed.append(0)
                jailbreak_succeeded.append(0)
            else:
                jailbreak_in_class = self.jailbreak_preds[mask]
                jailbreak_failed.append((jailbreak_in_class == 0).sum())
                jailbreak_succeeded.append((jailbreak_in_class == 1).sum())

        x = np.arange(len(self.refusal_class_names))
        width = 0.6

        p1 = ax.bar(x, jailbreak_failed, width, label='Jailbreak Failed', color='#27ae60')
        p2 = ax.bar(x, jailbreak_succeeded, width, bottom=jailbreak_failed,
                   label='Jailbreak Succeeded', color='#c0392b')

        ax.set_xlabel('Refusal Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Jailbreak Distribution by Refusal Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.refusal_class_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        bar_path = os.path.join(visualizations_path, "correlation_stacked_bar.png")
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths['stacked_bar'] = bar_path
        print(f"✓ Saved stacked bar: {bar_path}")

        print(f"\n{'='*60}\n")

        self.analysis_results['visualizations'] = viz_paths
        return viz_paths


    # =========================================================================
    # COMPREHENSIVE ANALYSIS RUNNER
    # =========================================================================

    def run_full_analysis(self, save_visualizations: bool = True, top_k_disagreements: int = 50) -> Dict:
        """
        Run all correlation analysis modules.

        Args:
            save_visualizations: Whether to save visualizations
            top_k_disagreements: Number of disagreement cases to extract

        Returns:
            Complete analysis results
        """
        print(f"\n{'#'*60}")
        print("COMPREHENSIVE REFUSAL-JAILBREAK CORRELATION ANALYSIS")
        print(f"{'#'*60}\n")

        # Run all modules
        self.calculate_agreement()
        self.analyze_by_refusal_class()
        self.extract_disagreement_cases(top_k=top_k_disagreements)
        self.test_independence()
        self.visualize_correlation(save_visualizations=save_visualizations)

        # Save results
        self.save_analysis_results()

        print(f"\n{'='*60}")
        print("CORRELATION ANALYSIS COMPLETE")
        print(f"{'='*60}\n")

        return self.analysis_results


    def save_analysis_results(self, output_path: str = None) -> str:
        """
        Save analysis results to pickle file.

        Args:
            output_path: Path to save results

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = os.path.join(
                results_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_correlation_analysis.pkl"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(self.analysis_results, f)

        print(f"✓ Correlation analysis results saved: {output_path}\n")

        return output_path


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_refusal_jailbreak_correlation(
    refusal_preds: np.ndarray,
    jailbreak_preds: np.ndarray,
    refusal_labels: np.ndarray,
    jailbreak_labels: np.ndarray,
    texts: List[str],
    save_visualizations: bool = True
) -> Dict:
    """
    Quick function to run complete correlation analysis.

    Args:
        refusal_preds: Predicted refusal classes
        jailbreak_preds: Predicted jailbreak classes
        refusal_labels: True refusal labels
        jailbreak_labels: True jailbreak labels
        texts: Response texts
        save_visualizations: Whether to save plots

    Returns:
        Complete analysis results
    """
    analyzer = RefusalJailbreakCorrelationAnalyzer(
        refusal_preds=refusal_preds,
        jailbreak_preds=jailbreak_preds,
        refusal_labels=refusal_labels,
        jailbreak_labels=jailbreak_labels,
        texts=texts,
        refusal_class_names=CLASS_NAMES,
        jailbreak_class_names=JAILBREAK_CLASS_NAMES
    )

    results = analyzer.run_full_analysis(
        save_visualizations=save_visualizations,
        top_k_disagreements=50
    )

    return results


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 1, 2025
@author: ramyalsaffar
"""
