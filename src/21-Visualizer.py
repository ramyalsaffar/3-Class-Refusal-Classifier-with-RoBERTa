# Visualization Module
#---------------------
# Generate visualizations for results.
# All imports are in 00-Imports.py
###############################################################################


class Visualizer:
    """Generate visualizations for results."""

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.class_colors = PLOT_COLORS_LIST
        self.model_colors = MODEL_COLORS
        self.dpi = VISUALIZATION_CONFIG['dpi']
        sns.set_style(VISUALIZATION_CONFIG['style'])

    def plot_confusion_matrix(self, cm: np.ndarray, output_path: str):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confusion matrix to {output_path}")

    def plot_per_class_f1(self, f1_scores: Dict[str, float], output_path: str):
        """Plot per-class F1 scores."""
        plt.figure(figsize=(10, 6))
        classes = list(f1_scores.keys())
        scores = list(f1_scores.values())

        bars = plt.bar(classes, scores, color=self.class_colors, alpha=0.8, edgecolor='black')
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Per-Class F1 Scores', fontsize=16, fontweight='bold')
        plt.ylim([0, 1])
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target (0.80)')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved per-class F1 to {output_path}")

    def plot_per_model_f1(self, results: Dict, output_path: str):
        """Plot per-model F1 comparison."""
        models = [m for m in results.keys() if m != 'analysis']
        f1_scores = [results[m]['f1_macro'] for m in models]

        plt.figure(figsize=(10, 6))
        colors = [self.model_colors.get(m, '#95a5a6') for m in models]
        bars = plt.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black')

        plt.ylabel('F1 Score (Macro)', fontsize=12)
        plt.title('Cross-Model Performance', fontsize=16, fontweight='bold')
        plt.ylim([0, 1])
        plt.axhline(y=np.mean(f1_scores), color='red', linestyle='--',
                   alpha=0.5, label=f'Mean ({np.mean(f1_scores):.3f})')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved per-model F1 to {output_path}")

    def plot_adversarial_robustness(self, results: Dict, output_path: str):
        """Plot adversarial robustness comparison."""
        dimensions = list(results['paraphrased_f1'].keys())
        paraphrased_f1 = list(results['paraphrased_f1'].values())
        original_f1 = results['original_f1']

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(dimensions))
        width = 0.35

        # Original (repeated for comparison)
        ax.bar(x - width/2, [original_f1] * len(dimensions), width,
               label='Original', color='#2ecc71', alpha=0.8, edgecolor='black')

        # Paraphrased
        ax.bar(x + width/2, paraphrased_f1, width,
               label='Paraphrased', color='#e74c3c', alpha=0.8, edgecolor='black')

        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Adversarial Robustness: Original vs Paraphrased',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions)
        ax.legend()
        ax.set_ylim([0, 1])

        # Add drop percentage
        for i, (orig, para) in enumerate(zip([original_f1] * len(dimensions), paraphrased_f1)):
            drop = (orig - para) / orig * 100
            ax.text(i, max(orig, para) + 0.02, f'-{drop:.1f}%',
                   ha='center', fontsize=9, color='red')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved adversarial robustness to {output_path}")

    def plot_confidence_distributions(self, labels: List, confidences: List, output_path: str):
        """
        Plot confidence distributions per class.

        GENERIC: Works with any number of classes (2, 3, or more).
        """
        # Dynamic subplot configuration based on number of classes
        fig, axes = plt.subplots(1, self.num_classes, figsize=(5 * self.num_classes, 4))

        # Handle single class case (axes won't be a list)
        if self.num_classes == 1:
            axes = [axes]

        for class_idx in range(self.num_classes):
            class_confidences = [c for l, c in zip(labels, confidences) if l == class_idx]

            # Use modulo for color cycling if we have more classes than colors
            color_idx = class_idx % len(self.class_colors)

            axes[class_idx].hist(class_confidences, bins=ANALYSIS_CONFIG['confidence_bins'],
                               alpha=0.7, color=self.class_colors[color_idx], edgecolor='black')
            axes[class_idx].set_title(f'{self.class_names[class_idx]}',
                                     fontsize=12, fontweight='bold')
            axes[class_idx].set_xlabel('Confidence', fontsize=10)
            axes[class_idx].set_ylabel('Count', fontsize=10)
            axes[class_idx].set_xlim([0, 1])

            if len(class_confidences) > 0:
                axes[class_idx].axvline(x=np.mean(class_confidences), color='red',
                                       linestyle='--', label=f'Mean: {np.mean(class_confidences):.3f}')
                axes[class_idx].legend()

        plt.suptitle('Confidence Distributions by Class', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confidence distributions to {output_path}")

    def plot_training_curves(self, history: Dict, output_path: str):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#3498db', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'o-', label='Val Loss', color='#e74c3c', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # F1 curves
        ax2.plot(epochs, history['val_f1'], 'o-', label='Val F1', color='#2ecc71', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('Validation F1', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
