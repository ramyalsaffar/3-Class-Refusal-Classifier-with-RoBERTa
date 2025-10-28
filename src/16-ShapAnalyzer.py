# SHAP Analyzer Module
#----------------------
# Computes SHAP values for model interpretability.
# All imports are in 00-Imports.py
###############################################################################


class ShapAnalyzer:
    """Compute and visualize SHAP values for model interpretability."""

    def __init__(self, model, tokenizer, device):
        """
        Initialize SHAP analyzer.

        Args:
            model: Trained RefusalClassifier
            tokenizer: RoBERTa tokenizer
            device: torch device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for texts (needed for SHAP).

        Args:
            texts: List of text strings

        Returns:
            Probability array (num_samples, num_classes)
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for text in texts:
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

                # Get prediction
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs[0].cpu().numpy())

        return np.array(all_probs)

    def compute_shap_values(self, texts: List[str], background_texts: List[str] = None,
                           max_background: int = 50):
        """
        Compute SHAP values for text samples.

        Args:
            texts: List of texts to explain
            background_texts: Background dataset for SHAP (uses subset if None)
            max_background: Maximum background samples to use

        Returns:
            Dictionary with SHAP values and metadata
        """
        try:
            import shap
        except ImportError:
            print("❌ SHAP not installed. Install with: pip install shap")
            return None

        print("\n" + "="*60)
        print("COMPUTING SHAP VALUES")
        print("="*60)

        # Prepare background data
        if background_texts is None:
            background_texts = texts[:max_background]
        else:
            background_texts = background_texts[:max_background]

        print(f"Background samples: {len(background_texts)}")
        print(f"Texts to explain: {len(texts)}")

        # Create SHAP explainer
        print("\nInitializing SHAP explainer...")
        explainer = shap.Explainer(self._predict_proba, background_texts)

        # Compute SHAP values
        print("Computing SHAP values (this may take a while)...")
        shap_values = explainer(texts)

        print("✓ SHAP computation complete")

        return {
            'shap_values': shap_values,
            'texts': texts,
            'background_texts': background_texts
        }

    def visualize_shap(self, text: str, shap_data: dict, output_path: str,
                      class_idx: int = None):
        """
        Create SHAP visualization for a single text.

        Args:
            text: Input text to visualize
            shap_data: SHAP data from compute_shap_values
            output_path: Path to save visualization
            class_idx: Which class to visualize (None for predicted class)
        """
        try:
            import shap
        except ImportError:
            print("❌ SHAP not installed")
            return

        # Find the text in shap_data
        text_idx = shap_data['texts'].index(text) if text in shap_data['texts'] else 0
        shap_values = shap_data['shap_values']

        # Create waterfall plot
        if class_idx is None:
            # Use predicted class
            probs = self._predict_proba([text])[0]
            class_idx = np.argmax(probs)

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[text_idx, :, class_idx], show=False)
        plt.title(
            f'SHAP Values for {CLASS_NAMES[class_idx]}\n{text[:80]}...',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved SHAP visualization to {output_path}")

    def analyze_samples(self, test_df: pd.DataFrame, num_samples: int = 20,
                       output_dir: str = None):
        """
        Analyze SHAP values for multiple samples.

        Args:
            test_df: Test DataFrame
            num_samples: Number of samples to analyze
            output_dir: Directory to save results

        Returns:
            Dictionary with SHAP analysis results
        """
        if output_dir is None:
            output_dir = os.path.join(visualizations_path, "shap_analysis")
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*60)
        print("SHAP ANALYSIS")
        print("="*60)

        # Sample texts
        sample_indices = np.random.choice(
            len(test_df),
            min(num_samples, len(test_df)),
            replace=False
        )
        sampled_df = test_df.iloc[sample_indices]

        texts = sampled_df['response'].tolist()
        labels = sampled_df['label'].tolist()

        # Compute SHAP values
        shap_data = self.compute_shap_values(texts)

        if shap_data is None:
            return None

        # Visualize samples from each class
        results = {'by_class': {}, 'examples': []}

        for class_idx in range(3):
            class_name = CLASS_NAMES[class_idx]
            class_mask = np.array(labels) == class_idx

            if not class_mask.any():
                continue

            class_indices = np.where(class_mask)[0]

            print(f"\nVisualizing {class_name} samples...")

            # Visualize first 2 samples from this class
            for i, idx in enumerate(class_indices[:2]):
                text = texts[idx]
                output_path = os.path.join(
                    output_dir,
                    f"shap_{class_name.replace(' ', '_').lower()}_sample_{i+1}.png"
                )
                self.visualize_shap(text, shap_data, output_path, class_idx)

                results['examples'].append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'true_label': class_name,
                    'class_idx': int(class_idx)
                })

        # Create summary plot
        try:
            import shap
            print("\nCreating SHAP summary plot...")

            # For each class, create a summary
            for class_idx in range(3):
                class_name = CLASS_NAMES[class_idx]

                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_data['shap_values'].values[:, :, class_idx],
                    texts,
                    show=False,
                    max_display=20
                )
                plt.title(
                    f'SHAP Summary - {class_name}',
                    fontsize=14, fontweight='bold'
                )
                summary_path = os.path.join(
                    output_dir,
                    f"shap_summary_{class_name.replace(' ', '_').lower()}.png"
                )
                plt.tight_layout()
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"✓ Saved summary for {class_name}")

        except Exception as e:
            print(f"⚠️  Could not create summary plot: {e}")

        # Save results
        results_path = os.path.join(output_dir, "shap_analysis_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved SHAP analysis to {results_path}")

        return results

    def get_top_features(self, text: str, class_idx: int = None, top_k: int = 10):
        """
        Get top features (tokens) influencing prediction.

        Args:
            text: Input text
            class_idx: Target class (None for predicted)
            top_k: Number of top features to return

        Returns:
            Dictionary with top positive and negative features
        """
        try:
            import shap
        except ImportError:
            print("❌ SHAP not installed")
            return None

        # Compute SHAP values
        shap_data = self.compute_shap_values([text])

        if shap_data is None:
            return None

        # Get predicted class if not specified
        if class_idx is None:
            probs = self._predict_proba([text])[0]
            class_idx = np.argmax(probs)

        # Get SHAP values for this class
        shap_vals = shap_data['shap_values'].values[0, :, class_idx]

        # Tokenize to get words
        tokens = text.split()  # Simple tokenization

        # Match SHAP values to tokens (approximate)
        if len(shap_vals) != len(tokens):
            # Pad or truncate
            if len(shap_vals) > len(tokens):
                shap_vals = shap_vals[:len(tokens)]
            else:
                tokens = tokens[:len(shap_vals)]

        # Get top positive and negative
        sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
        top_indices = sorted_indices[:top_k]

        top_features = []
        for idx in top_indices:
            top_features.append({
                'token': tokens[idx] if idx < len(tokens) else 'UNK',
                'shap_value': float(shap_vals[idx]),
                'impact': 'positive' if shap_vals[idx] > 0 else 'negative'
            })

        return {
            'text': text,
            'class': CLASS_NAMES[class_idx],
            'top_features': top_features
        }


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
