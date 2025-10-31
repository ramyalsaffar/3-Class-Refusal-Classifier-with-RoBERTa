# Adversarial Testing Module
#----------------------------
# Test classifier robustness via paraphrasing.
# All imports are in 00-Imports.py
###############################################################################


class AdversarialTester:
    """Test classifier robustness via paraphrasing."""

    def __init__(self, model, tokenizer, device, openai_key: str):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.openai_client = OpenAI(api_key=openai_key)
        self.paraphrase_model = API_CONFIG['paraphrase_model']
        self.temperature = API_CONFIG['temperature_paraphrase']
        self.max_tokens = API_CONFIG['max_tokens_paraphrase']
        self.dimensions = ANALYSIS_CONFIG['paraphrase_dimensions']

    def test_robustness(self, test_df: pd.DataFrame, num_samples: int = None) -> Dict:
        """
        Test robustness on paraphrased inputs.

        Args:
            test_df: Test dataframe
            num_samples: Number of samples to paraphrase (uses config if None)

        Returns:
            Dictionary with robustness metrics
        """
        num_samples = num_samples or ANALYSIS_CONFIG['adversarial_samples']

        print("\n" + "="*50)
        print("ADVERSARIAL ROBUSTNESS TESTING")
        print("="*50)

        # Sample from test set
        sample_df = test_df.sample(n=min(num_samples, len(test_df)), random_state=EXPERIMENT_CONFIG['random_seed'])

        # Evaluate on original
        print("\nEvaluating on original samples...")
        original_f1 = self._evaluate_samples(sample_df)

        results = {
            'original_f1': float(original_f1),
            'paraphrased_f1': {},
            'samples_tested': len(sample_df)
        }

        # Test each dimension
        for dimension in self.dimensions:
            print(f"\nTesting {dimension} paraphrases...")
            paraphrased_f1 = self._test_dimension(sample_df, dimension)
            results['paraphrased_f1'][dimension] = float(paraphrased_f1)

        # Calculate summary
        avg_paraphrased = np.mean(list(results['paraphrased_f1'].values()))
        results['avg_paraphrased_f1'] = float(avg_paraphrased)
        results['f1_drop'] = float(original_f1 - avg_paraphrased)
        results['relative_drop_pct'] = float((original_f1 - avg_paraphrased) / original_f1 * 100)

        print(f"\n{'='*50}")
        print(f"Original F1: {original_f1:.3f}")
        print(f"Avg Paraphrased F1: {avg_paraphrased:.3f}")
        print(f"F1 Drop: {results['f1_drop']:.3f} ({results['relative_drop_pct']:.1f}%)")
        print("\nPer-dimension F1:")
        for dim, f1 in results['paraphrased_f1'].items():
            print(f"  {dim}: {f1:.3f}")

        return results

    def _test_dimension(self, df: pd.DataFrame, dimension: str) -> float:
        """Test one paraphrasing dimension."""
        paraphrased_texts = []

        for text in tqdm(df['response'].tolist(), desc=f"Paraphrasing ({dimension})"):
            try:
                paraphrase = self._generate_paraphrase(text, dimension)
                if self._sanity_check(text, paraphrase):
                    paraphrased_texts.append(paraphrase)
                else:
                    paraphrased_texts.append(text)  # Fallback to original
                time.sleep(API_CONFIG['rate_limit_delay'])
            except Exception as e:
                print(f"Error paraphrasing: {e}")
                paraphrased_texts.append(text)

        # Create paraphrased dataframe
        paraphrased_df = df.copy()
        paraphrased_df['response'] = paraphrased_texts

        # Evaluate
        return self._evaluate_samples(paraphrased_df)

    def _generate_paraphrase(self, text: str, dimension: str) -> str:
        """Generate paraphrase using GPT-4."""
        templates = {
            'synonym': f"Rewrite using synonyms, keep meaning identical:\n{text}",
            'restructure': f"Restructure sentences, keep meaning identical:\n{text}",
            'formality': f"Change formality level, keep meaning identical:\n{text}",
            'compression': f"Make more concise, keep key meaning:\n{text}"
        }

        response = self.openai_client.chat.completions.create(
            model=self.paraphrase_model,
            messages=[
                {"role": "system", "content": "You paraphrase text while preserving exact meaning."},
                {"role": "user", "content": templates[dimension]}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content.strip()

    def _sanity_check(self, original: str, paraphrase: str) -> bool:
        """Check if paraphrase is valid."""
        refusal_keywords = ['cannot', "can't", 'unable', 'not able', "won't", 'will not', 'refuse']

        original_has_refusal = any(kw in original.lower() for kw in refusal_keywords)
        paraphrase_has_refusal = any(kw in paraphrase.lower() for kw in refusal_keywords)

        # Both must have or not have refusal keywords
        if original_has_refusal != paraphrase_has_refusal:
            return False

        # Length check
        length_ratio = len(paraphrase) / len(original)
        if not (0.3 < length_ratio < 3.0):
            return False

        return True

    def _evaluate_samples(self, df: pd.DataFrame) -> float:
        """Evaluate F1 on samples."""
        dataset = ClassificationDataset(
            df['response'].tolist(),
            df['label'].tolist(),
            self.tokenizer
        )
        loader = DataLoader(dataset, batch_size=API_CONFIG['inference_batch_size'], shuffle=False)

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

        return f1_score(all_labels, all_preds, average='macro')

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved adversarial results to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
