# WildJailbreakLoader Module
#----------------------------
# Loads and samples from the WildJailbreak dataset for jailbreak detection supplementation.
# Dataset: https://huggingface.co/datasets/allenai/wildjailbreak
# All imports are in 01-Imports.py
###############################################################################


class WildJailbreakLoader:
    """
    Loads successful jailbreak samples from AllenAI's WildJailbreak dataset.

    Purpose: Supplement training data when modern LLMs successfully defend against
    all jailbreak attempts in our pipeline, resulting in insufficient positive samples.

    Dataset: WildJailbreak contains 262K safety-training prompt-response pairs
    - adversarial_harmful: 82,728 successful jailbreak examples
    - adversarial_benign: 78,706 benign adversarial queries
    - vanilla_harmful: 50,050 direct harmful requests
    - vanilla_benign: 50,050 harmless prompts

    Citation:
        WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models
        Liwei Jiang et al., NeurIPS 2024
        https://arxiv.org/abs/2406.18510
    """

    def __init__(self, random_seed: int = None):
        """
        Initialize WildJailbreak loader.

        Args:
            random_seed: Random seed for reproducible sampling (default: from WILDJAILBREAK_CONFIG)
        """
        # Use config value if not provided - NO HARDCODED VALUES
        if random_seed is None:
            random_seed = WILDJAILBREAK_CONFIG['random_seed']

        self.random_seed = random_seed
        self.dataset = None
        self.filtered_data = None

    def load_dataset(self) -> bool:
        """
        Load WildJailbreak dataset from HuggingFace.

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n{'='*60}")
            print(f"📦 LOADING WILDJAILBREAK DATASET")
            print(f"{'='*60}")
            print(f"  Source: {WILDJAILBREAK_CONFIG['dataset_name']}")
            print(f"  Split: {WILDJAILBREAK_CONFIG['dataset_split']}")
            print(f"  Loading...")

            # Import datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                print(f"\n❌ ERROR: 'datasets' library not installed")
                print(f"   Install with: pip install datasets")
                return False

            # Load dataset from config (no hardcoded values)
            self.dataset = load_dataset(
                WILDJAILBREAK_CONFIG['dataset_name'],
                split=WILDJAILBREAK_CONFIG['dataset_split']
            )

            print(f"  ✓ Loaded {len(self.dataset):,} total samples")
            print(f"{'='*60}\n")

            return True

        except Exception as e:
            print(f"\n❌ ERROR loading WildJailbreak dataset: {e}")
            print(f"   Ensure you have internet connection and 'datasets' library installed")
            return False

    def filter_successful_jailbreaks(self) -> pd.DataFrame:
        """
        Filter for successful jailbreak samples (adversarial_harmful).

        Returns:
            DataFrame with filtered samples
        """
        if self.dataset is None:
            print(f"❌ ERROR: Dataset not loaded. Call load_dataset() first.")
            return pd.DataFrame()

        try:
            print(f"🔍 Filtering for successful jailbreaks...")

            # Convert to pandas for easier filtering
            df = self.dataset.to_pandas()

            # Filter for specified data type from config (no hardcoded values)
            # These are adversarial jailbreaks that successfully conveyed harmful requests
            data_type = WILDJAILBREAK_CONFIG['data_type_filter']
            self.filtered_data = df[df['data_type'] == data_type].copy()

            print(f"  ✓ Found {len(self.filtered_data):,} successful jailbreak samples")
            print(f"  Type: {data_type} (successful jailbreak attempts)")

            # Print available fields for diversity sampling
            print(f"\n  Available fields:")
            for col in self.filtered_data.columns:
                print(f"    - {col}")

            return self.filtered_data

        except Exception as e:
            print(f"❌ ERROR filtering dataset: {e}")
            return pd.DataFrame()

    def sample_diverse(self, n_samples: int, diversity_fields: List[str] = None) -> pd.DataFrame:
        """
        Sample diverse jailbreak examples using stratified sampling.

        Args:
            n_samples: Number of samples to select
            diversity_fields: Fields to use for stratification (e.g., ['tactics'])
                            If None, uses ['tactics'] by default

        Returns:
            DataFrame with sampled examples in standardized format
        """
        if self.filtered_data is None or len(self.filtered_data) == 0:
            print(f"❌ ERROR: No filtered data available. Run filter_successful_jailbreaks() first.")
            return pd.DataFrame()

        try:
            print(f"\n{'='*60}")
            print(f"🎲 SAMPLING {n_samples:,} DIVERSE JAILBREAK EXAMPLES")
            print(f"{'='*60}")

            # Set random seed for reproducibility
            np.random.seed(self.random_seed)

            # Default diversity fields
            if diversity_fields is None:
                # Try to use tactics if available, otherwise random sample
                diversity_fields = ['tactics'] if 'tactics' in self.filtered_data.columns else []

            # If we need more samples than available, take all
            if n_samples >= len(self.filtered_data):
                print(f"  ⚠️  Requested {n_samples:,} samples, but only {len(self.filtered_data):,} available")
                print(f"  Using all available samples")
                sampled = self.filtered_data.copy()

            # Stratified sampling for diversity
            elif diversity_fields and all(f in self.filtered_data.columns for f in diversity_fields):
                print(f"  Stratifying by: {', '.join(diversity_fields)}")

                # For list fields like 'tactics', we'll sample proportionally across unique tactics
                if 'tactics' in diversity_fields and 'tactics' in self.filtered_data.columns:
                    # Explode tactics list to get individual tactics
                    tactics_expanded = self.filtered_data.explode('tactics')
                    tactic_counts = tactics_expanded['tactics'].value_counts()

                    print(f"  Found {len(tactic_counts)} unique jailbreak tactics")
                    print(f"  Top 5 tactics:")
                    for tactic, count in tactic_counts.head(5).items():
                        print(f"    - {tactic}: {count:,} samples")

                    # Sample proportionally from each tactic
                    sampled_indices = []
                    samples_per_tactic = max(1, n_samples // len(tactic_counts))

                    for tactic in tactic_counts.index:
                        # Get all samples with this tactic
                        tactic_mask = self.filtered_data['tactics'].apply(
                            lambda x: tactic in x if isinstance(x, list) else False
                        )
                        tactic_indices = self.filtered_data[tactic_mask].index.tolist()

                        # Sample from this tactic
                        n_to_sample = min(samples_per_tactic, len(tactic_indices))
                        if n_to_sample > 0:
                            sampled_indices.extend(
                                np.random.choice(tactic_indices, n_to_sample, replace=False)
                            )

                    # If we still need more samples, randomly add from remaining
                    if len(sampled_indices) < n_samples:
                        remaining_indices = list(set(self.filtered_data.index) - set(sampled_indices))
                        additional_needed = n_samples - len(sampled_indices)
                        if len(remaining_indices) > 0:
                            sampled_indices.extend(
                                np.random.choice(
                                    remaining_indices,
                                    min(additional_needed, len(remaining_indices)),
                                    replace=False
                                )
                            )

                    sampled = self.filtered_data.loc[sampled_indices[:n_samples]].copy()
                else:
                    # Fallback to random sampling
                    sampled = self.filtered_data.sample(n=n_samples, random_state=self.random_seed)

            # Random sampling if no diversity fields available
            else:
                print(f"  Using random sampling (no diversity fields available)")
                sampled = self.filtered_data.sample(n=n_samples, random_state=self.random_seed)

            print(f"  ✓ Sampled {len(sampled):,} examples")
            print(f"{'='*60}\n")

            # Convert to standardized format
            standardized = self._standardize_format(sampled)

            return standardized

        except Exception as e:
            print(f"❌ ERROR sampling data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _standardize_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert WildJailbreak format to our pipeline's standardized format.

        Expected pipeline format:
        - prompt: The jailbreak prompt text
        - response: The model response
        - expected_label: 'jailbreaks' (for jailbreak prompts)
        - jailbreak_label: 1 (Jailbreak Succeeded)
        - data_source: 'wildjailbreak'
        - model: WILDJAILBREAK_CONFIG['synthetic_model_label'] (for filtering from model analysis)

        Args:
            df: DataFrame in WildJailbreak format

        Returns:
            DataFrame in standardized format
        """
        try:
            standardized = pd.DataFrame()

            # Map fields (adjust based on actual WildJailbreak schema)
            # WildJailbreak likely has: 'prompt', 'completion', 'data_type', 'tactics'
            if 'prompt' in df.columns:
                standardized['prompt'] = df['prompt']
            elif 'vanilla_prompt' in df.columns:
                standardized['prompt'] = df['vanilla_prompt']
            else:
                print(f"⚠️  WARNING: 'prompt' field not found, using index as placeholder")
                standardized['prompt'] = [f"WildJailbreak sample {i}" for i in range(len(df))]

            # Response field
            if 'completion' in df.columns:
                standardized['response'] = df['completion']
            elif 'response' in df.columns:
                standardized['response'] = df['response']
            else:
                print(f"⚠️  WARNING: 'response'/'completion' field not found, using empty string")
                standardized['response'] = ""

            # Expected label (all are jailbreak prompts)
            standardized['expected_label'] = 'jailbreaks'

            # Jailbreak label (all are successful = 1)
            standardized['jailbreak_label'] = 1

            # Data source tracking
            standardized['data_source'] = 'wildjailbreak'

            # Model attribution (NEW - V09)
            # Mark as synthetic to enable filtering from per-model vulnerability analysis
            standardized['model'] = WILDJAILBREAK_CONFIG['synthetic_model_label']

            # Optional: preserve tactics for analysis
            if 'tactics' in df.columns:
                standardized['wildjailbreak_tactics'] = df['tactics']

            print(f"  ✓ Standardized {len(standardized)} samples to pipeline format")

            return standardized

        except Exception as e:
            print(f"❌ ERROR standardizing format: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def load_and_sample(self, n_samples: int) -> pd.DataFrame:
        """
        Convenience method: Load dataset, filter, and sample in one call.

        Args:
            n_samples: Number of samples to retrieve

        Returns:
            DataFrame with sampled examples in standardized format
        """
        # Load dataset
        if not self.load_dataset():
            return pd.DataFrame()

        # Filter for successful jailbreaks
        filtered = self.filter_successful_jailbreaks()
        if len(filtered) == 0:
            return pd.DataFrame()

        # Sample diversely
        sampled = self.sample_diverse(n_samples)

        return sampled


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
