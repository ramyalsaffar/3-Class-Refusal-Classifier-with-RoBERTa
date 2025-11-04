# DataCleaner Module
#-------------------
# Comprehensive data quality management for text classification datasets.
# Detects and cleans duplicates, outliers, invalid data, and label inconsistencies.
# Adapted from OutlierHandler principles for NLP text data.
# All imports are in 01-Imports.py
###############################################################################


class DataCleaner:
    """
    Comprehensive data cleaning and quality validation for refusal classification.

    Handles:
    - Duplicate detection (exact and near-duplicates)
    - Empty/null value validation
    - Length outliers (too short/long responses)
    - Invalid character detection
    - Label consistency checks
    - Response quality indicators
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize DataCleaner.

        Args:
            verbose: Print detailed cleaning reports
        """
        self.verbose = verbose
        self.cleaning_log = []

        # Quality thresholds (from config)
        self.min_response_length = DATA_CLEANING_CONFIG['min_response_length']
        self.max_response_length = DATA_CLEANING_CONFIG['max_response_length']
        self.min_prompt_length = DATA_CLEANING_CONFIG['min_prompt_length']
        self.max_prompt_length = DATA_CLEANING_CONFIG['max_prompt_length']
        self.similarity_threshold = DATA_CLEANING_CONFIG['similarity_threshold']

        # Error indicators
        self.error_patterns = [
            '[ERROR',
            'Failed to generate',
            '<!DOCTYPE html>',
            '<html>',
            'An error occurred',
            'Rate limit',
            'API error'
        ]

    def clean_dataset(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Main cleaning pipeline - runs all cleaning steps.

        Args:
            df: DataFrame with columns: prompt, response, model, refusal_label, jailbreak_label
            strategy: Cleaning strategy ('auto', 'conservative', 'aggressive', 'none')

        Returns:
            Cleaned DataFrame with quality report
        """
        if self.verbose:
            print("\n" + "="*60)
            print("DATA CLEANING PIPELINE")
            print("="*60)
            print(f"Strategy: {strategy}")
            print(f"Initial dataset size: {len(df)} samples")

        original_size = len(df)
        df_clean = df.copy()

        # Step 1: Validate basic data integrity
        df_clean = self._validate_data_integrity(df_clean)

        # Step 2: Remove exact duplicates
        df_clean = self._remove_duplicates(df_clean, strategy)

        # Step 3: Filter length outliers
        df_clean = self._filter_length_outliers(df_clean, strategy)

        # Step 4: Detect and handle error responses
        df_clean = self._handle_error_responses(df_clean)

        # Step 5: Validate label consistency
        df_clean = self._validate_label_consistency(df_clean, strategy)

        # Step 6: Quality assessment
        quality_report = self._assess_quality(df_clean, original_size)

        if self.verbose:
            self._print_cleaning_report(quality_report)

        return df_clean

    def _validate_data_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for null values and required columns.

        WHY: DataCleaner is called both BEFORE and AFTER labeling
        - Before labeling: ['prompt', 'response', 'model', 'expected_label', 'timestamp']
        - After labeling: ['prompt', 'response', 'model', 'refusal_label', 'jailbreak_label', ...]
        """
        # Base columns that must ALWAYS exist (before and after labeling)
        required_cols = ['prompt', 'response', 'model']

        # Optional columns (add if they exist)
        optional_cols = ['refusal_label', 'jailbreak_label', 'expected_label']
        for col in optional_cols:
            if col in df.columns:
                required_cols.append(col)

        # Check required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Count nulls
        null_counts = df[required_cols].isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            if self.verbose:
                print(f"\n⚠️  Found {total_nulls} null values:")
                for col, count in null_counts[null_counts > 0].items():
                    print(f"   {col}: {count} nulls")

            # Remove rows with nulls in critical columns
            before = len(df)
            df = df.dropna(subset=required_cols)
            removed = before - len(df)

            if removed > 0:
                self.cleaning_log.append({
                    'step': 'null_removal',
                    'removed': removed,
                    'reason': 'Null values in required columns'
                })
                if self.verbose:
                    print(f"   ✓ Removed {removed} rows with null values")

        return df

    def _remove_duplicates(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Remove exact and near-duplicate entries."""
        before = len(df)

        # Exact duplicates (prompt + response)
        df = df.drop_duplicates(subset=['prompt', 'response'], keep='first')
        exact_removed = before - len(df)

        if exact_removed > 0:
            self.cleaning_log.append({
                'step': 'exact_duplicates',
                'removed': exact_removed,
                'reason': 'Identical prompt-response pairs'
            })
            if self.verbose:
                print(f"\n✓ Removed {exact_removed} exact duplicates")

        # Near-duplicates (same prompt, very similar response)
        if strategy in ['aggressive', 'auto']:
            before_near = len(df)
            df = self._remove_near_duplicates(df)
            near_removed = before_near - len(df)

            if near_removed > 0:
                self.cleaning_log.append({
                    'step': 'near_duplicates',
                    'removed': near_removed,
                    'reason': 'Very similar responses to same prompt'
                })
                if self.verbose:
                    print(f"✓ Removed {near_removed} near-duplicates")

        return df

    def _remove_near_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove responses that are too similar for the same prompt."""
        # Group by prompt
        drop_indices = []

        for prompt, group in df.groupby('prompt'):
            if len(group) <= 1:
                continue

            responses = group['response'].tolist()
            indices = group.index.tolist()

            # Check pairwise similarity (simple approach)
            for i in range(len(responses)):
                if indices[i] in drop_indices:
                    continue
                for j in range(i + 1, len(responses)):
                    if indices[j] in drop_indices:
                        continue

                    # Calculate similarity (Jaccard on words)
                    words_i = set(responses[i].lower().split())
                    words_j = set(responses[j].lower().split())

                    if len(words_i | words_j) == 0:
                        continue

                    jaccard = len(words_i & words_j) / len(words_i | words_j)

                    # If very similar, keep first occurrence
                    if jaccard > self.similarity_threshold:
                        drop_indices.append(indices[j])

        if drop_indices:
            df = df.drop(index=drop_indices)

        return df

    def _filter_length_outliers(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Filter responses/prompts that are too short or too long."""
        before = len(df)

        # Response length filtering
        df['response_len'] = df['response'].str.len()
        df['prompt_len'] = df['prompt'].str.len()

        # Calculate statistics
        response_stats = df['response_len'].describe()
        prompt_stats = df['prompt_len'].describe()

        if self.verbose:
            print(f"\n📊 Length Statistics:")
            print(f"   Response: mean={response_stats['mean']:.0f}, "
                  f"median={response_stats['50%']:.0f}, "
                  f"std={response_stats['std']:.0f}")
            print(f"   Prompt: mean={prompt_stats['mean']:.0f}, "
                  f"median={prompt_stats['50%']:.0f}")

        # Filter based on strategy
        if strategy == 'conservative':
            # Only remove extreme outliers (outside whiskers)
            q1_resp = response_stats['25%']
            q3_resp = response_stats['75%']
            iqr_resp = q3_resp - q1_resp
            lower_bound = max(self.min_response_length, q1_resp - 3 * iqr_resp)
            upper_bound = min(self.max_response_length, q3_resp + 3 * iqr_resp)
        elif strategy == 'aggressive':
            # Tighter bounds (1.5 IQR)
            q1_resp = response_stats['25%']
            q3_resp = response_stats['75%']
            iqr_resp = q3_resp - q1_resp
            lower_bound = max(self.min_response_length, q1_resp - 1.5 * iqr_resp)
            upper_bound = min(self.max_response_length, q3_resp + 1.5 * iqr_resp)
        else:  # auto
            # Use absolute thresholds
            lower_bound = self.min_response_length
            upper_bound = self.max_response_length

        # Apply filters
        df = df[
            (df['response_len'] >= lower_bound) &
            (df['response_len'] <= upper_bound) &
            (df['prompt_len'] >= self.min_prompt_length) &
            (df['prompt_len'] <= self.max_prompt_length)
        ]

        # Clean up temporary columns
        df = df.drop(columns=['response_len', 'prompt_len'])

        removed = before - len(df)
        if removed > 0:
            self.cleaning_log.append({
                'step': 'length_outliers',
                'removed': removed,
                'reason': f'Length outside bounds ({lower_bound:.0f}-{upper_bound:.0f})'
            })
            if self.verbose:
                print(f"   ✓ Removed {removed} length outliers")

        return df

    def _handle_error_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and remove error responses from API failures."""
        before = len(df)

        # Check for error patterns
        error_mask = df['response'].str.contains('|'.join(self.error_patterns), case=False, na=False)
        error_count = error_mask.sum()

        if error_count > 0:
            if self.verbose:
                print(f"\n⚠️  Found {error_count} error responses")
                # Show examples
                error_samples = df[error_mask]['response'].head(3)
                for i, sample in enumerate(error_samples, 1):
                    print(f"   Example {i}: {sample[:100]}...")

            df = df[~error_mask]

            self.cleaning_log.append({
                'step': 'error_responses',
                'removed': error_count,
                'reason': 'API error or failed generation'
            })
            if self.verbose:
                print(f"   ✓ Removed {error_count} error responses")

        return df

    def _validate_label_consistency(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        Check for label inconsistencies and suspicious patterns.

        WHY: This method is only relevant AFTER labeling
        Skip validation if 'refusal_label' doesn't exist yet
        """
        # Skip if labels don't exist yet (pre-labeling cleaning)
        if 'refusal_label' not in df.columns:
            if self.verbose:
                print(f"\n🔍 Label Consistency Check: Skipped (no labels yet - pre-labeling cleaning)")
            return df

        if self.verbose:
            print(f"\n🔍 Label Consistency Check:")

        # Check for invalid refusal labels (GENERIC: works with any N-class classifier)
        valid_refusal_labels = list(range(len(CLASS_NAMES)))
        invalid_refusal = ~df['refusal_label'].isin(valid_refusal_labels)
        invalid_count = invalid_refusal.sum()

        if invalid_count > 0:
            if self.verbose:
                print(f"   ⚠️  Found {invalid_count} invalid refusal labels")
            df = df[~invalid_refusal]
            self.cleaning_log.append({
                'step': 'invalid_labels',
                'removed': invalid_count,
                'reason': 'Invalid refusal label values'
            })

        # Check jailbreak labels if present (GENERIC: works with any N-class classifier)
        if 'jailbreak_label' in df.columns:
            valid_jailbreak_labels = list(range(len(JAILBREAK_CLASS_NAMES)))
            invalid_jailbreak = ~df['jailbreak_label'].isin(valid_jailbreak_labels)
            invalid_jb_count = invalid_jailbreak.sum()

            if invalid_jb_count > 0:
                if self.verbose:
                    print(f"   ⚠️  Found {invalid_jb_count} invalid jailbreak labels")
                df = df[~invalid_jailbreak]
                self.cleaning_log.append({
                    'step': 'invalid_jailbreak_labels',
                    'removed': invalid_jb_count,
                    'reason': 'Invalid jailbreak label values'
                })

        # Check label distribution
        label_dist = df['refusal_label'].value_counts()
        if self.verbose:
            print(f"   Label distribution after cleaning:")
            for label, count in label_dist.items():
                pct = 100 * count / len(df)
                label_name = CLASS_NAMES[label]
                print(f"      {label_name}: {count} ({pct:.1f}%)")

        # Warn about severe imbalance (if one class < 5%)
        min_class_pct = (label_dist.min() / len(df)) * 100
        if min_class_pct < 5:
            print(f"   ⚠️  WARNING: Severe class imbalance detected (min class: {min_class_pct:.1f}%)")
            print(f"      Consider collecting more data for underrepresented classes")

        return df

    def _assess_quality(self, df: pd.DataFrame, original_size: int) -> Dict:
        """Assess overall data quality after cleaning."""
        cleaned_size = len(df)
        removed_total = original_size - cleaned_size
        removal_pct = (removed_total / original_size) * 100 if original_size > 0 else 0

        # Quality score calculation
        if removal_pct < 2:
            quality = "Excellent"
        elif removal_pct < 5:
            quality = "Good"
        elif removal_pct < 10:
            quality = "Acceptable"
        else:
            quality = "Concerning"

        return {
            'original_size': original_size,
            'cleaned_size': cleaned_size,
            'removed_total': removed_total,
            'removal_percentage': removal_pct,
            'quality_rating': quality,
            'cleaning_log': self.cleaning_log
        }

    def _print_cleaning_report(self, report: Dict):
        """Print comprehensive cleaning report."""
        print("\n" + "="*60)
        print("DATA CLEANING REPORT")
        print("="*60)
        print(f"Original dataset: {report['original_size']} samples")
        print(f"Cleaned dataset:  {report['cleaned_size']} samples")
        print(f"Removed:          {report['removed_total']} samples ({report['removal_percentage']:.2f}%)")
        print(f"Quality Rating:   {report['quality_rating']}")

        if report['cleaning_log']:
            print(f"\nCleaning Steps:")
            for entry in report['cleaning_log']:
                print(f"   • {entry['step']}: {entry['removed']} samples")
                print(f"     Reason: {entry['reason']}")

        print("="*60)

    def get_outlier_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate outlier analysis report without modifying data.
        Similar to OutlierHandler.get_outlier_report().

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with outlier statistics and recommendations
        """
        report = {
            'total_samples': len(df),
            'issues_found': []
        }

        # Check for nulls
        # WHY: refusal_label may not exist yet if called before labeling
        null_check_cols = ['prompt', 'response']
        if 'refusal_label' in df.columns:
            null_check_cols.append('refusal_label')

        null_counts = df[null_check_cols].isnull().sum()
        if null_counts.sum() > 0:
            report['issues_found'].append({
                'type': 'null_values',
                'count': int(null_counts.sum()),
                'percentage': (null_counts.sum() / len(df)) * 100
            })

        # Check for duplicates
        duplicate_count = df.duplicated(subset=['prompt', 'response']).sum()
        if duplicate_count > 0:
            report['issues_found'].append({
                'type': 'exact_duplicates',
                'count': int(duplicate_count),
                'percentage': (duplicate_count / len(df)) * 100
            })

        # Check length outliers
        df_temp = df.copy()
        df_temp['response_len'] = df_temp['response'].str.len()
        q1 = df_temp['response_len'].quantile(0.25)
        q3 = df_temp['response_len'].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df_temp['response_len'] < q1 - 3 * iqr) |
                   (df_temp['response_len'] > q3 + 3 * iqr)).sum()

        if outliers > 0:
            report['issues_found'].append({
                'type': 'length_outliers',
                'count': int(outliers),
                'percentage': (outliers / len(df)) * 100
            })

        # Recommendation based on total issues
        total_issues = sum(issue['count'] for issue in report['issues_found'])
        issue_pct = (total_issues / len(df)) * 100 if len(df) > 0 else 0

        if issue_pct < 2.5:
            report['recommendation'] = 'remove'
            report['reasoning'] = 'Very few issues - safe to remove'
        elif issue_pct < 10:
            report['recommendation'] = 'conservative'
            report['reasoning'] = 'Moderate issues - use conservative cleaning'
        else:
            report['recommendation'] = 'aggressive'
            report['reasoning'] = 'Many issues detected - aggressive cleaning recommended'

        return report


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
