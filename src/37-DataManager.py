# DataManager Module
#-------------------
# Smart data retention and PostgreSQL management for production monitoring.
# Implements intelligent sampling and retention strategies.
# NOTE: This is a standalone production script with additional imports.
# Requires: pip install psycopg2-binary (imports psycopg2 on line 27)
###############################################################################

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataManager:
    """Manage production data with smart retention strategies."""

    def __init__(self, db_config: Dict = None):
        """
        Initialize data manager with PostgreSQL connection.

        Args:
            db_config: Database configuration (uses PRODUCTION_CONFIG if None)
        """
        self.db_config = db_config or PRODUCTION_CONFIG['database']
        self.conn = None
        self.connect()
        self._create_tables()

    def connect(self):
        """Establish PostgreSQL connection."""
        try:
            import psycopg2
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            print(f"✓ Connected to PostgreSQL: {self.db_config['database']}")
        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")
            print("   Make sure PostgreSQL is running and credentials are correct")
            raise

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        try:
            # Predictions log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    confidence FLOAT NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    latency_ms FLOAT,
                    judge_label INTEGER,
                    is_monitored BOOLEAN DEFAULT FALSE,
                    metadata JSONB
                );
            """)

            # Create indices for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_model_version ON predictions_log(model_version);
                CREATE INDEX IF NOT EXISTS idx_is_monitored ON predictions_log(is_monitored);
            """)

            # Monitoring runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_runs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    sample_size INTEGER NOT NULL,
                    disagreement_rate FLOAT NOT NULL,
                    action_taken VARCHAR(50) NOT NULL,
                    metrics JSONB,
                    notes TEXT
                );
            """)

            # Model versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version VARCHAR(50) PRIMARY KEY,
                    deployed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    f1_score_validation FLOAT NOT NULL,
                    is_active BOOLEAN DEFAULT FALSE,
                    is_challenger BOOLEAN DEFAULT FALSE,
                    traffic_percentage FLOAT DEFAULT 0.0,
                    rollback_available BOOLEAN DEFAULT TRUE,
                    metadata JSONB
                );
            """)

            self.conn.commit()
            print("✓ Database tables verified")
        finally:
            cursor.close()

    def log_prediction(self, prompt: str, response: str, prediction: int,
                      confidence: float, model_version: str, latency_ms: float = None,
                      metadata: Dict = None):
        """
        Log a prediction to database.

        Args:
            prompt: User prompt
            response: LLM response
            prediction: Model prediction (0/1/2)
            confidence: Prediction confidence
            model_version: Model version string
            latency_ms: Inference latency in milliseconds
            metadata: Additional metadata (JSON)
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO predictions_log
                (prompt, response, prediction, confidence, model_version, latency_ms, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (prompt, response, prediction, confidence, model_version, latency_ms,
                  json.dumps(metadata) if metadata else None))

            self.conn.commit()
        finally:
            cursor.close()

    def get_recent_predictions(self, hours: int = None, limit: int = None,
                              model_version: str = None) -> pd.DataFrame:
        """
        Get recent predictions from database.

        Args:
            hours: Number of hours to look back (default: from PRODUCTION_CONFIG)
            limit: Maximum number of samples to return
            model_version: Filter by specific model version

        Returns:
            DataFrame with predictions
        """
        # Use config value if not provided - NO HARDCODING!
        if hours is None:
            hours = PRODUCTION_CONFIG['monitoring_thresholds']['check_interval_hours']
        
        cursor = self.conn.cursor()
        try:
            query = """
                SELECT id, timestamp, prompt, response, prediction, confidence,
                       model_version, judge_label, is_monitored
                FROM predictions_log
                WHERE timestamp > NOW() - INTERVAL '%s hours'
            """
            params = [hours]

            if model_version:
                query += " AND model_version = %s"
                params.append(model_version)

            query += " ORDER BY timestamp DESC"

            if limit:
                query += " LIMIT %s"
                params.append(limit)

            cursor.execute(query, params)

            columns = ['id', 'timestamp', 'prompt', 'response', 'prediction',
                      'confidence', 'model_version', 'judge_label', 'is_monitored']
            df = pd.DataFrame(cursor.fetchall(), columns=columns)

            return df
        finally:
            cursor.close()

    def sample_for_monitoring(self, sample_size: int = None,
                            hours: int = None) -> pd.DataFrame:
        """
        GENERIC: Sample predictions for monitoring (stratified by prediction class).
        Works with any number of classes.

        Args:
            sample_size: Number of samples to return (default: from PRODUCTION_CONFIG)
            hours: Look back period (default: from PRODUCTION_CONFIG)

        Returns:
            DataFrame with sampled predictions
        """
        # Use config values if not provided - NO HARDCODING!
        if sample_size is None:
            sample_size = PRODUCTION_CONFIG['monitoring_thresholds']['small_sample_size']
        if hours is None:
            hours = PRODUCTION_CONFIG['monitoring_thresholds']['check_interval_hours']
        
        cursor = self.conn.cursor()
        try:
            # GENERIC: Get stratified sample (balanced across all classes)
            num_classes = len(CLASS_NAMES)
            samples_per_class = sample_size // num_classes
            logger.info(f"Sampling {samples_per_class} per class ({num_classes} classes)")

            # Build dynamic query for all classes
            query_parts = []
            params = []
            for class_id in range(num_classes):
                query_parts.append("""
                    (SELECT * FROM predictions_log
                     WHERE timestamp > NOW() - INTERVAL '%s hours'
                     AND prediction = %s
                     AND is_monitored = FALSE
                     ORDER BY RANDOM()
                     LIMIT %s)
                """)
                params.extend([hours, class_id, samples_per_class])

            query = " UNION ALL ".join(query_parts)

            cursor.execute(query, params)

            columns = ['id', 'timestamp', 'prompt', 'response', 'prediction',
                      'confidence', 'model_version', 'latency_ms', 'judge_label',
                      'is_monitored', 'metadata']
            df = pd.DataFrame(cursor.fetchall(), columns=columns)

            return df
        finally:
            cursor.close()

    def update_with_judge_labels(self, sample_ids: List[int],
                                judge_labels: List[int]):
        """
        Update samples with LLM judge labels.

        Args:
            sample_ids: List of prediction IDs
            judge_labels: List of judge labels (refusal classification)
        """
        cursor = self.conn.cursor()
        try:
            for sample_id, label in tqdm(
                zip(sample_ids, judge_labels),
                total=len(sample_ids),
                desc="Updating judge labels"
            ):
                cursor.execute("""
                    UPDATE predictions_log
                    SET judge_label = %s, is_monitored = TRUE
                    WHERE id = %s
                """, (int(label), int(sample_id)))

            self.conn.commit()
        finally:
            cursor.close()

    def get_retraining_data(self) -> pd.DataFrame:
        """
        Get data for retraining using smart retention strategy.

        Strategy (configurable in PRODUCTION_CONFIG['retention']):
        - Recent: 100% of problematic samples + configured % of correct samples
        - Medium-term: Configured % stratified sample
        - Long-term: Configured % representative sample

        Returns:
            DataFrame ready for retraining
        """
        from scipy.stats import chi2_contingency
        
        # Get retention config - NO HARDCODING!
        retention = PRODUCTION_CONFIG['retention']
        recent_days = retention['recent_days']
        recent_correct_rate = retention['recent_correct_rate']
        medium_days = retention['medium_days']
        medium_rate = retention['medium_rate']
        longterm_days = retention['longterm_days']
        longterm_rate = retention['longterm_rate']
        high_conf_threshold = PRODUCTION_CONFIG['retraining']['high_confidence_threshold']
        
        # Use Utils print_banner for consistent formatting
        print_banner("COLLECTING RETRAINING DATA")

        cursor = self.conn.cursor()
        try:
            all_samples = []
            num_classes = len(CLASS_NAMES)

            # Recent: 0-recent_days
            print(f"\n1. Recent data (0-{recent_days} days):")
            print("   - 100% of problematic samples (disagreements)")
            cursor.execute(f"""
                SELECT prompt, response, judge_label as label
                FROM predictions_log
                WHERE timestamp > NOW() - INTERVAL '{recent_days} days'
                AND judge_label IS NOT NULL
                AND judge_label != prediction
            """)
            problematic = cursor.fetchall()
            all_samples.extend(problematic)
            print(f"   - Collected {len(problematic)} problematic samples")

            # Calculate denominator from rate (e.g., 0.2 → 1/5)
            recent_correct_denominator = int(1 / recent_correct_rate) if recent_correct_rate > 0 else 1
            print(f"   - {int(recent_correct_rate * 100)}% of correct high-confidence samples")
            cursor.execute(f"""
                SELECT prompt, response, judge_label as label
                FROM predictions_log
                WHERE timestamp > NOW() - INTERVAL '{recent_days} days'
                AND judge_label IS NOT NULL
                AND judge_label = prediction
                AND confidence > {high_conf_threshold}
                ORDER BY RANDOM()
                LIMIT (SELECT COUNT(*) FROM predictions_log
                       WHERE timestamp > NOW() - INTERVAL '{recent_days} days'
                       AND judge_label = prediction) / {recent_correct_denominator}
            """)
            recent_correct = cursor.fetchall()
            all_samples.extend(recent_correct)
            print(f"   - Collected {len(recent_correct)} correct samples")
            
            # Hypothesis test: Verify recent data maintains class distribution
            if len(all_samples) > 0:
                recent_df = pd.DataFrame(all_samples, columns=['prompt', 'response', 'label'])
                recent_counts = [sum(recent_df['label'] == i) for i in range(num_classes)]
                
                # Chi-square test: H0 = Recent data follows expected class distribution
                # We expect higher representation of problematic classes (those with disagreements)
                # This is intentional, so we test if distribution is NOT uniform (which would be bad)
                if sum(recent_counts) >= num_classes and min(recent_counts) > 0:
                    # Test against uniform distribution (null hypothesis)
                    expected_uniform = [len(recent_df) / num_classes] * num_classes
                    chi2_stat, p_value = chi2_contingency([recent_counts, expected_uniform])[:2]
                    
                    if p_value < 0.05:
                        logger.info(f"✓ Recent data distribution differs from uniform (p={p_value:.4f}) - Expected for problematic sampling")
                    else:
                        logger.warning(f"⚠️  Recent data looks too uniform (p={p_value:.4f}) - Check if problematic samples exist")


            # Medium-term: (recent_days+1)-medium_days (stratified sample at configured rate)
            medium_rate_denominator = int(1 / medium_rate) if medium_rate > 0 else 1
            print(f"\n2. Medium-term data ({recent_days+1}-{medium_days} days): {int(medium_rate * 100)}% stratified sample")
            medium_start_count = len(all_samples)
            for label in range(num_classes):
                cursor.execute(f"""
                    SELECT prompt, response, judge_label as label
                    FROM predictions_log
                    WHERE timestamp BETWEEN NOW() - INTERVAL '{medium_days} days' AND NOW() - INTERVAL '{recent_days} days'
                    AND judge_label = %s
                    ORDER BY RANDOM()
                    LIMIT (SELECT COUNT(*) FROM predictions_log
                           WHERE timestamp BETWEEN NOW() - INTERVAL '{medium_days} days' AND NOW() - INTERVAL '{recent_days} days'
                           AND judge_label = %s) / {medium_rate_denominator}
                """, (label, label))
                medium_samples = cursor.fetchall()
                all_samples.extend(medium_samples)
            medium_collected = len(all_samples) - medium_start_count
            print(f"   - Collected {medium_collected} samples")
            
            # Hypothesis test: Verify medium-term maintains stratified sampling
            if medium_collected > 0:
                medium_df = pd.DataFrame(all_samples[medium_start_count:], columns=['prompt', 'response', 'label'])
                medium_counts = [sum(medium_df['label'] == i) for i in range(num_classes)]
                
                # Chi-square test: H0 = Medium data follows stratified (balanced) distribution
                if sum(medium_counts) >= num_classes and min(medium_counts) > 0:
                    expected_balanced = [len(medium_df) / num_classes] * num_classes
                    chi2_stat, p_value = chi2_contingency([medium_counts, expected_balanced])[:2]
                    
                    if p_value >= 0.05:
                        logger.info(f"✓ Medium-term data properly stratified (p={p_value:.4f})")
                    else:
                        logger.warning(f"⚠️  Medium-term data imbalance detected (p={p_value:.4f})")


            # Long-term: (medium_days+1)-longterm_days (representative sample at configured rate)
            longterm_rate_denominator = int(1 / longterm_rate) if longterm_rate > 0 else 1
            print(f"\n3. Long-term data ({medium_days+1}-{longterm_days} days): {int(longterm_rate * 100)}% representative sample")
            longterm_start_count = len(all_samples)
            for label in range(num_classes):
                cursor.execute(f"""
                    SELECT prompt, response, judge_label as label
                    FROM predictions_log
                    WHERE timestamp BETWEEN NOW() - INTERVAL '{longterm_days} days' AND NOW() - INTERVAL '{medium_days} days'
                    AND judge_label = %s
                    ORDER BY RANDOM()
                    LIMIT (SELECT COUNT(*) FROM predictions_log
                           WHERE timestamp BETWEEN NOW() - INTERVAL '{longterm_days} days' AND NOW() - INTERVAL '{medium_days} days'
                           AND judge_label = %s) / {longterm_rate_denominator}
                """, (label, label))
                long_samples = cursor.fetchall()
                all_samples.extend(long_samples)
            
            longterm_collected = len(all_samples) - longterm_start_count
            print(f"   - Collected {longterm_collected} samples")
            
            # Hypothesis test: Verify long-term maintains representative sampling
            if longterm_collected > 0:
                longterm_df = pd.DataFrame(all_samples[longterm_start_count:], columns=['prompt', 'response', 'label'])
                longterm_counts = [sum(longterm_df['label'] == i) for i in range(num_classes)]
                
                # Chi-square test: H0 = Long-term data follows representative (balanced) distribution
                if sum(longterm_counts) >= num_classes and min(longterm_counts) > 0:
                    expected_balanced = [len(longterm_df) / num_classes] * num_classes
                    chi2_stat, p_value = chi2_contingency([longterm_counts, expected_balanced])[:2]
                    
                    if p_value >= 0.05:
                        logger.info(f"✓ Long-term data properly representative (p={p_value:.4f})")
                    else:
                        logger.warning(f"⚠️  Long-term data imbalance detected (p={p_value:.4f})")

            # Convert to DataFrame
            df = pd.DataFrame(all_samples, columns=['prompt', 'response', 'label'])

            # Summary with Utils print_banner
            print_banner("RETRAINING DATA SUMMARY")
            print(f"Total samples: {len(df)}")
            logger.info(f"Collected {len(df)} samples for retraining")
            
            # Use safe_divide from Utils for percentage calculations
            for i in range(num_classes):
                count = (df['label'] == i).sum()
                pct = safe_divide(count * 100, len(df), 0.0)
                print(f"  {CLASS_NAMES[i]}: {count} ({pct:.1f}%)")
            
            # Final hypothesis test: Overall class balance
            if len(df) > 0:
                overall_counts = [sum(df['label'] == i) for i in range(num_classes)]
                if sum(overall_counts) >= num_classes and min(overall_counts) > 0:
                    expected_balanced = [len(df) / num_classes] * num_classes
                    chi2_stat, p_value = chi2_contingency([overall_counts, expected_balanced])[:2]
                    
                    print(f"\n📊 Overall Distribution Test:")
                    print(f"   Chi-square statistic: {chi2_stat:.4f}")
                    print(f"   P-value: {p_value:.4f}")
                    if p_value < 0.001:
                        print(f"   ⚠️  Significant imbalance detected (p<0.001) - Consider adjusting retention rates")
                    elif p_value < 0.05:
                        print(f"   ℹ️  Moderate imbalance (p<0.05) - Expected due to problematic sample prioritization")
                    else:
                        print(f"   ✓ Well-balanced distribution (p≥0.05)")

            return df
        finally:
            cursor.close()

    def log_monitoring_run(self, sample_size: int, disagreement_rate: float,
                          action_taken: str, metrics: Dict = None, notes: str = None):
        """
        Log a monitoring run to database.

        Args:
            sample_size: Number of samples checked
            disagreement_rate: Disagreement rate found
            action_taken: Action taken (continue/escalate/retrain)
            metrics: Additional metrics (JSON)
            notes: Notes about the run
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO monitoring_runs
                (sample_size, disagreement_rate, action_taken, metrics, notes)
                VALUES (%s, %s, %s, %s, %s)
            """, (sample_size, disagreement_rate, action_taken,
                  json.dumps(metrics) if metrics else None, notes))

            self.conn.commit()
        finally:
            cursor.close()

    def register_model_version(self, version: str, f1_score: float,
                              is_active: bool = False, is_challenger: bool = False,
                              metadata: Dict = None):
        """
        Register a new model version.

        Args:
            version: Model version string
            f1_score: Validation F1 score
            is_active: Whether this is the active production model
            is_challenger: Whether this is a challenger in A/B test
            metadata: Additional metadata
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO model_versions
                (version, f1_score_validation, is_active, is_challenger, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (version) DO UPDATE
                SET f1_score_validation = EXCLUDED.f1_score_validation,
                    is_active = EXCLUDED.is_active,
                    is_challenger = EXCLUDED.is_challenger,
                    metadata = EXCLUDED.metadata
            """, (version, f1_score, is_active, is_challenger,
                  json.dumps(metadata) if metadata else None))

            self.conn.commit()
        finally:
            cursor.close()

    def get_active_model_version(self) -> str:
        """Get currently active model version."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT version FROM model_versions
                WHERE is_active = TRUE
                ORDER BY deployed_at DESC
                LIMIT 1
            """)
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            cursor.close()

    def cleanup_old_data(self, days_to_keep: int = None):
        """
        Clean up data older than specified days.

        Args:
            days_to_keep: Number of days to retain (default: from PRODUCTION_CONFIG)
        """
        # Use config value if not provided - NO HARDCODING!
        if days_to_keep is None:
            days_to_keep = PRODUCTION_CONFIG['retention']['archive_after_days']
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                DELETE FROM predictions_log
                WHERE timestamp < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))

            deleted = cursor.rowcount
            self.conn.commit()

            print(f"✓ Cleaned up {deleted} old predictions (older than {days_to_keep} days)")
        finally:
            cursor.close()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 30, 2025
@author: ramyalsaffar
"""
