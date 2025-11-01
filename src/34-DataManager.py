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

    def get_recent_predictions(self, hours: int = 24, limit: int = None,
                              model_version: str = None) -> pd.DataFrame:
        """
        Get recent predictions from database.

        Args:
            hours: Number of hours to look back
            limit: Maximum number of samples to return
            model_version: Filter by specific model version

        Returns:
            DataFrame with predictions
        """
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

    def sample_for_monitoring(self, sample_size: int = 100,
                            hours: int = 24) -> pd.DataFrame:
        """
        GENERIC: Sample predictions for monitoring (stratified by prediction class).
        Works with any number of classes.

        Args:
            sample_size: Number of samples to return
            hours: Look back period

        Returns:
            DataFrame with sampled predictions
        """
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

        Strategy:
        - Recent (0-7 days): 100% of problematic samples + 20% of correct samples
        - Medium-term (8-30 days): 50% stratified sample
        - Long-term (31-180 days): 10% representative sample

        Returns:
            DataFrame ready for retraining
        """
        print("\n" + "="*60)
        print("COLLECTING RETRAINING DATA")
        print("="*60)

        cursor = self.conn.cursor()
        try:
            all_samples = []

            # Recent: 0-7 days
            print("\n1. Recent data (0-7 days):")
            print("   - 100% of problematic samples (disagreements)")
            cursor.execute("""
                SELECT prompt, response, judge_label as label
                FROM predictions_log
                WHERE timestamp > NOW() - INTERVAL '7 days'
                AND judge_label IS NOT NULL
                AND judge_label != prediction
            """)
            problematic = cursor.fetchall()
            all_samples.extend(problematic)
            print(f"   - Collected {len(problematic)} problematic samples")

            print("   - 20% of correct high-confidence samples")
            cursor.execute("""
                SELECT prompt, response, judge_label as label
                FROM predictions_log
                WHERE timestamp > NOW() - INTERVAL '7 days'
                AND judge_label IS NOT NULL
                AND judge_label = prediction
                AND confidence > 0.8
                ORDER BY RANDOM()
                LIMIT (SELECT COUNT(*) FROM predictions_log
                       WHERE timestamp > NOW() - INTERVAL '7 days'
                       AND judge_label = prediction) / 5
            """)
            recent_correct = cursor.fetchall()
            all_samples.extend(recent_correct)
            print(f"   - Collected {len(recent_correct)} correct samples")

            # Medium-term: 8-30 days (50% sample, stratified)
            # GENERIC: Dynamic class count
            num_classes = len(CLASS_NAMES)
            print("\n2. Medium-term data (8-30 days): 50% stratified sample")
            for label in range(num_classes):
                cursor.execute("""
                    SELECT prompt, response, judge_label as label
                    FROM predictions_log
                    WHERE timestamp BETWEEN NOW() - INTERVAL '30 days' AND NOW() - INTERVAL '7 days'
                    AND judge_label = %s
                    ORDER BY RANDOM()
                    LIMIT (SELECT COUNT(*) FROM predictions_log
                           WHERE timestamp BETWEEN NOW() - INTERVAL '30 days' AND NOW() - INTERVAL '7 days'
                           AND judge_label = %s) / 2
                """, (label, label))
                medium_samples = cursor.fetchall()
                all_samples.extend(medium_samples)
            print(f"   - Collected {len(all_samples) - len(problematic) - len(recent_correct)} samples")

            # Long-term: 31-180 days (10% sample, stratified)
            print("\n3. Long-term data (31-180 days): 10% representative sample")
            for label in range(num_classes):
                cursor.execute("""
                    SELECT prompt, response, judge_label as label
                    FROM predictions_log
                    WHERE timestamp BETWEEN NOW() - INTERVAL '180 days' AND NOW() - INTERVAL '30 days'
                    AND judge_label = %s
                    ORDER BY RANDOM()
                    LIMIT (SELECT COUNT(*) FROM predictions_log
                           WHERE timestamp BETWEEN NOW() - INTERVAL '180 days' AND NOW() - INTERVAL '30 days'
                           AND judge_label = %s) / 10
                """, (label, label))
                long_samples = cursor.fetchall()
                all_samples.extend(long_samples)

            total_longterm = len(all_samples) - len(problematic) - len(recent_correct) - \
                            (len(all_samples) - len(problematic) - len(recent_correct))
            print(f"   - Collected {total_longterm} samples")

            # Convert to DataFrame
            df = pd.DataFrame(all_samples, columns=['prompt', 'response', 'label'])

            print(f"\n{'='*60}")
            print(f"RETRAINING DATA SUMMARY")
            print(f"{'='*60}")
            print(f"Total samples: {len(df)}")
            logger.info(f"Collected {len(df)} samples for retraining")
            # GENERIC: Dynamic class count
            for i in range(num_classes):
                count = (df['label'] == i).sum()
                pct = count / len(df) * 100 if len(df) > 0 else 0
                print(f"  {CLASS_NAMES[i]}: {count} ({pct:.1f}%)")

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

    def cleanup_old_data(self, days_to_keep: int = 180):
        """
        Clean up data older than specified days.

        Args:
            days_to_keep: Number of days to retain
        """
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
