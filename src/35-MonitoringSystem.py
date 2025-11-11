# MonitoringSystem Module
#------------------------
# Daily performance monitoring with escalating validation.
# Detects model drift and triggers retraining when needed.
# NOTE: This is a standalone production script - core imports from 01-Imports.py
###############################################################################

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringSystem:
    """Monitor production model performance with escalating checks."""

    def __init__(self, data_manager: 'DataManager', judge_api_key: str):
        """
        Initialize monitoring system.

        Args:
            data_manager: DataManager instance
            judge_api_key: OpenAI API key for LLM judge
        """
        self.data_manager = data_manager
        self.judge = DataLabeler(api_key=judge_api_key)
        self.thresholds = PRODUCTION_CONFIG['monitoring_thresholds']

    def run_daily_check(self) -> Dict:
        """
        Run daily performance check with small sample.

        Returns:
            Dictionary with check results
        """
        print()
        print_banner("DAILY PERFORMANCE CHECK", width=60)
        print(f"Timestamp: {get_timestamp('display')}")

        # Sample predictions from last 24 hours
        sample_size = self.thresholds['small_sample_size']
        print(f"\n1. Sampling {sample_size} predictions from last 24 hours...")

        sample_df = self.data_manager.sample_for_monitoring(
            sample_size=sample_size,
            hours=24
        )

        if len(sample_df) == 0:
            print("⚠️  No predictions found in last 24 hours")
            print("   Skipping monitoring check")
            return {
                'status': 'skipped',
                'reason': 'no_predictions',
                'action': 'continue'
            }

        print(f"✓ Sampled {len(sample_df)} predictions")

        # Get judge labels with progress bar
        print(f"\n2. Running LLM Judge on sample...")
        logger.info(f"Running LLM Judge on {len(sample_df)} samples")
        judge_labels = []

        try:
            for _, row in tqdm(sample_df.iterrows(), total=len(sample_df),
                              desc="LLM Judge evaluation"):
                try:
                    refusal_label, is_jailbreak_attempt, jailbreak_success, _, _ = self.judge.label_response(
                        response=row['response'],
                        prompt=row['prompt']
                    )
                    judge_labels.append(refusal_label)  # Only use refusal label for monitoring
                    time.sleep(PRODUCTION_CONFIG.get('judge_rate_limit', 1.0))
                except Exception as e:
                    logger.error(f"LLM Judge failed for sample {row['id']}: {e}")
                    # Use model prediction as fallback
                    judge_labels.append(row['prediction'])

            # Update database with judge labels
            self.data_manager.update_with_judge_labels(
                sample_ids=sample_df['id'].tolist(),
                judge_labels=judge_labels
            )
        except Exception as e:
            logger.error(f"Failed to get judge labels: {e}")
            raise

        # Calculate disagreement rate
        disagreements = sum(
            1 for pred, judge in zip(sample_df['prediction'], judge_labels)
            if pred != judge
        )
        disagreement_rate = disagreements / len(sample_df)

        print(f"\n3. Results:")
        print(f"   Total samples: {len(sample_df)}")
        print(f"   Disagreements: {disagreements}")
        print(f"   Disagreement rate: {disagreement_rate*100:.2f}%")

        # Determine action
        action = self._determine_action(disagreement_rate, sample_size='small')

        # Log monitoring run
        metrics = {
            'total_samples': len(sample_df),
            'disagreements': disagreements,
            'disagreement_rate': disagreement_rate,
            'avg_model_confidence': float(sample_df['confidence'].mean())
        }

        self.data_manager.log_monitoring_run(
            sample_size=len(sample_df),
            disagreement_rate=disagreement_rate,
            action_taken=action,
            metrics=metrics
        )

        print(f"\n4. Action: {action.upper()}")

        return {
            'status': 'completed',
            'sample_size': len(sample_df),
            'disagreement_rate': disagreement_rate,
            'action': action,
            'metrics': metrics
        }

    def run_escalated_check(self) -> Dict:
        """
        Run escalated check with larger sample (triggered when small sample shows issues).

        Returns:
            Dictionary with check results
        """
        print()
        print_banner("ESCALATED PERFORMANCE CHECK", width=60)
        print(f"Timestamp: {get_timestamp('display')}")

        # Sample from last 7 days
        sample_size = self.thresholds['large_sample_size']
        print(f"\n1. Sampling {sample_size} predictions from last 7 days...")

        sample_df = self.data_manager.sample_for_monitoring(
            sample_size=sample_size,
            hours=168  # 7 days
        )

        print(f"✓ Sampled {len(sample_df)} predictions")

        # Get judge labels with progress bar
        print(f"\n2. Running LLM Judge on sample (this will take longer)...")
        logger.info(f"Running LLM Judge on {len(sample_df)} samples (escalated)")
        judge_labels = []

        try:
            for _, row in tqdm(sample_df.iterrows(), total=len(sample_df),
                              desc="LLM Judge evaluation (escalated)"):
                try:
                    refusal_label, is_jailbreak_attempt, jailbreak_success, _, _ = self.judge.label_response(
                        response=row['response'],
                        prompt=row['prompt']
                    )
                    judge_labels.append(refusal_label)  # Only use refusal label for monitoring
                    time.sleep(PRODUCTION_CONFIG.get('judge_rate_limit', 1.0))
                except Exception as e:
                    logger.error(f"LLM Judge failed for sample {row['id']}: {e}")
                    # Use model prediction as fallback
                    judge_labels.append(row['prediction'])

            # Update database
            self.data_manager.update_with_judge_labels(
                sample_ids=sample_df['id'].tolist(),
                judge_labels=judge_labels
            )
        except Exception as e:
            logger.error(f"Failed to get judge labels: {e}")
            raise

        # Calculate disagreement rate
        disagreements = sum(
            1 for pred, judge in zip(sample_df['prediction'], judge_labels)
            if pred != judge
        )
        disagreement_rate = disagreements / len(sample_df)

        print(f"\n3. Results:")
        print(f"   Total samples: {len(sample_df)}")
        print(f"   Disagreements: {disagreements}")
        print(f"   Disagreement rate: {disagreement_rate*100:.2f}%")

        # Determine action
        action = self._determine_action(disagreement_rate, sample_size='large')

        # Log monitoring run
        # GENERIC: Dynamic class count
        num_classes = MODEL_CONFIG['num_classes']
        metrics = {
            'total_samples': len(sample_df),
            'disagreements': disagreements,
            'disagreement_rate': disagreement_rate,
            'avg_model_confidence': float(sample_df['confidence'].mean()),
            'disagreement_by_class': {
                CLASS_NAMES[i]: sum(
                    1 for pred, judge, true_pred in
                    zip(sample_df['prediction'], judge_labels, sample_df['prediction'])
                    if true_pred == i and pred != judge
                )
                for i in range(num_classes)
            }
        }

        self.data_manager.log_monitoring_run(
            sample_size=len(sample_df),
            disagreement_rate=disagreement_rate,
            action_taken=action,
            metrics=metrics
        )

        print(f"\n4. Action: {action.upper()}")

        return {
            'status': 'completed',
            'sample_size': len(sample_df),
            'disagreement_rate': disagreement_rate,
            'action': action,
            'metrics': metrics
        }

    def _determine_action(self, disagreement_rate: float, sample_size: str) -> str:
        """
        Determine action based on disagreement rate and sample size.

        Args:
            disagreement_rate: Disagreement rate between model and judge
            sample_size: 'small' or 'large'

        Returns:
            Action to take: 'continue', 'escalate', or 'retrain'
        """
        if sample_size == 'small':
            if disagreement_rate < self.thresholds['warning_threshold']:
                return 'continue'
            elif disagreement_rate < self.thresholds['escalate_threshold']:
                return 'monitor'  # Continue monitoring but flag
            else:
                return 'escalate'

        else:  # large sample
            if disagreement_rate < self.thresholds['retrain_threshold']:
                return 'continue'
            else:
                return 'retrain'

    def run_monitoring_cycle(self) -> Dict:
        """
        Run complete monitoring cycle with escalation if needed.

        Returns:
            Dictionary with final results
        """
        print()
        print_banner("MONITORING CYCLE", width=80)
        print()

        # Step 1: Daily check
        daily_result = self.run_daily_check()

        if daily_result['action'] == 'skipped':
            return daily_result

        # Step 2: Escalate if needed
        if daily_result['action'] == 'escalate':
            print("\n⚠️  Small sample showed issues - escalating to larger sample")
            logger.warning("Escalating to larger sample due to high disagreement rate")
            # Brief pause for UX (use get with default)
            ui_delay = PRODUCTION_CONFIG.get('escalation_ui_delay', 2.0)
            time.sleep(ui_delay)

            escalated_result = self.run_escalated_check()

            if escalated_result['action'] == 'retrain':
                print("\n🚨 Large sample confirms degradation - RETRAINING NEEDED")

            return escalated_result

        else:
            if daily_result['action'] == 'monitor':
                print("\n⚠️  Warning threshold exceeded - continuing to monitor")
            else:
                print("\n✅ Model performance is healthy")

            return daily_result

    def get_monitoring_history(self, days: int = None) -> pd.DataFrame:
        """
        Get monitoring history from database.

        Args:
            days: Number of days to look back (default: from PRODUCTION_CONFIG)

        Returns:
            DataFrame with monitoring history
        """
        # Use config value if not provided - NO HARDCODING!
        if days is None:
            days = PRODUCTION_CONFIG['monitoring_thresholds']['trend_window_days']
        
        cursor = self.data_manager.conn.cursor()

        cursor.execute("""
            SELECT timestamp, sample_size, disagreement_rate, action_taken, metrics
            FROM monitoring_runs
            WHERE timestamp > NOW() - INTERVAL '%s days'
            ORDER BY timestamp DESC
        """, (days,))

        columns = ['timestamp', 'sample_size', 'disagreement_rate', 'action_taken', 'metrics']
        df = pd.DataFrame(cursor.fetchall(), columns=columns)

        cursor.close()
        return df

    def plot_monitoring_trends(self, days: int = None, output_path: str = None):
        """
        Plot monitoring trends over time.

        Args:
            days: Number of days to plot (default: from PRODUCTION_CONFIG)
            output_path: Path to save plot (shows if None)
        """
        # Use config value if not provided - NO HARDCODING!
        if days is None:
            days = PRODUCTION_CONFIG['monitoring_thresholds']['trend_window_days']
        
        df = self.get_monitoring_history(days=days)

        if len(df) == 0:
            print("⚠️  No monitoring history available")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Disagreement rate over time
        ax1.plot(df['timestamp'], df['disagreement_rate'] * 100, marker='o')
        ax1.axhline(y=self.thresholds['warning_threshold'] * 100,
                   color='orange', linestyle='--', label='Warning Threshold')
        ax1.axhline(y=self.thresholds['retrain_threshold'] * 100,
                   color='red', linestyle='--', label='Retrain Threshold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Disagreement Rate (%)')
        ax1.set_title('Model-Judge Disagreement Rate Over Time')
        ax1.legend()
        ax1.grid(True, alpha=VISUALIZATION_CONFIG['alpha_grid'])

        # Plot 2: Action taken over time
        action_colors = {'continue': 'green', 'monitor': 'orange',
                        'escalate': 'red', 'retrain': 'darkred'}
        colors = [action_colors.get(action, 'gray') for action in df['action_taken']]

        ax2.scatter(df['timestamp'], df['sample_size'], c=colors, s=100, alpha=0.6)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sample Size')
        ax2.set_title('Monitoring Sample Size and Actions')
        ax2.grid(True, alpha=VISUALIZATION_CONFIG['alpha_grid'])

        # Legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=action.title())
                          for action, color in action_colors.items()]
        ax2.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            print(f"✓ Saved monitoring trends to {output_path}")
        else:
            plt.show()

        plt.close()


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 30, 2025
@author: ramyalsaffar
"""
