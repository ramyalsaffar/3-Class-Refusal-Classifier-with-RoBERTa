# CheckpointManager Utility
#---------------------------
# Handles checkpoint saving/loading for error recovery during long-running operations.
# Provides automatic cleanup and resumption capabilities.
# All imports are in 00-Imports.py
###############################################################################


class CheckpointManager:
    """
    Manages checkpoints for error recovery during long-running API operations.

    Features:
    - Automatic checkpoint saving at configurable intervals
    - Resume from latest checkpoint
    - Automatic cleanup of old checkpoints
    - Validation of checkpoint integrity
    """

    def __init__(self, checkpoint_dir: str, operation_name: str,
                 checkpoint_every: int = 100, auto_cleanup: bool = True,
                 keep_last_n: int = 2):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            operation_name: Name of operation (e.g., 'labeling', 'collection')
            checkpoint_every: Save checkpoint every N items
            auto_cleanup: Automatically delete old checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.operation_name = operation_name
        self.checkpoint_every = checkpoint_every
        self.auto_cleanup = auto_cleanup
        self.keep_last_n = keep_last_n

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, data: pd.DataFrame, last_index: int,
                       metadata: Dict = None) -> str:
        """
        Save checkpoint to disk.

        Args:
            data: DataFrame with processed data
            last_index: Last processed index
            metadata: Optional metadata dictionary

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{self.operation_name}_{timestamp}.pkl"
        )

        checkpoint_data = {
            'data': data.copy(),
            'last_index': last_index,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"💾 Checkpoint saved: {checkpoint_path}")
        print(f"   Progress: {last_index} items completed")

        # Auto-cleanup if enabled
        if self.auto_cleanup:
            self.cleanup_checkpoints(keep_last_n=self.keep_last_n)

        return checkpoint_path

    def load_latest_checkpoint(self, max_age_hours: int = 48) -> Optional[Dict]:
        """
        Load the most recent checkpoint if available.

        Args:
            max_age_hours: Maximum age of checkpoint in hours

        Returns:
            Checkpoint data dictionary or None if no valid checkpoint found
        """
        # Find all checkpoints for this operation
        pattern = f"checkpoint_{self.operation_name}_*.pkl"
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))

        if not checkpoints:
            return None

        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        latest_checkpoint = checkpoints[0]

        # Check age
        file_age_hours = (time.time() - os.path.getmtime(latest_checkpoint)) / 3600
        if file_age_hours > max_age_hours:
            print(f"⚠️  Latest checkpoint is {file_age_hours:.1f} hours old (max: {max_age_hours}h)")
            print(f"   Skipping checkpoint: {latest_checkpoint}")
            return None

        # Load checkpoint
        try:
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Validate checkpoint
            if not self._validate_checkpoint(checkpoint_data):
                print(f"⚠️  Checkpoint validation failed: {latest_checkpoint}")
                return None

            print(f"✅ Loaded checkpoint: {latest_checkpoint}")
            print(f"   Resuming from index: {checkpoint_data['last_index']}")
            print(f"   Checkpoint age: {file_age_hours:.1f} hours")

            return checkpoint_data

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return None

    def cleanup_checkpoints(self, keep_last_n: int = 2):
        """
        Delete old checkpoints, keeping only the N most recent.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        pattern = f"checkpoint_{self.operation_name}_*.pkl"
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))

        if len(checkpoints) <= keep_last_n:
            return

        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)

        # Delete old checkpoints
        to_delete = checkpoints[keep_last_n:]
        for checkpoint in to_delete:
            try:
                os.remove(checkpoint)
                print(f"🗑️  Deleted old checkpoint: {os.path.basename(checkpoint)}")
            except Exception as e:
                print(f"⚠️  Could not delete {checkpoint}: {e}")

    def _validate_checkpoint(self, checkpoint_data: Dict) -> bool:
        """
        Validate checkpoint integrity.

        Args:
            checkpoint_data: Checkpoint dictionary

        Returns:
            True if valid, False otherwise
        """
        required_keys = ['data', 'last_index', 'timestamp']

        # Check required keys
        if not all(key in checkpoint_data for key in required_keys):
            return False

        # Check data types
        if not isinstance(checkpoint_data['data'], pd.DataFrame):
            return False

        if not isinstance(checkpoint_data['last_index'], int):
            return False

        # Check DataFrame not empty
        if checkpoint_data['data'].empty:
            return False

        return True


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
