# CheckpointManager Module
#--------------------------
# Centralized checkpoint management for error recovery.
# Supports saving/loading/validating checkpoints for long-running operations.
# All imports are in 00-Imports.py
###############################################################################


class CheckpointManager:
    """
    Centralized checkpoint management for error recovery.

    Features:
    - Save checkpoints incrementally during long operations
    - Resume from last valid checkpoint
    - Automatic cleanup after success
    - Checkpoint validation and age checks
    """

    def __init__(self, checkpoint_dir: str, checkpoint_prefix: str,
                 verbose: bool = True):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            checkpoint_prefix: Prefix for checkpoint files (e.g., 'labeling', 'collection')
            verbose: Print checkpoint progress messages
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.verbose = verbose

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, data: pd.DataFrame, last_index: int, metadata: Dict = None):
        """
        Save a checkpoint with current progress.

        Args:
            data: DataFrame with partial results
            last_index: Last processed index (for resume)
            metadata: Optional metadata to save with checkpoint
        """
        checkpoint_file = self._get_checkpoint_filename(last_index)

        checkpoint_data = {
            'data': data,
            'last_index': last_index,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pd.to_pickle(checkpoint_data, f)

        if self.verbose:
            print(f"✓ Checkpoint saved: {os.path.basename(checkpoint_file)} (index: {last_index})")

    def load_latest_checkpoint(self, max_age_hours: int = 48) -> Optional[Dict]:
        """
        Load the latest valid checkpoint.

        Args:
            max_age_hours: Max checkpoint age in hours (ignore older checkpoints)

        Returns:
            Checkpoint data dict with keys: ['data', 'last_index', 'timestamp', 'metadata']
            Returns None if no valid checkpoint found
        """
        checkpoint_files = self._get_all_checkpoints()

        if not checkpoint_files:
            return None

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        # Try loading checkpoints from newest to oldest
        for checkpoint_file in checkpoint_files:
            try:
                # Check age
                file_age_hours = (time.time() - os.path.getmtime(checkpoint_file)) / 3600
                if file_age_hours > max_age_hours:
                    if self.verbose:
                        print(f"⚠️  Checkpoint too old ({file_age_hours:.1f}h): {os.path.basename(checkpoint_file)}")
                    continue

                # Load checkpoint
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pd.read_pickle(f)

                # Validate
                if self._validate_checkpoint(checkpoint_data):
                    if self.verbose:
                        print(f"📂 Loaded checkpoint: {os.path.basename(checkpoint_file)}")
                        print(f"   Last index: {checkpoint_data['last_index']}")
                        print(f"   Timestamp: {checkpoint_data['timestamp']}")
                    return checkpoint_data
                else:
                    if self.verbose:
                        print(f"⚠️  Invalid checkpoint: {os.path.basename(checkpoint_file)}")

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to load {os.path.basename(checkpoint_file)}: {e}")
                continue

        return None

    def cleanup_checkpoints(self, keep_last_n: int = 2):
        """
        Clean up old checkpoints, optionally keeping the N most recent.

        Args:
            keep_last_n: Number of most recent checkpoints to keep (0 = delete all)
        """
        checkpoint_files = self._get_all_checkpoints()

        if not checkpoint_files:
            return

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        # Determine which files to delete
        if keep_last_n == 0:
            files_to_delete = checkpoint_files
        else:
            files_to_delete = checkpoint_files[keep_last_n:]

        # Delete files
        for checkpoint_file in files_to_delete:
            try:
                os.remove(checkpoint_file)
                if self.verbose:
                    print(f"🗑️  Deleted checkpoint: {os.path.basename(checkpoint_file)}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to delete {os.path.basename(checkpoint_file)}: {e}")

    def _get_checkpoint_filename(self, last_index: int) -> str:
        """Generate checkpoint filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.checkpoint_prefix}_checkpoint_{timestamp}_idx{last_index}.pkl"
        return os.path.join(self.checkpoint_dir, filename)

    def _get_all_checkpoints(self) -> List[str]:
        """Get all checkpoint files for this prefix."""
        pattern = f"{self.checkpoint_prefix}_checkpoint_*.pkl"
        checkpoint_files = []

        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(self.checkpoint_prefix) and filename.endswith('.pkl'):
                checkpoint_files.append(os.path.join(self.checkpoint_dir, filename))

        return checkpoint_files

    def _validate_checkpoint(self, checkpoint_data: Dict) -> bool:
        """
        Validate checkpoint data structure.

        Args:
            checkpoint_data: Checkpoint dict to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = ['data', 'last_index', 'timestamp']

        # Check required keys
        for key in required_keys:
            if key not in checkpoint_data:
                return False

        # Validate data is DataFrame
        if not isinstance(checkpoint_data['data'], pd.DataFrame):
            return False

        # Validate last_index is int
        if not isinstance(checkpoint_data['last_index'], (int, np.integer)):
            return False

        # Validate timestamp format
        try:
            datetime.fromisoformat(checkpoint_data['timestamp'])
        except:
            return False

        return True


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 5, 2025
@author: ramyalsaffar
"""
