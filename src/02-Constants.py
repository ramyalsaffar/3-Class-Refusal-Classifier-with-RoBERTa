# Constants
#----------


# Device Configuration
#---------------------
# Auto-detect best available device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"🚀 CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
elif IS_MAC and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🚀 MPS available - using Apple Silicon GPU")
else:
    DEVICE = torch.device('cpu')
    print("⚠️  No GPU available - using CPU (training will be slow)")


#------------------------------------------------------------------------------


# Python Display Settings
#------------------------
pd.set_option('display.max_rows', 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", 250)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.4f}'.format


#------------------------------------------------------------------------------


# macOS Sleep Prevention
#-----------------------
# Prevents Mac from sleeping during long-running experiments
# Uses native 'caffeinate' command to keep system awake

class KeepAwake:
    """
    Context manager to prevent macOS from sleeping during long operations.

    Uses the native 'caffeinate' command which prevents:
    - Display sleep
    - System sleep
    - Disk sleep

    Automatically restores normal sleep behavior when done or on error.
    Only active on macOS - no-op on other platforms.

    Usage:
        with KeepAwake():
            # Your long-running code here
            pipeline.run_full_pipeline()
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize KeepAwake context manager.

        Args:
            verbose: Print status messages (default: True)
        """
        self.verbose = verbose
        self.process = None
        self.is_mac = sys.platform == 'darwin'

    def __enter__(self):
        """Start preventing sleep when entering context."""
        if not self.is_mac:
            if self.verbose:
                print("ℹ️  Sleep prevention only available on macOS")
            return self

        try:
            # Start caffeinate process
            # -d: Prevent display sleep
            # -i: Prevent system idle sleep
            # -m: Prevent disk sleep
            self.process = subprocess.Popen(
                ['caffeinate', '-dims'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            if self.verbose:
                print("☕ caffeinate activated - Mac will stay awake during execution")

            # Register cleanup in case of unexpected termination
            atexit.register(self._cleanup)

        except FileNotFoundError:
            if self.verbose:
                print("⚠️  caffeinate command not found - sleep prevention unavailable")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not activate sleep prevention: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop preventing sleep when exiting context."""
        self._cleanup()
        return False  # Don't suppress exceptions

    def _cleanup(self):
        """Terminate caffeinate process and restore normal sleep behavior."""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                if self.verbose:
                    print("☕ caffeinate deactivated - Mac sleep settings restored")
            except Exception:
                # Force kill if terminate doesn't work
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None
                # Unregister from atexit to avoid double cleanup
                try:
                    atexit.unregister(self._cleanup)
                except Exception:
                    pass




# =============================================================================
# CLASS LABELS AND MAPPINGS
# =============================================================================

# Refusal Classifier - Class names for display
#----------------------------------------------
CLASS_NAMES = ['No Refusal', 'Hard Refusal', 'Soft Refusal']

# Jailbreak Detector - Class names for display
#----------------------------------------------
JAILBREAK_CLASS_NAMES = ['Jailbreak Failed', 'Jailbreak Succeeded']


# =============================================================================
# VISUALIZATION COLORS
# =============================================================================

# Refusal Class colors (for plots)
#-----------------------------------
PLOT_COLORS = {
    'no_refusal': '#2ecc71',        # Green
    'hard_refusal': '#e74c3c',      # Red
    'soft_refusal': '#f39c12'       # Orange
}

# Class colors as list (for matplotlib)
PLOT_COLORS_LIST = ['#2ecc71', '#e74c3c', '#f39c12']

# Jailbreak Class colors (for plots)
#------------------------------------
JAILBREAK_PLOT_COLORS = {
    'jailbreak_failed': '#27ae60',          # Dark Green (success - model defended)
    'jailbreak_succeeded': '#c0392b'        # Dark Red (failure - model broken)
}

# Jailbreak colors as list (for matplotlib)
JAILBREAK_PLOT_COLORS_LIST = ['#27ae60', '#c0392b']

# Model colors (for per-model analysis)
#---------------------------------------
# WHY: Matches API_CONFIG['response_models'] naming convention
MODEL_COLORS = {
    'claude-sonnet-4-5': '#3498db',     # Blue
    'gpt-5': '#9b59b6',                 # Purple
    'gemini-2.5-flash': '#e67e22'       # Orange
}


# =============================================================================
# ERROR HANDLING
# =============================================================================

# Error indicator value (used to mark failed evaluations)
ERROR_VALUE = -1

# Error response placeholder
ERROR_RESPONSE = "[ERROR]"


# =============================================================================
# WILDJAILBREAK DATASET CITATION (NEW - V09)
# =============================================================================

# WildJailbreak Dataset - Used for supplementing jailbreak training data
#------------------------------------------------------------------------
# When modern LLMs successfully defend against all jailbreak attempts,
# we supplement training data from the WildJailbreak dataset to ensure
# the jailbreak detector has sufficient positive samples.

WILDJAILBREAK_DATASET_INFO = {
    'name': 'WildJailbreak',
    'source': 'AllenAI',
    'url': 'https://huggingface.co/datasets/allenai/wildjailbreak',
    'paper_url': 'https://arxiv.org/abs/2406.18510',
    'paper_title': 'WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models',
    'authors': 'Liwei Jiang, Kavel Rao, Seungju Han, et al.',
    'conference': 'NeurIPS',
    'year': '2024',
    'size': '262K prompt-response pairs',
    'adversarial_harmful_samples': '82,728',  # Successful jailbreaks
    'license': 'Apache 2.0'
}

WILDJAILBREAK_CITATION = """
@inproceedings{jiang2024wildteaming,
    title={WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models},
    author={Liwei Jiang and Kavel Rao and Seungju Han and Allyson Ettinger and Faeze Brahman and Sachin Kumar and Niloofar Mireshghallah and Ximing Lu and Maarten Sap and Yejin Choi and Nouha Dziri},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2024},
    volume={37},
    url={https://arxiv.org/abs/2406.18510}
}
"""

# Dataset Acknowledgment
WILDJAILBREAK_ACKNOWLEDGMENT = """
This project uses the WildJailbreak dataset from AllenAI for supplementing
jailbreak detection training data when insufficient positive samples are
collected from our primary pipeline. WildJailbreak provides diverse,
in-the-wild jailbreak tactics that enhance model robustness.
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_timestamp(format_type='file'):
    """
    Get standard timestamp for consistent naming across the project.

    Matches the format used in EXPERIMENT_CONFIG['experiment_name'] for consistency.

    Args:
        format_type: 'file' for filenames (YYYYmmdd_HHMM),
                     'display' for human-readable (YYYY-MM-DD HH:MM:SS)

    Returns:
        Formatted timestamp string

    Examples:
        >>> get_timestamp('file')
        '20250115_1430'
        >>> get_timestamp('display')
        '2025-01-15 14:30:00'
    """
    if format_type == 'file':
        return datetime.now().strftime("%Y%m%d_%H%M")
    elif format_type == 'display':
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return datetime.now().strftime("%Y%m%d_%H%M")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
