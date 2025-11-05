# Weighted Loss Module
#---------------------
# Handles class imbalance with weighted CrossEntropyLoss.
# All imports are in 00-Imports.py
###############################################################################


def calculate_class_weights(class_counts: List[int], device: torch.device,
                           allow_zero: bool = False, zero_weight: float = 1.0) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.

    Formula: weight_i = total_samples / (num_classes * count_i)

    Args:
        class_counts: List of sample counts per class [count_class0, count_class1, ...]
        device: torch device
        allow_zero: If True, handle zero counts gracefully. If False, raise error.
        zero_weight: Weight to use for classes with zero samples (only if allow_zero=True)

    Returns:
        torch.Tensor of class weights

    Raises:
        ValueError: If any class has zero samples and allow_zero=False

    Example:
        class_counts = [3000, 2385, 615]  # [no_refusal, hard_refusal, soft_refusal]
        weights = calculate_class_weights(class_counts, device)
        # Result: [0.67, 0.84, 3.25]
    """
    # Check for zero counts
    zero_indices = [i for i, count in enumerate(class_counts) if count == 0]

    if zero_indices and not allow_zero:
        raise ValueError(
            f"Class counts contain zeros: {class_counts}. "
            f"Classes with zero samples: {zero_indices}. "
            f"All classes must have at least one sample for weighted loss calculation."
        )

    total_samples = sum(class_counts)
    num_classes = len(class_counts)

    # Calculate weights, handling zeros if allowed
    weights = []
    for count in class_counts:
        if count == 0:
            weights.append(zero_weight)  # Use configurable zero weight
        else:
            weights.append(total_samples / (num_classes * count))

    return torch.FloatTensor(weights).to(device)


def get_weighted_criterion(class_counts: List[int], device: torch.device,
                          class_names: List[str] = None,
                          allow_zero: bool = False, zero_weight: float = 1.0) -> nn.CrossEntropyLoss:
    """
    Get weighted CrossEntropyLoss criterion.

    Args:
        class_counts: List of sample counts per class
        device: torch device
        class_names: Optional list of class names for display (e.g., CLASS_NAMES or JAILBREAK_CLASS_NAMES)
        allow_zero: If True, handle zero counts gracefully. If False, raise error.
        zero_weight: Weight to use for classes with zero samples (only if allow_zero=True)

    Returns:
        nn.CrossEntropyLoss with class weights

    Raises:
        ValueError: If any class has zero samples and allow_zero=False
    """
    class_weights = calculate_class_weights(class_counts, device, allow_zero, zero_weight)

    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Print class-specific weights with names if provided
    if class_names:
        for i, weight in enumerate(class_weights):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"  Class {i} ({class_name}): {weight:.3f}")
    else:
        # Default 3-class refusal names
        for i, weight in enumerate(class_weights):
            if len(class_weights) == 3:
                names = ['No Refusal', 'Hard Refusal', 'Soft Refusal']
                print(f"  Class {i} ({names[i]}): {weight:.3f}")
            else:
                print(f"  Class {i}: {weight:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    return criterion


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
