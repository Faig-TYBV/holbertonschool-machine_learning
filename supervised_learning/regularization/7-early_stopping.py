#!/usr/bin/env python3
"""early stopping gradient descent function
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should stop early.
    Args:
        cost:      float - current validation cost
        opt_cost:  float - lowest recorded validation cost
        threshold: float - threshold for early stopping
        patience:  int - patience count for early stopping
        count:     int - how long threshold has not been met
    Returns:
        bool - whether to stop early, and the updated count
    """

    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count >= patience:
        return True, count
    return False, count
