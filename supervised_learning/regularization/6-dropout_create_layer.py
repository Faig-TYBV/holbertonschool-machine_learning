#!/usr/bin/env python3
"""dropout create layer function
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

    if cost - opt_cost >= threshold:
        count += 1
    else:
        count = 0
    if count >= patience:
        return True, count
    return False, count
