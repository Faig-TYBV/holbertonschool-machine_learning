#!/usr/bin/env python3
"""Removing Close's being NaN"""


def prune(df):
    """Removing invalid entries"""

    df.dropna(subset=["Close"], inplace=True)
    return df
