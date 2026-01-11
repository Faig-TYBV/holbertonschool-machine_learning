#!/usr/bin/env python3
"""Concatenate two DataFrames with a MultiIndex and select a time range."""
import pandas as pd


def hierarchy(df1, df2):
    """Concatenate two DataFrames with a MultiIndex and select a time range."""

    # Import the index function
    index = __import__('10-index').index

    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Concatenate with keys
    df = pd.concat(
        [df2, df1],
        keys=['bitstamp', 'coinbase']
    )

    # Rearrange MultiIndex so Timestamp is the first level
    df = df.swaplevel(0, 1)

    # Ensure chronological order
    df = df.sort_index()

    # Select timestamps from 1417411980 to 1417417980 (inclusive)
    df = df.loc[1417411980:1417417980]

    return df
