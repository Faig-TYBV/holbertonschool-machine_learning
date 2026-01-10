#!/usr/bin/env python3
"""Concatenate two DataFrames along rows with keys after indexing."""
import pandas as pd


def concat(df1, df2):
    """Concatenate two DataFrames along rows with keys after indexing."""

    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:1417411920]
    df = pd.concat(
        [df2, df1],
        keys=['bitstamp', 'coinbase']
    )
    return df
