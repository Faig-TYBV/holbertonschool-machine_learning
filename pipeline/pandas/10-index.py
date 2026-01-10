#!/usr/bin/env python3
"""Set Timestamp as index"""


def index(df):
    """Sets the Timestamp column as the DataFrame index"""
    df = df.set_index("Timestamp")
    return df
