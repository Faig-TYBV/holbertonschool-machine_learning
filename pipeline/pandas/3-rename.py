#!/usr/bin/env python3
"""renaming column"""
import pandas as pd


def rename(df):
    """Modifying our dataframe"""

    df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df[["Datetime", "Close"]]
    return df
