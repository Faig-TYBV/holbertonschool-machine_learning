#!/usr/bin/env python3
"""renaming column"""
import pandas


def rename(df):
    """Modifying our dataframe"""

    df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df[["Datetime", "Close"]]
