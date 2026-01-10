#!/usr/bin/env python3
"""Fill missing values and clean dataframe"""


def fill(df):
    """Cleans and fills missing values in the DataFrame"""

    df = df.drop(columns=["Weighted_Price"], errors="ignore")
    df["Close"] = df["Close"].fillna(method="ffill")
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])
    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)   
    return df
