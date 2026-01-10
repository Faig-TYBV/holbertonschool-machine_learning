#!/usr/bin/env python3
""" sorting according to High """


def high(df):
    """Sorting according to High"""

    df.sort_values(by="High", ascending=False, inplace=True)
