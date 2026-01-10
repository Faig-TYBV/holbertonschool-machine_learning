#!/usr/bin/env python3
"""sorting based on time"""


def flip_switch(df):
    """Sorting"""

    df = df.sort_values(by="Timestamp", ascending=False)
    return df.T
