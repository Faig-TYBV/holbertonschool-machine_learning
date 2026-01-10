#!/usr/bin/env python3
""" taking a slice from dataframe """


def slice(df):
    """ taking a slice from dataframe """

    return df.loc[::60, ["High", "Low", "Close", "Volume_(BTC)"]]
