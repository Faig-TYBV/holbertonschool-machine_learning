#!/usr/bin/env python3
""" from dataframe to numpy array """


def array(df):
    """ from dataframe to numpy array """

    return df[["High", "Close"]].tail(10).to_numpy()
