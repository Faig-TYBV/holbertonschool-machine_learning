#!/usr/bin/env python3
""" descriptive statistics of a DataFrame """


def analyze(df):
    """ descriptive statistics of a DataFrame """

    return df.drop(columns=['Timestamp']).describe()
