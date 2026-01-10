#!/usr/bin/env python3
""" descriptive statistics of a DataFrame """
import pandas as pd


def analyze(df):
    """ descriptive statistics of a DataFrame """

    return df.drop(columns=['Timestamp']).describe()