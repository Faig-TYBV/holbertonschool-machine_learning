#!/usr/bin/env python3
""" from file to dataframe """
import pandas as pd


def from_file(filename, delimiter):
    """ from file to dataframe """

    return pd.read_csv(filename, delimiter=delimiter)
