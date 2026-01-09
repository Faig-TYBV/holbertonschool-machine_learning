#!/usr/bin/env python3
""" Learning pandas """
import pandas as pd


def from_numpy(array):
    """ from array to dataframe """
    lst = [chr(ord('A') + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=lst)
