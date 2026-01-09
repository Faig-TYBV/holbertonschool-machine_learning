#!/usr/bin/env python3
""" Learning pandas """
import pandas


def from_numpy(array):
    lst = [chr(ord('A') + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=lst)
