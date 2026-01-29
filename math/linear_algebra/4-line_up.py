#!/usr/bin/env python3
'''finding sum of 2 arrays'''


def add_arrays(arr1, arr2):
    '''finding sum of 2 arrays'''

    if len(arr2) != len(arr1):
        return None
    arr3 = []
    for i in range(len(arr1)):
        arr3.append(arr1[i]+arr2[i])
    return arr3
