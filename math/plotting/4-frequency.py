#!/usr/bin/env python3
"""Using histograms"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """finding frequency"""

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 31, 5))
    plt.ylim(0, 30)
    plt.xlim(0, 100)
    plt.title("Project A")
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor="black")
    plt.show()
