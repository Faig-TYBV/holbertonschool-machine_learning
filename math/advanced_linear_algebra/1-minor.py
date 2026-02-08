#!/usr/bin/env python3
'''
Docstring for holbertonschool-machine_learning.math.advanced_linear_algebra.1-minor
'''


def determinant(matrix):
    '''Calculates the determinant of a matrix:'''

    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for j in range(n):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(minor)
    return det

def minor(matrix):    
  '''Calculates the minor matrix of a matrix:'''

  # 1. Validation: Check if matrix is a list of lists
  if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
      raise TypeError("matrix must be a list of lists")
  # 2. Handle the 0x0 case [[]] or []
  if matrix == [] or matrix == [[]]:
      return 1
  n = len(matrix)
  # 3. Validation: Check if matrix is square
  for row in matrix:
      if len(row) != n:
          raise ValueError("matrix must be a square matrix")
  # 4. Calculate the minor matrix
  minor_matrix = []
  for i in range(n):
      minor_row = []
      for j in range(n):
          # Create the minor by removing row i and column j
          minor = [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]
          minor_row.append(determinant(minor))
      minor_matrix.append(minor_row)
  return minor_matrix