#!/usr/bin/env python3
'''time to get the inverse of a matrix'''


def determinant(matrix):
    '''Calculates the determinant of a matrix:'''

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix) or len(matrix) == 0:
            raise ValueError("matrix must be a non-empty square matrix")
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for j in range(len(matrix[0])):
        minor = []
        for row in matrix[1:]:
            minor.append(row[:j] + row[j+1:])
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(minor)
    return det


def cofactor(matrix):
    '''
    Calculates the cofactor matrix of a matrix:
    '''

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix) or len(matrix) == 0:
            raise ValueError("matrix must be a non-empty square matrix")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    n = len(matrix)
    if n == 1:
        return [[1]]
    if n == 2:
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            minor = []
            for r in range(n):
                if r == i:
                    continue
                minor.append(matrix[r][:j] + matrix[r][j+1:])
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * determinant(minor))
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix


def adjugate(matrix):
    '''
    Calculates the adjugate matrix of a matrix:
    '''

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix) or len(matrix) == 0:
            raise ValueError("matrix must be a non-empty square matrix")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    n = len(matrix)
    if n == 1:
        return [[1]]
    if n == 2:
        return [[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]]
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []
    for j in range(n):
        adjugate_row = []
        for i in range(n):
            adjugate_row.append(cofactor_matrix[i][j])
        adjugate_matrix.append(adjugate_row)
    return adjugate_matrix


def inverse(matrix):
    '''
    Calculates the inverse of a matrix:
    '''

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix) or len(matrix) == 0:
            raise ValueError("matrix must be a non-empty square matrix")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    n = len(matrix)
    if n == 1:
        return [[1 / matrix[0][0]]]
    det = determinant(matrix)
    if det == 0:
        return None
    adjugate_matrix = adjugate(matrix)
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adjugate_matrix[i][j] / det)
        inverse_matrix.append(inverse_row)
    return inverse_matrix
