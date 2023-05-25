//
// Created by szymc on 01.05.2023.
//

#include "matrix_utils.h"
#include <stdio.h>
#include <stdlib.h>

void fill_matrix(Matrix *matrix, double value) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->values[i][j] = value;
        }
    }
}

// create matrix with given dimensions
Matrix *create_matrix(int rows, int cols) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    // allocate memory for matrix
    matrix->values = malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        matrix->values[i] = malloc(cols * sizeof(double));
    }
    fill_matrix(matrix, 0);
    return matrix;
}

void free_matrix(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->values[i]);
    }
    free(matrix->values);
    free(matrix);
}

// multiply two matrices
void matrix_multiply(Matrix *m1, Matrix *m2, Matrix *result) {
    if (m1->cols != m2->rows) {
        printf("Error: Matrix dimensions do not match!\n");
        return;
    }
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->cols; j++) {
            double sum = 0;
            for (int k = 0; k < m1->cols; k++) {
                sum += m1->values[i][k] * m2->values[k][j];
            }
            result->values[i][j] = sum;
        }
    }
}

// add two matrices
void add_matrices(Matrix *m1, Matrix *m2, Matrix *result) {
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        printf("Error: Matrix dimensions do not match!\n");
        return;
    }
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->values[i][j] = m1->values[i][j] + m2->values[i][j];
        }
    }
}

void matrix_subtract(Matrix *m1, Matrix *m2, Matrix *result) {
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        printf("Error: Matrix dimensions do not match!\n");
        return;
    }
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->values[i][j] = m1->values[i][j] - m2->values[i][j];
        }
    }
}

Matrix matrix_transpose(Matrix *matrix) {
    Matrix *result = create_matrix(matrix->cols, matrix->rows);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->values[j][i] = matrix->values[i][j];
        }
    }
    return *result;
}

//add a number to a matrix
void matrix_add_scalar(Matrix *matrix, double scalar, Matrix *result) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->values[i][j] += scalar;
        }
    }
}

// apply function to each element of matrix
void matrix_apply_function(Matrix *matrix, double (*function)(double), Matrix *result) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->values[i][j] = function(matrix->values[i][j]);
        }
    }
}

void print_matrix(Matrix *matrix) {
    printf("[");
    for (int i = 0; i < matrix->rows; i++) {
        printf("[");
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f", matrix->values[i][j]);
            if (j < matrix->cols - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i < matrix->rows - 1) {
            printf(",\n");
        }
    }
    printf("]\n");
}

// calculate the index of the maximum value in a vector
int vector_max_index(Matrix *matrix) {
    int max_index = 0;
    for (int i = 0; i < matrix->rows; i++) {
        if (matrix->values[i][0] > matrix->values[max_index][0]) {
            max_index = i;
        }
    }
    return max_index;
}


// fill matrix with random values
void randomize_matrix(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->values[i][j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        }
    }
}

void element_wise_multiply(Matrix *m1, Matrix *m2, Matrix *result) {
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        printf("Error: Matrix dimensions do not match!\n");
        return;
    }
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->values[i][j] = m1->values[i][j] * m2->values[i][j];
        }
    }
}

void copy_matrix(Matrix *matrix, Matrix *destination) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            destination->values[i][j] = matrix->values[i][j];
        }
    }
}