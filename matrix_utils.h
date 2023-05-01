//
// Created by szymc on 01.05.2023.
//

#ifndef SEM2LAB2_MATRIX_UTILS_H
#define SEM2LAB2_MATRIX_UTILS_H

struct Matrix {
    int rows;
    int cols;
    double **values;
} typedef Matrix;

void fill_matrix(Matrix *matrix, double value);

Matrix *create_matrix(int rows, int cols);

void free_matrix(Matrix *matrix);

void matrix_multiply(Matrix *m1, Matrix *m2, Matrix *result);

void matrix_add(Matrix *m1, Matrix *m2, Matrix *result);

void matrix_subtract(Matrix *m1, Matrix *m2, Matrix *result);

Matrix matrix_transpose(Matrix *matrix);

void matrix_add_scalar(Matrix *matrix, double scalar, Matrix *result);

void matrix_apply_function(Matrix *matrix, double (*function)(double), Matrix *result);

void print_matrix(Matrix *matrix);

int vector_max_index(Matrix *matrix);

void randomize_matrix(Matrix *matrix);

#endif //SEM2LAB2_MATRIX_UTILS_H
