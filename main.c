#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// define matrix struct
struct Matrix {
    int rows;
    int cols;
    double **data;
} typedef Matrix;

// create matrix with given dimensions
Matrix *create_matrix(int rows, int cols) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    // allocate memory for matrix
    matrix->data = malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        matrix->data[i] = malloc(cols * sizeof(double));
    }
    return matrix;
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
                sum += m1->data[i][k] * m2->data[k][j];
            }
            result->data[i][j] = sum;
        }
    }
}

// add two matrices
void matrix_add(Matrix *m1, Matrix *m2, Matrix *result) {
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        printf("Error: Matrix dimensions do not match!\n");
        return;
    }
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
    }
}

// apply function to each element of matrix
void apply_function(Matrix *matrix, double (*function)(float)) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] = function(matrix->data[i][j]);
        }
    }
}

// ReLU activation function
double ReLU(float x) {
    return fmax(0, x);
};

//normalized softmax function for the output layer
Matrix *softmax(Matrix *matrix) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);

    //calculate sum for normalization
    double sum = 0;
    for (int i = 0; i < matrix->rows; i++) {
        sum += exp(matrix->data[i][0]);
    }

    //calculate softmax
    for (int i = 0; i < matrix->rows; i++) {
        result->data[i][0] = exp(matrix->data[i][0]) / sum;
    }
    return result;
}

// fill matrix with random values
void randomize_matrix(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        }
    }
}

// define network struct
struct network {
    Matrix *input_layer;
    Matrix *weights1;
    Matrix *hidden_layer1;
    Matrix *bias1;
    Matrix *weights2;
    Matrix *hidden_layer2;
    Matrix *bias2;
    Matrix *weights3;
    Matrix *output_layer;
} typedef Network;

// create network
Network *create_network() {
    Network *network = malloc(sizeof(Network));
    network->input_layer = create_matrix(3, 1);
    network->weights1 = create_matrix(16, 3);
    network->hidden_layer1 = create_matrix(16, 1);
    network->bias1 = create_matrix(16, 1);
    network->weights2 = create_matrix(16, 16);
    network->hidden_layer2 = create_matrix(16, 1);
    network->bias2 = create_matrix(16, 1);
    network->weights3 = create_matrix(12, 16);
    network->output_layer = create_matrix(12, 1);
    return network;
}

//add bias vectors
Matrix propagate_forward(Network *network) {
    Matrix *hidden_layer1 = create_matrix(16, 1);
    Matrix *hidden_layer2 = create_matrix(16, 1);
    Matrix *output_layer = create_matrix(12, 1);

    matrix_multiply(network->weights1, network->input_layer, hidden_layer1);
    matrix_add(hidden_layer1, network->bias1, hidden_layer1);
    apply_function(hidden_layer1, ReLU);

    matrix_multiply(network->weights2, hidden_layer1, hidden_layer2);
    matrix_add(hidden_layer2, network->bias2, hidden_layer2);
    apply_function(hidden_layer2, ReLU);

    matrix_multiply(network->weights3, hidden_layer2, output_layer);

    output_layer = softmax(output_layer);

    return *output_layer;
}

double calculate_loss(Matrix *output_layer, Matrix *target) {
    double loss = 0;
    for (int i = 0; i < output_layer->rows; i++) {
        loss += pow(output_layer->data[i][0] - target->data[i][0], 2);
    }
    return loss;
}


double propagate_backward(Network *network) {
    //TODO implement backpropagation
    return 0;
}




int main() {
    Network *network = create_network();

    //weights
    randomize_matrix(network->weights1);
    randomize_matrix(network->weights2);
    randomize_matrix(network->weights3);

    //bias
    randomize_matrix(network->bias1);
    randomize_matrix(network->bias2);

    //input data
    network->input_layer->data[0][0] = 0.1; //R
    network->input_layer->data[1][0] = 0.2; //G
    network->input_layer->data[2][0] = 0.3; //B

    //print output layer
    Matrix out = propagate_forward(network);
    for (int i = 0; i < 12; i++) {
        printf("%f\n", out.data[i][0]);
    }

    //print output layer sum
    double sum = 0;
    for (int i = 0; i < 12; i++) {
        sum += out.data[i][0];
    }
    printf("Sum: %f\n", sum);

    return 0;
}
// 10 neuronów wejściowych
// rzędy w kalawieturze
// połówki alfabetu


// klasyfikator kolorów
// 3 nauronowe wejście jako rgb kolor
// klasyfikacja na 12 kolorów

//zwalniaj pamięć