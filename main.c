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

void matrix_subtract(Matrix *m1, Matrix *m2, Matrix *result) {
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        printf("Error: Matrix dimensions do not match!\n");
        return;
    }
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }
}

Matrix matrix_transpose(Matrix *matrix) {
    Matrix *result = create_matrix(matrix->cols, matrix->rows);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[j][i] = matrix->data[i][j];
        }
    }
    return *result;
}

//add a number to a matrix
void matrix_add_number(Matrix *matrix, double number) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] += number;
        }
    }
}

// apply function to each element of matrix
void matrix_apply_function(Matrix *matrix, double (*function)(double)) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] = function(matrix->data[i][j]);
        }
    }
}

struct TrainingDataPacket {
    Matrix *input;
    Matrix *target;
} typedef TrainingDataPacket;

//create training data
TrainingDataPacket *create_training_data() {
    TrainingDataPacket *training_data = malloc(sizeof(TrainingDataPacket));
    training_data->input = create_matrix(3, 1);
    training_data->target = create_matrix(16, 1);
    return training_data;
}

//read training data to a list of packets from training.txt file
//template of the file to be read
// 0.1 0.2 0.3 14
// 0.4 0.5 0.6 15
// 0.7 0.8 0.9 12
// 0.1 0.3 0.3 14
TrainingDataPacket **read_training_data(char file_name[], int lenght_of_training_data) {
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Error: Could not open file!\n");
        return NULL;
    }
    TrainingDataPacket **training_data = malloc(lenght_of_training_data * sizeof(TrainingDataPacket *));
    for (int i = 0; i < lenght_of_training_data; i++) {
        training_data[i] = create_training_data();
    }
    for (int i = 0; i < lenght_of_training_data; i++) {
        fscanf(file, "%lf %lf %lf",
               &training_data[i]->input->data[0][0],
               &training_data[i]->input->data[1][0],
               &training_data[i]->input->data[2][0]);
        int target_index;
        fscanf(file, "%d", &target_index);
        training_data[i]->target->data[target_index][0] = 1;
    }
    fclose(file);
    return training_data;
}

// ReLU activation function
double ReLU(double x) {
    return fmax(0, x);
};

double dReLU(double x) {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

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

struct Hidden_layer {
    int index;
    int input_size;
    int output_size;
    Matrix *input;
    Matrix *output;
    Matrix *weights;
    double bias;
    Matrix *weighted_sums;
    Matrix *activations;
} typedef Hidden_layer;

// define network struct
struct network {
    Matrix *input_layer;
    Matrix *weights1;
    Matrix *hidden_layer1;
    Matrix *weights2;
    Matrix *hidden_layer2;
    Matrix *weights3;
    Matrix *output_layer;
    Matrix *bias;
} typedef Network;

// create network
Network *create_network() {
    Network *network = malloc(sizeof(Network));
    network->input_layer = create_matrix(3, 1);

    network->weights1 = create_matrix(5, 3);
    network->hidden_layer1 = create_matrix(5, 1);

    network->weights2 = create_matrix(20, 5);
    network->hidden_layer2 = create_matrix(20, 1);

    network->weights3 = create_matrix(16, 20);
    network->output_layer = create_matrix(16, 1);

    network->bias = create_matrix(3, 1);
    return network;
}

Matrix propagate_forward(Network *network) {
    matrix_multiply(network->weights1, network->input_layer, network->hidden_layer1);

    matrix_add_number(network->hidden_layer1, network->bias->data[0][0]);
    matrix_apply_function(network->hidden_layer1, ReLU);

    matrix_multiply(network->weights2, network->hidden_layer1, network->hidden_layer2);
    matrix_add_number(network->hidden_layer1, network->bias->data[1][0]);
    matrix_apply_function(network->hidden_layer2, ReLU);

    matrix_multiply(network->weights3, network->hidden_layer2, network->output_layer);
    matrix_add_number(network->hidden_layer1, network->bias->data[2][0]);

    network->output_layer = softmax(network->output_layer);

    return *network->output_layer;
}

double calculate_loss(Matrix *output_layer, Matrix *target) {
    double loss = 0;
    for (int i = 0; i < output_layer->rows; i++) {
        loss += pow(output_layer->data[i][0] - target->data[i][0], 2);
    }
    return loss;
}

Matrix propagate_backward(Network *network) {
    //TODO implement backpropagation

}

double calculate_average_loss(Network *network, TrainingDataPacket **training_data, int lenght_of_training_data) {
    double loss = 0;
    for (int j = 0; j < lenght_of_training_data; j++) {
        network->input_layer = training_data[j]->input;
        propagate_forward(network);
        loss += calculate_loss(network->output_layer, training_data[j]->target);
    }
    return loss / lenght_of_training_data;
}

void train(Network *network, TrainingDataPacket **training_data, int lenght_of_training_data) {
    for (int j = 0; j < lenght_of_training_data; j++) {
        network->input_layer = training_data[j]->input;
        propagate_forward(network);
        //TODO implement backpropagation
        //TODO implement weight update
    }
}

int main() {
    Network *network = create_network();

    //weights
    randomize_matrix(network->weights1);
    randomize_matrix(network->weights2);
    randomize_matrix(network->weights3);

    //bias
    randomize_matrix(network->bias);

    TrainingDataPacket **training_data = read_training_data(
            "C:\\Users\\szymc\\CLionProjects\\Sem2Lab2\\training_data.txt",
            250);
    propagate_forward(network);
    double loss = calculate_average_loss(network, training_data, 250);
    printf("avg loss: %f\n", loss);

    return 0;
}

// 10 neuronów wejściowych
// rzędy w kalawieturze
// połówki alfabetu


// klasyfikator kolorów
// 3 nauronowe wejście jako rgb kolor
// klasyfikacja na 12 kolorów

//zwalniaj pamięć