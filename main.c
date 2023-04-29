#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

void free_matrix(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
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
void matrix_add_scalar(Matrix *matrix, double scalar, Matrix *result) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[i][j] += scalar;
        }
    }
}

// apply function to each element of matrix
void matrix_apply_function(Matrix *matrix, double (*function)(double), Matrix *result) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[i][j] = function(matrix->data[i][j]);
        }
    }
}

void print_matrix(Matrix *matrix) {
    printf("[");
    for (int i = 0; i < matrix->rows; i++) {
        printf("[");
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f", matrix->data[i][j]);
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
void *softmax(Matrix *matrix, Matrix *result) {

    //calculate sum for normalization
    double sum = 0;
    for (int i = 0; i < matrix->rows; i++) {
        sum += exp(matrix->data[i][0]);
    }

    //calculate softmax
    for (int i = 0; i < matrix->rows; i++) {
        result->data[i][0] = exp(matrix->data[i][0]) / sum;
    }
}

// fill matrix with random values
void randomize_matrix(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        }
    }
}

struct Layer {
    int index;
    int input_size;
    int layer_size;
    int output_size;
    Matrix *input;
    Matrix *output;
    Matrix *weights;
    double bias;
    Matrix *weighted_sums;
    Matrix *activations;
} typedef Layer;

// define network struct
struct network {
    Layer **layers;
    int number_of_layers;
} typedef Network;

// create network
Network *create_network(int number_of_layers, int *layer_sizes) {
    // allocate memory for network
    Network *network = malloc(sizeof(Network));

    network->number_of_layers = number_of_layers;
    // allocate memory for layers
    network->layers = malloc(number_of_layers * sizeof(Layer *));
    // create layers
    for (int i = 0; i < number_of_layers; i++) {
        network->layers[i] = malloc(sizeof(Layer));
        network->layers[i]->index = i;
        network->layers[i]->layer_size = layer_sizes[i];
        if (i == 0) {//input layer
            network->layers[i]->input_size = 0;
            network->layers[i]->input = create_matrix(0, 0);
        } else {
            network->layers[i]->input_size = layer_sizes[i - 1];
            network->layers[i]->input = network->layers[i - 1]->activations;
        }
        if (i == number_of_layers - 1) {//output layer
            network->layers[i]->output_size = 0;
        } else { //hidden layers
            network->layers[i]->output_size = layer_sizes[i + 1];
        }

        network->layers[i]->weights = create_matrix(network->layers[i]->layer_size, network->layers[i]->input_size);
        network->layers[i]->weighted_sums = create_matrix(network->layers[i]->layer_size, 1);
        network->layers[i]->activations = create_matrix(network->layers[i]->layer_size, 1);

        //randomize weights
        randomize_matrix(network->layers[i]->weights);

        //initialize bias
        network->layers[i]->bias = 0;
    }
    return network;
}


// propagate forward through the network
void propagate_forward(Network *network) {
    //calculate weighted sums and activations for hidden layers and output layer (exclude input layer i=1)
    for (int i = 1; i < network->number_of_layers - 1; i++) {
        //calculate weighted sums
        matrix_multiply(network->layers[i]->weights, network->layers[i]->input, network->layers[i]->weighted_sums);
        //add bias
        matrix_add_scalar(network->layers[i]->weighted_sums, network->layers[i]->bias,
                          network->layers[i]->weighted_sums);

        //calculate activations
        matrix_apply_function(network->layers[i]->weighted_sums, ReLU, network->layers[i]->activations);
        //set activations as input for next layer
        network->layers[i + 1]->input = network->layers[i]->activations;
    }
    //calculate output layer weighted sums
    matrix_multiply(network->layers[network->number_of_layers - 1]->weights,
                    network->layers[network->number_of_layers - 1]->input,
                    network->layers[network->number_of_layers - 1]->weighted_sums);
    //add bias
    matrix_add_scalar(network->layers[network->number_of_layers - 1]->weighted_sums,
                      network->layers[network->number_of_layers - 1]->bias,
                      network->layers[network->number_of_layers - 1]->weighted_sums);
    //apply softmax
    softmax(network->layers[network->number_of_layers - 1]->weighted_sums,
            network->layers[network->number_of_layers - 1]->activations);
}

// print the entire structure of the network
void print_network(Network *network) {
    for (int i = 0; i < network->number_of_layers; i++) {
        printf("Layer %d\n", i);
        printf("Input size: %d\n", network->layers[i]->input_size);
        printf("Layer size: %d\n", network->layers[i]->layer_size);
        printf("Output size: %d\n", network->layers[i]->output_size);
        printf("Input:\n");
        print_matrix(network->layers[i]->input);
        printf("Weights:\n");
        print_matrix(network->layers[i]->weights);
        printf("Weighted sums:\n");
        print_matrix(network->layers[i]->weighted_sums);
        printf("Activations:\n");
        print_matrix(network->layers[i]->activations);
        printf("\n");
    }
}

double calculate_loss(Matrix *output_layer, Matrix *target) {
    double loss = 0;
    for (int i = 0; i < output_layer->rows; i++) {
        loss += pow(output_layer->data[i][0] - target->data[i][0], 2);
    }
    return loss;
}

double calculate_average_loss(Network *network, TrainingDataPacket **training_data, int length_of_training_data) {
    double loss = 0;
    for (int j = 0; j < length_of_training_data; j++) {
        //free the empty matrix and assign input to the activations of the input layer
        network->layers[0]->activations->data[0][0] = training_data[j]->input->data[0][0];
        network->layers[0]->activations->data[1][0] = training_data[j]->input->data[1][0];
        network->layers[0]->activations->data[2][0] = training_data[j]->input->data[2][0];
        //propagate forward
        propagate_forward(network);
        loss += calculate_loss(network->layers[network->number_of_layers - 1]->activations, training_data[j]->target);
    }
    return loss / length_of_training_data;
}

int main() {
    srand(time(NULL));
    Network *network = create_network(4, (int[]) {3, 5, 20, 16});

    TrainingDataPacket **training_data = read_training_data(
            "C:\\Users\\szymc\\CLionProjects\\Sem2Lab2\\training_data.txt",
            250);


    network->layers[0]->activations->data[0][0] = training_data[0]->input->data[0][0];
    network->layers[0]->activations->data[1][0] = training_data[0]->input->data[1][0];
    network->layers[0]->activations->data[2][0] = training_data[0]->input->data[2][0];
    print_matrix(network->layers[0]->activations);
    propagate_forward(network);
    print_matrix(network->layers[0]->activations);
    print_network(network);

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