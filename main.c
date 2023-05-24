#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix_utils.h"
#include "training.h"


// ReLU activation function
double ReLU(double x) {
    if (x < 0) {
        return x*0.2;
    }
    return x;
};

//normalized softmax function for the output layer
void *softmax(Matrix *matrix, Matrix *result) {

    //calculate sum for normalization
    double sum = 0;
    for (int i = 0; i < matrix->rows; i++) {
        sum += exp(matrix->values[i][0]);
    }

    //calculate softmax
    for (int i = 0; i < matrix->rows; i++) {
        result->values[i][0] = exp(matrix->values[i][0]) / sum;
    }
}

struct Layer {
    int input_size;
    int layer_size;
    int output_size;
    Matrix *input;
    Matrix *weights;
    Matrix *delta_weights;
    Matrix *biases;
    Matrix *delta_biases;
    Matrix *weighted_sums;
    Matrix *activations;
    Matrix *deltas; //error of the layer
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
        network->layers[i]->delta_weights = create_matrix(network->layers[i]->layer_size,
                                                          network->layers[i]->input_size);
        network->layers[i]->weighted_sums = create_matrix(network->layers[i]->layer_size, 1);
        network->layers[i]->activations = create_matrix(network->layers[i]->layer_size, 1);
        network->layers[i]->deltas = create_matrix(network->layers[i]->layer_size, 1);

        //randomize weights
        randomize_matrix(network->layers[i]->weights);

        //initialize bias
        network->layers[i]->biases = create_matrix(network->layers[i]->layer_size, 1);
        network->layers[i]->delta_biases = create_matrix(network->layers[i]->layer_size, 1);
        randomize_matrix(network->layers[i]->biases);
    }
    return network;
}

void free_network(Network *network) {
    for (int i = 0; i < network->number_of_layers; i++) {
        free_matrix(network->layers[i]->weights);
        free_matrix(network->layers[i]->delta_weights);
        free_matrix(network->layers[i]->biases);
        free_matrix(network->layers[i]->delta_biases);
        free_matrix(network->layers[i]->weighted_sums);
        free_matrix(network->layers[i]->activations);
        free_matrix(network->layers[i]->deltas);
        free(network->layers[i]);
    }
    free(network->layers);
    free(network);
}

// propagate forward through the network
void propagate_forward(Network *network, Matrix *input) {
    //assign input to the activations of the input layer
    network->layers[0]->activations->values[0][0] = input->values[0][0];
    network->layers[0]->activations->values[1][0] = input->values[1][0];
    network->layers[0]->activations->values[2][0] = input->values[2][0];

    //calculate weighted sums and activations for hidden layers and output layer (exclude input layer i=1)
    for (int i = 1; i < network->number_of_layers - 1; i++) {
        //calculate weighted sums for hidden layers
        matrix_multiply(network->layers[i]->weights, network->layers[i]->input, network->layers[i]->weighted_sums);
        //add bias
        add_matrices(network->layers[i]->weighted_sums, network->layers[i]->biases,
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
    add_matrices(network->layers[network->number_of_layers - 1]->weighted_sums,
                 network->layers[network->number_of_layers - 1]->biases,
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
        loss += pow(output_layer->values[i][0] - target->values[i][0], 2);
    }
    return loss;
}

double calculate_average_loss(Network *network, TrainingDataPacket **training_data, int length_of_training_data) {
    double loss = 0;
    for (int j = 0; j < length_of_training_data; j++) {
        //propagate forward
        propagate_forward(network, training_data[j]->input);
        //calculate loss
        loss += calculate_loss(network->layers[network->number_of_layers - 1]->activations, training_data[j]->target);
    }
    //return average loss
    return loss / length_of_training_data;
}


// calculate average success rate of the network on the training values
// by comparing the output to the target and counting the number of correct outputs
double
calculate_average_success_rate(Network *network, TrainingDataPacket **training_data, int length_of_training_data) {
    double success_rate = 0;
    for (int j = 0; j < length_of_training_data; j++) {
        propagate_forward(network, training_data[j]->input);

        //check if the output matches the target
        if (vector_max_index(network->layers[network->number_of_layers - 1]->activations) ==
            vector_max_index(training_data[j]->target)) {
            success_rate++;
        }
    }
    return success_rate / length_of_training_data;
}

double output_node_cost_derivative(double output, double target) {
    return 2 * (output - target);
}

double ReLU_derivative(double x) {
    if (x > 0) {
        return 1;
    } else {
        return 0.2;
    }
}

void calculate_deltas_for_layer(Network *network, int layer_index, Matrix *target) {
    //calculate deltas for output layer
    //the equation is delta_i = 2(a_i - y_i) * ReLU'(z_i)
    if (layer_index == network->number_of_layers - 1) {
        //calculate deltas for each neuron in the output layer
        for (int i = 0; i < network->layers[layer_index]->layer_size; i++) {
            network->layers[layer_index]->deltas->values[i][0] = output_node_cost_derivative(
                    network->layers[layer_index]->activations->values[i][0], target->values[i][0]) *
                                                                 ReLU_derivative(
                                                                         network->layers[layer_index]->weighted_sums->values[i][0]);
        }
    } else {
        //calculate deltas for each neuron in the hidden layer
        //the equation is delta_i = sum(delta_j * w_ij) * ReLU'(z_i)
        for (int i = 0; i < network->layers[layer_index]->layer_size; i++) {
            double sum = 0;
            //calculate the sum of the deltas of the neurons in the next layer multiplied by their weights
            for (int j = 0; j < network->layers[layer_index + 1]->layer_size; j++) {
                sum += network->layers[layer_index + 1]->weights->values[j][i] *
                       network->layers[layer_index + 1]->deltas->values[j][0];
            }
            //multiply the sum by the derivative of the weighted sum of the current neuron
            network->layers[layer_index]->deltas->values[i][0] = sum *
                                                                 ReLU_derivative(
                                                                         network->layers[layer_index]->weighted_sums->values[i][0]);
        }
    }
}

// add the delta weights for a layer for the current pass
void add_delta_weights_for_layer(Network *network, int layer_index) {
    for (int i = 0; i < network->layers[layer_index]->layer_size; i++) {
        for (int j = 0; j < network->layers[layer_index]->input_size; j++) {
            network->layers[layer_index]->delta_weights->values[i][j] +=
                    network->layers[layer_index]->deltas->values[i][0] *
                    network->layers[layer_index]->input->values[j][0];
        }
    }
}

// add the delta biases for a layer for the current pass
void add_delta_biases_for_layer(Network *network, int layer_index) {
    for (int i = 0; i < network->layers[layer_index]->layer_size; i++) {
        network->layers[layer_index]->delta_biases->values[i][0] += network->layers[layer_index]->deltas->values[i][0];
    }
}

void average_delta_weights_for_layer(Network *network, int layer_index, int length_of_training_data) {
    for (int i = 0; i < network->layers[layer_index]->layer_size; i++) {
        for (int j = 0; j < network->layers[layer_index]->input_size; j++) {
            network->layers[layer_index]->delta_weights->values[i][j] /= length_of_training_data;
        }
    }
}

void average_delta_biases_for_layer(Network *network, int layer_index, int length_of_training_data) {
    for (int i = 0; i < network->layers[layer_index]->layer_size; i++) {
        network->layers[layer_index]->delta_biases->values[i][0] /= length_of_training_data;
    }
}

// move the weights in the direction of the -gradient in proportion to the learning rate
void update_weights_for_layer(Network *network, int layer_index, double learning_rate) {
    for (int i = 0; i < network->layers[layer_index]->layer_size; i++) {
        for (int j = 0; j < network->layers[layer_index]->input_size; j++) {
            network->layers[layer_index]->weights->values[i][j] -=
                    learning_rate * network->layers[layer_index]->delta_weights->values[i][j];
        }
    }
}

// move the biases in the direction of the -gradient in proportion to the learning rate
void update_biases_for_layer(Network *network, int layer_index, double learning_rate) {
    for (int i = 0; i < network->layers[layer_index]->layer_size; i++) {
        network->layers[layer_index]->biases->values[i][0] -=
                learning_rate * network->layers[layer_index]->delta_biases->values[i][0];
    }
}

// train the network on the given training data for the given number of epochs
void train_network(Network *network, TrainingDataPacket **training_data, int length_of_training_data, int epochs,
                   double learning_rate) {

    double last_loss = calculate_average_loss(network, training_data, length_of_training_data);
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < length_of_training_data; j++) {
            //propagate forward
            propagate_forward(network, training_data[j]->input);
            //calculate deltas for output layer
            calculate_deltas_for_layer(network, network->number_of_layers - 1, training_data[j]->target);
            //calculate deltas for hidden layers
            for (int k = network->number_of_layers - 2; k >= 0; k--) {
                calculate_deltas_for_layer(network, k, training_data[j]->target);
            }
            //calculate delta weights for output layer
            add_delta_weights_for_layer(network, network->number_of_layers - 1);
            //calculate delta weights for hidden layers
            for (int k = network->number_of_layers - 2; k >= 0; k--) {
                add_delta_weights_for_layer(network, k);
            }
            //calculate delta biases for output layer
            add_delta_biases_for_layer(network, network->number_of_layers - 1);
            //calculate delta biases for hidden layers
            for (int k = network->number_of_layers - 2; k >= 0; k--) {
                add_delta_biases_for_layer(network, k);
            }
        }

        //calculate the gradient for all data:
        //average delta weights for output layer
        average_delta_weights_for_layer(network, network->number_of_layers - 1, length_of_training_data);
        //average delta weights for hidden layers
        for (int k = network->number_of_layers - 2; k >= 0; k--) {
            average_delta_weights_for_layer(network, k, length_of_training_data);
        }
        //average delta biases for output layer
        average_delta_biases_for_layer(network, network->number_of_layers - 1, length_of_training_data);
        //average delta biases for hidden layers
        for (int k = network->number_of_layers - 2; k >= 0; k--) {
            average_delta_biases_for_layer(network, k, length_of_training_data);
        }

        //apply the gradient to the weights and biases:
        //update weights for output layer
        update_weights_for_layer(network, network->number_of_layers - 1, learning_rate);
        //update weights for hidden layers
        for (int k = network->number_of_layers - 2; k >= 0; k--) {
            update_weights_for_layer(network, k, learning_rate);
        }
        //update biases for output layer
        update_biases_for_layer(network, network->number_of_layers - 1, learning_rate);
        //update biases for hidden layers
        for (int k = network->number_of_layers - 2; k >= 0; k--) {
            update_biases_for_layer(network, k, learning_rate);
        }

        // calculate average loss and success rate every 10 epochs
        if (i % 100 == 0) {
            double loss = calculate_average_loss(network, training_data, length_of_training_data);
            printf("avg loss: %f\n", loss);
            double success_rate = calculate_average_success_rate(network, training_data, length_of_training_data);
            printf("success rate: %f\n", success_rate);
            if (loss > last_loss) {
                learning_rate *= 0.8;
                printf("learning rate: %f\n", learning_rate);
            }
            last_loss = loss;
        }
    }
}

// training function with no loss calculation for stochastic gradient descent
void train_network_no_loss_calc(Network *network, TrainingDataPacket **training_data, int length_of_training_data,
                                int epochs,
                                double learning_rate) {
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < length_of_training_data; j++) {
            //propagate forward
            propagate_forward(network, training_data[j]->input);
            //calculate deltas for output layer
            calculate_deltas_for_layer(network, network->number_of_layers - 1, training_data[j]->target);
            //calculate deltas for hidden layers
            for (int k = network->number_of_layers - 2; k >= 0; k--) {
                calculate_deltas_for_layer(network, k, training_data[j]->target);
            }
            //calculate delta weights for output layer
            add_delta_weights_for_layer(network, network->number_of_layers - 1);
            //calculate delta weights for hidden layers
            for (int k = network->number_of_layers - 2; k >= 0; k--) {
                add_delta_weights_for_layer(network, k);
            }
            //calculate delta biases for output layer
            add_delta_biases_for_layer(network, network->number_of_layers - 1);
            //calculate delta biases for hidden layers
            for (int k = network->number_of_layers - 2; k >= 0; k--) {
                add_delta_biases_for_layer(network, k);
            }
        }

        //calculate the gradient for all data:
        //average delta weights for output layer
        average_delta_weights_for_layer(network, network->number_of_layers - 1, length_of_training_data);
        //average delta weights for hidden layers
        for (int k = network->number_of_layers - 2; k >= 0; k--) {
            average_delta_weights_for_layer(network, k, length_of_training_data);
        }
        //average delta biases for output layer
        average_delta_biases_for_layer(network, network->number_of_layers - 1, length_of_training_data);
        //average delta biases for hidden layers
        for (int k = network->number_of_layers - 2; k >= 0; k--) {
            average_delta_biases_for_layer(network, k, length_of_training_data);
        }

        //apply the gradient to the weights and biases:
        //update weights for output layer
        update_weights_for_layer(network, network->number_of_layers - 1, learning_rate);
        //update weights for hidden layers
        for (int k = network->number_of_layers - 2; k >= 0; k--) {
            update_weights_for_layer(network, k, learning_rate);
        }
        //update biases for output layer
        update_biases_for_layer(network, network->number_of_layers - 1, learning_rate);
        //update biases for hidden layers
        for (int k = network->number_of_layers - 2; k >= 0; k--) {
            update_biases_for_layer(network, k, learning_rate);
        }
    }
}

//split the training data into 200 element packets randomly and train on that
void train_stochastic(Network *network, TrainingDataPacket **training_data, int length_of_training_data, int epochs,
                      int split_size,
                      double learning_rate) {
    double last_loss = calculate_average_loss(network, training_data, length_of_training_data);
    for (int i = 0; i < epochs; i++) {
        TrainingDataPacket **packets = malloc(sizeof(TrainingDataPacket *) * split_size);
        for (int j = 0; j < split_size; j++) {
            packets[j] = training_data[rand() % length_of_training_data];
        }
        train_network_no_loss_calc(network, packets, split_size, 1, learning_rate);
        free(packets);
        //calculate average loss and success rate every 10 epochs
        if (i % 100 == 0) {
            printf("____________________________________________________\n");
            //print finished percentage
            printf("finished: %.2f%%\n", (double) i / epochs * 100);
            //print the average loss and success rate
            double loss = calculate_average_loss(network, training_data, length_of_training_data);
            printf("avg loss: %f\n", loss);
            double success_rate = calculate_average_success_rate(network, training_data, length_of_training_data);
            //print succes rate in green color
            printf("\033[0;32m");
            printf("success rate: %.2f%%\n", success_rate*100);
            printf("\033[0m");
            if (loss > last_loss) {
                learning_rate *= 0.9;
                printf("learning rate: %f\n", learning_rate);
            }
            last_loss = loss;
        }
    }
}

//save the network configuration and the weights and biases to a file
void save_network_to_file(Network *network) {
    FILE *file = fopen("C:\\Users\\Szymon\\CLionProjects\\Sem2Lab2\\network.txt", "w");
    fprintf(file, "%d\n", network->number_of_layers);
    for (int i = 0; i < network->number_of_layers; i++) {
        fprintf(file, "%d\n", network->layers[i]->layer_size);
    }
    for (int i = 1; i < network->number_of_layers; i++) {
        for (int j = 0; j < network->layers[i]->layer_size; j++) {
            for (int k = 0; k < network->layers[i]->input_size; k++) {
                fprintf(file, "%f\n", network->layers[i]->weights->values[j][k]);
            }
        }
    }
    for (int i = 1; i < network->number_of_layers; i++) {
        for (int j = 0; j < network->layers[i]->layer_size; j++) {
            fprintf(file, "%f\n", network->layers[i]->biases->values[j][0]);
        }
    }
    fclose(file);
}

//load the network configuration and the weights and biases from a file
Network *load_network_from_file(char file_name[]) {
    FILE *file = fopen(file_name, "r");
    int number_of_layers;
    fscanf(file, "%d", &number_of_layers);
    int *layer_sizes = malloc(sizeof(int) * number_of_layers);
    for (int i = 0; i < number_of_layers; i++) {
        fscanf(file, "%d", &layer_sizes[i]);
    }
    Network *network = create_network(number_of_layers, layer_sizes);
    for (int i = 1; i < network->number_of_layers; i++) {
        for (int j = 0; j < network->layers[i]->layer_size; j++) {
            for (int k = 0; k < network->layers[i]->input_size; k++) {
                fscanf(file, "%lf", &network->layers[i]->weights->values[j][k]);
            }
        }
    }
    for (int i = 1; i < network->number_of_layers; i++) {
        for (int j = 0; j < network->layers[i]->layer_size; j++) {
            fscanf(file, "%lf", &network->layers[i]->biases->values[j][0]);
        }
    }
    fclose(file);
    return network;
}

//function for using the network
void use_network(Network *network, Matrix *input) {
    propagate_forward(network, input);
    char colors[16][25] = {"white", "gray", "black", "red", "pink", "dark red", "orange",
                           "brown", "yellow", "green","dark green",
                           "teal", "light blue", "blue", "dark blue", "purple"};
    printf("output: %s\n",
           colors[vector_max_index(network->layers[network->number_of_layers - 1]->activations)]);
}

int main() {
    //seed the random number generator
    srand(time(NULL));
    //interface to choose to create a new network or load a file
//    printf("1. Create new network\n2. Load network from file\n");
//    int choice;
//    scanf("%d", &choice);
//    if (choice == 1) {
//        printf("Enter number of layers: ");
//        int number_of_layers;
//        scanf("%d", &number_of_layers);
//        int *layer_sizes = malloc(sizeof(int) * number_of_layers);
//        for (int i = 0; i < number_of_layers; i++) {
//            printf("Enter size of layer %d: ", i);
//            scanf("%d", &layer_sizes[i]);
//        }
//        Network *network = create_network(number_of_layers, layer_sizes);
//        //ask user for training data file
//        printf("Enter training data file name: ");
//        char training_data_file_name[100];
//        scanf("%s", training_data_file_name);
//        //ask how much training data to use
//        printf("Enter number of training data to use: ");
//        int lenght_of_training_data;
//        scanf("%d", &lenght_of_training_data);
//        TrainingDataPacket **training_data = read_training_data(
//                training_data_file_name,
//                lenght_of_training_data);
//        //ask for epochs
//        printf("Enter number of epochs: ");
//        int epochs;
//        scanf("%d", &epochs);
//        //ask for batch size
//        printf("Enter batch size: ");
//        int batch_size;
//        scanf("%d", &batch_size);
//        //ask for learning rate
//        printf("Enter learning rate: ");
//        double learning_rate;
//        scanf("%lf", &learning_rate);
//        //train the network
//        train_stochastic(network, training_data, lenght_of_training_data, epochs, batch_size, learning_rate);
//        save_network_to_file(network);
//        return 0;
//    } else if (choice == 2) {
//        //ask for file name
//        printf("Enter file name: ");
//        char file_name[100];
//        scanf("%s", file_name);
//        Network *network = load_network_from_file(file_name);
//        //get input from user
//        Matrix *input = create_matrix(3, 1);
//        while (1){
//        printf("Enter 3 numbers: ");
//        scanf("%lf %lf %lf", &input->values[0][0], &input->values[1][0], &input->values[2][0]);
//        use_network(network, input);
//        print_matrix(network->layers[network->number_of_layers - 1]->activations);}
//        free_matrix(input);
//        return 0;
//    }




    //create the network
    Network *network = create_network(5, (int[]) {3, 10,16, 20, 16});
//    Network *network = load_network_from_file("C:\\Users\\szymc\\CLionProjects\\Sem2Lab2\\network.txt");
    //read the training data
    TrainingDataPacket **training_data = read_training_data(
            "C:\\Users\\Szymon\\CLionProjects\\Sem2Lab2\\training_lab_v2.txt",
            60000);

    //train the network
    //train_network(network, training_data, 60000, 2000, 1);
    train_stochastic(network, training_data, 60000, 6000, 600, 0.1);
    save_network_to_file(network);

    //get input from user
    Matrix *input = create_matrix(3, 1);
    printf("Enter 3 numbers: ");
    scanf("%lf %lf %lf", &input->values[0][0], &input->values[1][0], &input->values[2][0]);//TODO implement this on the copy matrix function
    propagate_forward(network, input);
    free_matrix(input);

    //print the output
    print_matrix(network->layers[network->number_of_layers - 1]->activations);
    //print the index of the largest value
    printf("index of the largest value: %d\n",
           vector_max_index(network->layers[network->number_of_layers - 1]->activations));

    //free the network
    free_network(network);

    // free the training data
    for (int i = 0; i < 60000; i++) {
        free_matrix(training_data[i]->input);
        free_matrix(training_data[i]->target);
        free(training_data[i]);
    }
    free(training_data);
    return 0;
}

// 10 neuronów wejściowych
// rzędy w kalawieturze
// połówki alfabetu


// klasyfikator kolorów
// 3 nauronowe wejście jako rgb kolor
// klasyfikacja na 12 kolorów

//zwalniaj pamięć