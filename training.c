//
// Created by szymc on 01.05.2023.
//

#include "training.h"
#include <stdio.h>
#include <stdlib.h>
#include "matrix_utils.h"


// create training values packet
TrainingDataPacket *create_training_data_packet(int size_of_input, int size_of_target) {
    TrainingDataPacket *training_data_packet = malloc(sizeof(TrainingDataPacket));
    training_data_packet->input = create_matrix(size_of_input, 1);
    training_data_packet->target = create_matrix(size_of_target, 1);
    fill_matrix(training_data_packet->target, 0);
    return training_data_packet;
}

//read training values to a list of packets from training.txt file
//template of the file to be read
// 0.1 0.2 0.3 14
// 0.4 0.5 0.6 15
// 0.7 0.8 0.9 12
// 0.1 0.3 0.3 14
// where 0.1 0.2 0.3 is input and 14 is one hot encoded target
// the function also normalizes the input values if needed by dividing by max_value_of_input
TrainingDataPacket **read_training_data(char *file_name, int lenght_of_training_data, int packet_size, int output_size,double max_value_of_input) {
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Error: Could not open file!\n");
        return NULL;
    }
    TrainingDataPacket **training_data = malloc(lenght_of_training_data * sizeof(TrainingDataPacket *));
    for (int i = 0; i < lenght_of_training_data; i++) {
        training_data[i] = create_training_data_packet(packet_size, output_size);
    }
    for (int i = 0; i < lenght_of_training_data; i++) {
        for (int j = 0; j < packet_size; j++) {
            double value;
            fscanf(file, "%lf", &value);
            training_data[i]->input->values[j][0] = value/max_value_of_input;
        }
        int target_index;
        fscanf(file, "%d", &target_index);
        training_data[i]->target->values[target_index][0] = 1;
    }
    fclose(file);
    return training_data;
}
