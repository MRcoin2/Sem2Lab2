//
// Created by szymc on 01.05.2023.
//

#ifndef SEM2LAB2_TRAINING_H
#define SEM2LAB2_TRAINING_H

struct TrainingDataPacket {
    Matrix *input;
    Matrix *target;
} typedef TrainingDataPacket;

//create training values
TrainingDataPacket *create_training_data_packet();

//read training values to a list of packets from training.txt file
//template of the file to be read
// 0.1 0.2 0.3 14
// 0.4 0.5 0.6 15
// 0.7 0.8 0.9 12
// 0.1 0.3 0.3 14
TrainingDataPacket **read_training_data(char file_name[], int lenght_of_training_data);

#endif //SEM2LAB2_TRAINING_H
