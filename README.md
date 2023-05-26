
# Sieć Neuronowa
"Uniwersalny" kod do trenowania sieci neuronowych
sieć została wytrenowana na wygenerowanym przeze mnie datasecie i kategoryzuje ona kolor z przestrzeni CIELab i nadaje mu nazwę (1/16)
## Generowane danych treningowych
Do generowania danych treningowych wykorzystane zostały manulanie skategoryzowane przeze mnie kolory:

<img src="https://github.com/MRcoin2/neural-net-in-C/assets/47926022/33ee0f08-83b0-433e-917c-2803eab02d47" height="400">
<img src="https://github.com/MRcoin2/neural-net-in-C/assets/47926022/58488fb2-2d7c-432c-92d8-e1fb2aa08af1" height="400">

Kolory były wybierane za pomocą **./.../human_chosen_lab.py** następnie aby wygenerować więcej danych wyliczana była odległość eukliesowa wylosowanego koloru do kolorów manualnie skategoryzowanych i przypisywana mu była nazwa najbliższego koloru.
```Python
def assign_category():
    random_color = rgb_to_lab(random_rgb())
    distances = []
    for color in colors:
        distances.append((distance(color[0], (random_color[0],random_color[1],random_color[2])), color[1]))
    distances.sort()
    category = distances[0][1]
    save_category(random_color, category)
```
Skutkowało to wygenerowaniem dobrej jakości danych do zadanego problemu ze względu na reprezentację kolorów w przestrzeni CIELab która dobrze odwzorowuje bliskość kolorów dla ludzkiego oka.

<img src="https://github.com/MRcoin2/neural-net-in-C/assets/47926022/294e4f5e-ecfa-4f8d-bbf8-1d3b6c68b5a9" height="400">

### Struktura danych treningowych:
| L [value/100]      | a [value/128]        | b [value/128]       | Category index |
|--------------------|----------------------|---------------------|----------------|
| 0.5107259339648228 | 0.3787139683595858   | 0.4510567438273844  | 6              |
| 0.6836574167142221 | 0.25010620322890675  | 0.37241226255254734 | 3              |
| 0.849179874445219  | -0.1326582048791153  | 0.655411046181078   | 8              |
| 0.4753758136031553 | -0.32550868314784426 | 0.1890248299851423  | 10             |
| ...                | ...                  | ...                 | ...            |

Ostatecznie dane miały 60 000 wierszy.

## Struktura sieci
![nn](https://github.com/MRcoin2/neural-net-in-C/assets/47926022/c222ac82-4558-4c30-ba2a-1c6522119855)

Wybrano sieć o strukturze (3,5,10,20,16)

W sieci została użyta funkcja aktywacyjna Leaky ReLU z współczynnikami 0.1 [x<0] i 1 [x>=0]:
```C
#define ReLU_A 0.1
#define ReLU_B 1
double ReLU(double x) {
    if (x < 0) {
        return x * ReLU_A;
    }
    return x * ReLU_B;
};
```
## Trenowanie
Sieć była trenowana stochastycznie używająć algorytmu backpropagation.
```C
//split the training data into packets randomly and train on that
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
            printf("success rate: %.2f%%\n", success_rate * 100);
            printf("\033[0m");
            if (loss > last_loss) {
                learning_rate *= 0.96;
                printf("learning rate: %f\n", learning_rate);
            }
            last_loss = loss;
        }
    }
}
```
Fragment logu z trenowania:
```console
____________________________________________________
finished: 0.00%
avg loss: 0.964217
success rate: 11.38%
____________________________________________________
finished: 1.00%
avg loss: 0.592193
success rate: 54.56%
____________________________________________________
finished: 2.00%
avg loss: 0.563284
success rate: 55.70%
____________________________________________________
finished: 3.00%
avg loss: 0.472964
success rate: 66.41%
____________________________________________________
finished: 4.00%
avg loss: 0.413702
success rate: 69.84%
____________________________________________________
finished: 5.00%
avg loss: 0.313641
success rate: 77.64%
____________________________________________________
finished: 6.00%
avg loss: 0.344279
success rate: 75.44%
learning rate: 0.096000
____________________________________________________
finished: 7.00%
avg loss: 0.278393
success rate: 80.20%
____________________________________________________
finished: 8.00%
avg loss: 0.342126
success rate: 75.17%
learning rate: 0.092160
____________________________________________________
finished: 9.00%
avg loss: 0.283430
success rate: 79.68%
____________________________________________________
finished: 10.00%
avg loss: 0.351548
success rate: 75.01%
learning rate: 0.088474
____________________________________________________
finished: 11.00%
avg loss: 0.253200
success rate: 81.95%
____________________________________________________
finished: 12.00%
avg loss: 0.277222
success rate: 80.01%
learning rate: 0.084935
____________________________________________________
```
## Interakcja z siecią
W celu łatwiejszej interakcji z siecią napisano program ./color_picker.py
![image](https://github.com/MRcoin2/neural-net-in-C/assets/47926022/0ca740aa-828c-4912-9274-25d15f8c375f)

Przykładowy output sieci:
```console
Enter 3 numbers: 0.47404308609286433 -0.07914935990061293 -0.24612795799576342
output: blue
[[0.000000],
[0.000072],
[0.000000],
[0.000000],
[0.000000],
[0.000000],
[0.000000],
[0.000000],
[0.000000],
[0.000000],
[0.000000],
[0.000003],
[0.000002],
[0.956414],
[0.043433],
[0.000076]]
want to continue? (Y/n)
```

