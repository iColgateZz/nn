#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.1
#define EPOCHS 1000000

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

void train(double weights[3][3], int inputs[4][2], int expected_outputs[4]) {
    double sums[3];
    double neuron_outputs[3];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int example = 0; example < 4; example++) {
            // Forward pass
            sums[0] = weights[0][0] * inputs[example][0] + weights[0][1] * inputs[example][1] + weights[0][2];
            neuron_outputs[0] = sigmoid(sums[0]);

            sums[1] = weights[1][0] * inputs[example][0] + weights[1][1] * inputs[example][1] + weights[1][2];
            neuron_outputs[1] = sigmoid(sums[1]);

            sums[2] = weights[2][0] * neuron_outputs[0] + weights[2][1] * neuron_outputs[1] + weights[2][2];
            neuron_outputs[2] = sigmoid(sums[2]);

            double error = expected_outputs[example] - neuron_outputs[2];
            double delta = error * sigmoid_derivative(sums[2]);

            weights[2][2] += delta;
            weights[2][0] += delta * neuron_outputs[0];
            weights[2][1] += delta * neuron_outputs[1];

            weights[0][0] += delta * weights[2][0] * sigmoid_derivative(sums[0]) * inputs[example][0];
            weights[0][1] += delta * weights[2][0] * sigmoid_derivative(sums[0]) * inputs[example][1];
            weights[0][2] += delta * weights[2][0] * sigmoid_derivative(sums[0]);

            weights[1][0] += delta * weights[2][1] * sigmoid_derivative(sums[1]) * inputs[example][0];
            weights[1][1] += delta * weights[2][1] * sigmoid_derivative(sums[1]) * inputs[example][1];
            weights[1][2] += delta * weights[2][1] * sigmoid_derivative(sums[1]);
        }
    }
}

int main(void) {
    double weights[3][3];
    int inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int outputs[4] = {0, 1, 1, 0};

    train(weights, inputs, outputs);

    for (int i = 0; i < 4; i++) {
        double hidden_layer[2];
        for (int j = 0; j < 2; j++) {
            hidden_layer[j] = sigmoid(inputs[i][0] * weights[j][0] + inputs[i][1] * weights[j][1] + weights[j][2]);
        }
        double output = sigmoid(hidden_layer[0] * weights[2][0] + hidden_layer[1] * weights[2][1] + weights[2][2]);
        printf("Input: (%d, %d) -> Output: %f\n", inputs[i][0], inputs[i][1], output);
    }
    return 0;
}
