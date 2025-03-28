#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LRATE 0.01
#define EPOCHS 1000000

long double activation(long double x) {
    return 1.0 / (1.0 + exp(-x));
}

long double act_deriv(long double x) {
    return activation(x) * (1 - activation(x));
}

void fill_weights(long double weights[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            weights[i][j] = ((long double)rand() / RAND_MAX) * 0.1 - 0.05;
        }
    }
}

void fit(long double weights[3][3], int inputs[4][2], int expected_outputs[4]) {
    long double sums[3];
    long double neuron_outputs[3];
    fill_weights(weights);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int current = 0; current < 4; current++) {
            // Forward pass
            sums[0] = weights[0][0] * inputs[current][0] + weights[0][1] * inputs[current][1] + weights[0][2];
            neuron_outputs[0] = activation(sums[0]);

            sums[1] = weights[1][0] * inputs[current][0] + weights[1][1] * inputs[current][1] + weights[1][2];
            neuron_outputs[1] = activation(sums[1]);

            sums[2] = weights[2][0] * neuron_outputs[0] + weights[2][1] * neuron_outputs[1] + weights[2][2];
            neuron_outputs[2] = activation(sums[2]);

            // Calculate deltas
            long double error = expected_outputs[current] - neuron_outputs[2];
            long double delta = error * act_deriv(sums[2]);
            long double hidden_delta[2];
            hidden_delta[0] = delta * weights[2][0] * act_deriv(sums[0]);
            hidden_delta[1] = delta * weights[2][1] * act_deriv(sums[1]);

            // Update weights for hidden layer
            weights[0][0] += LRATE * hidden_delta[0] * inputs[current][0];
            weights[0][1] += LRATE * hidden_delta[0] * inputs[current][1];
            weights[0][2] += LRATE * hidden_delta[0];

            weights[1][0] += LRATE * hidden_delta[1] * inputs[current][0];
            weights[1][1] += LRATE * hidden_delta[1] * inputs[current][1];
            weights[1][2] += LRATE * hidden_delta[1];

            // Update weights for output layer
            weights[2][0] += LRATE * delta * neuron_outputs[0];
            weights[2][1] += LRATE * delta * neuron_outputs[1];
            weights[2][2] += LRATE * delta;
        }
    }
}

void predict(long double weights[3][3], int inputs[4][2], long double predictions[4]) {
    long double sums[3];
    long double neuron_outputs[2];
    for (int current = 0; current < 4; current++) {
        sums[0] = weights[0][0] * inputs[current][0] + weights[0][1] * inputs[current][1] + weights[0][2];
        neuron_outputs[0] = activation(sums[0]);

        sums[1] = weights[1][0] * inputs[current][0] + weights[1][1] * inputs[current][1] + weights[1][2];
        neuron_outputs[1] = activation(sums[1]);

        sums[2] = weights[2][0] * neuron_outputs[0] + weights[2][1] * neuron_outputs[1] + weights[2][2];
        predictions[current] = activation(sums[2]);
    }
}

int main(void) {
    long double weights[3][3];
    int inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int outputs[4] = {0, 1, 1, 0};
    long double predictions[4];

    fit(weights, inputs, outputs);
    predict(weights, inputs, predictions);

    for (int i = 0; i < 4; i++)
        printf("Input: (%d, %d); Expected: %d; Predicted: %.4Lf\n",
                inputs[i][0], inputs[i][1], outputs[i], predictions[i]);
    return 0;
}
