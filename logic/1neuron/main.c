#include <stdio.h>
#include <stdbool.h>

#define LEARNING_RATE 0.1
#define EPOCHS 100

bool activation(double wb, double w1, double w2, int i1, int i2) {
    return wb + w1 * i1 + w2 * i2 > 0;
}

void train(double *wb, double *w1, double *w2, int inputs[][2], int labels[], int n) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < n; i++) {
            int i1 = inputs[i][0];
            int i2 = inputs[i][1];
            bool output = activation(*wb, *w1, *w2, i1, i2);
            int error = labels[i] - output;

            *wb += LEARNING_RATE * error;
            *w1 += LEARNING_RATE * error * i1;
            *w2 += LEARNING_RATE * error * i2;
        }
        printf("Weights after epoch %d\n", epoch + 1);
        printf("Biased weight: %f\n", *wb);
        printf("Weight 1: %f\n", *w1);
        printf("Weight 2: %f\n", *w2);
        printf("----------------------\n");
    }
}

int main(void) {
    double wb = 0, w1 = 0, w2 = 0;

    int inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int labels[4] = {0, 1, 1, 0};

    train(&wb, &w1, &w2, inputs, labels, 4);

    for (int i = 0; i < 4; i++) {
        int i1 = inputs[i][0];
        int i2 = inputs[i][1];
        bool output = activation(wb, w1, w2, i1, i2);
        printf("Input: (%d, %d) -> Output: %d\n", i1, i2, output);
    }

    return 0;
}
