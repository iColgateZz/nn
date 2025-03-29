#include "matrix.c"
#include <time.h>

#define EPOCH 1000000
#define LRATE 0.01

long double predict(struct Matrix* U, struct Matrix* A,
           struct Matrix* W, struct Matrix* C, struct Matrix* X)
{
    struct Matrix* UX = mtrx_mult(U, X);
    struct Matrix* F = mtrx_add(UX, A);
    mtrx_sigmoid(F);
    struct Matrix* WsigmF = mtrx_mult(W, F);
    struct Matrix* out = mtrx_add(WsigmF, C);
    mtrx_sigmoid(out);
    long double y_hat = out->matrix[0][0];
    mtrx_free(UX); mtrx_free(F); mtrx_free(WsigmF); mtrx_free(out);
    return y_hat;
}

void train(struct Matrix* U, struct Matrix* A,
           struct Matrix* W, struct Matrix* C, const int inputs[4][2], const int outputs[4])
{
    int x1, x2;
    int y;
    for (size_t _ = 0; _ < EPOCH; ++_) {
        for (size_t inp_idx = 0; inp_idx < 4; ++inp_idx)
        {
            x1 = inputs[inp_idx][0];
            x2 = inputs[inp_idx][1];
            y = outputs[inp_idx];

            struct Matrix* X = mtrx_init(2, 1);
            X->matrix[0][0] = x1;
            X->matrix[1][0] = x2;

            struct Matrix* UX = mtrx_mult(U, X);
            struct Matrix* F = mtrx_add(UX, A);
            mtrx_sigmoid(F);
            struct Matrix* WsigmF = mtrx_mult(W, F);
            struct Matrix* out = mtrx_add(WsigmF, C);
            mtrx_sigmoid(out);

            long double y_hat = out->matrix[0][0];
            long double err = y - y_hat;

            // Start backpropagation
            C->matrix[0][0] += LRATE * err;
            for (size_t i = 0; i < W->cols; ++i)
                W->matrix[0][i] += LRATE * err * F->matrix[i][0];

            for (size_t i = 0; i < A->rows; ++i)
                A->matrix[i][0] += LRATE * err * W->matrix[0][i] * F->matrix[i][0] * (1 - F->matrix[i][0]);

            for (size_t i = 0; i < U->rows; ++i) {
                for (size_t j = 0; j < U->cols; ++j)
                    U->matrix[i][j] += LRATE * err * W->matrix[0][i] * F->matrix[i][0] * (1 - F->matrix[i][0]) * X->matrix[j][0];
            }
            
            mtrx_free(UX); mtrx_free(F); mtrx_free(WsigmF); mtrx_free(out); mtrx_free(X);
        }
    }
}

int main(void)
{
    size_t m = 2; // inputs;
    size_t n = 5; // hidden layer;
    const int inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const int outputs[4] = {0, 1, 1, 0};
    srand(time(NULL));

    struct Matrix* U = mtrx_init(n, m);
    struct Matrix* A = mtrx_init(n, 1);
    struct Matrix* W = mtrx_init(1, m);
    struct Matrix* C = mtrx_init(1, 1);

    train(U, A, W, C, inputs, outputs);
    for (int i = 0; i < 4; ++i)
    {
        struct Matrix* X = mtrx_init(2, 1);
        X->matrix[0][0] = inputs[i][0];
        X->matrix[1][0] = inputs[i][1];

        printf("Input: {%d, %d}; Expected: %d; Predicted: %Lf\n", 
        inputs[i][0], inputs[i][1], outputs[i], predict(U, A, W, C, X));

        mtrx_free(X);
    }
    return 0;
}
