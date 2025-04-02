#include "matrix.c"
#include <time.h>

#define EPOCH 1000000
#define LRATE 1

void train(struct Matrix* W1, struct Matrix* B1, struct Matrix* W2, struct Matrix* B2,
           struct Matrix* W3, struct Matrix* B3, const double inputs[4][2], const int outputs[4])
{
    double x1, x2;
    int y;
    for (size_t _ = 0; _ < EPOCH; ++_) {
        for (size_t in_idx = 0; in_idx < 4; ++in_idx)
        {
            x1 = inputs[in_idx][0];
            x2 = inputs[in_idx][1];
            y = outputs[in_idx];

            struct Matrix* X = mtrx_init(2, 1);
            X->matrix[0][0] = x1;
            X->matrix[1][0] = x2;

            // start forward pass
            struct Matrix* W1_X = mtrx_mult(W1, X);
            struct Matrix* Z1 = mtrx_add(W1_X, B1);
            struct Matrix* A1 = mtrx_sigmoid(Z1);

            struct Matrix* W2_A1 = mtrx_mult(W2, A1);
            struct Matrix* Z2 = mtrx_add(W2_A1, B2);
            struct Matrix* A2 = mtrx_sigmoid(Z2);

            struct Matrix* W3_A2 = mtrx_mult(W3, A2);
            struct Matrix* Z3 = mtrx_add(W3_A2, B3);
            struct Matrix* A3 = mtrx_sigmoid(Z3);

            long double y_hat = A3->matrix[0][0]; // output predicted by model
            // error is defined as (y - y_hat) ^ 2
            long double d_err_3 = -2 * (y - y_hat);

            for (size_t i = 0; i < W3->rows; ++i) {
                for (size_t j = 0; j < W3->cols; ++j)
                    W3->matrix[i][j] -= LRATE * d_err_3 * d_s(Z3->matrix[i][0]) * A2->matrix[j][0];
            }

            for (size_t i = 0; i < W3->rows; ++i) {
                B3->matrix[i][0] -= LRATE * d_err_3 * d_s(Z3->matrix[i][0]);
            }

            long double* d_err_2 = malloc(sizeof(long double) * A2->rows);
            for (size_t i = 0; i < A2->rows; ++i) {
                long double sum = 0;
                for (size_t k = 0; k < A3->rows; ++k)
                    sum += d_err_3 * d_s(Z3->matrix[k][0]) * W3->matrix[k][i];
                d_err_2[i] = sum;
            }

            for (size_t i = 0; i < W2->rows; ++i) {
                for (size_t j = 0; j < W2->cols; ++j)
                    W2->matrix[i][j] -= LRATE * d_err_2[i] * d_s(Z2->matrix[i][0]) * A1->matrix[j][0];
            }

            for (size_t i = 0; i < W3->rows; ++i) {
                B2->matrix[i][0] -= LRATE * d_err_2[i] * d_s(Z2->matrix[i][0]);
            }

            long double* d_err_1 = malloc(sizeof(long double) * A1->rows);
            for (size_t i = 0; i < A1->rows; ++i) {
                long double sum = 0;
                for (size_t k = 0; k < A2->rows; ++k)
                    sum += d_err_2[i] * d_s(Z2->matrix[k][0]) * W2->matrix[k][i];
                d_err_1[i] = sum;
            }

            for (size_t i = 0; i < W1->rows; ++i) {
                for (size_t j = 0; j < W1->cols; ++j)
                    W1->matrix[i][j] -= LRATE * d_err_1[i] * d_s(Z1->matrix[i][0]) * X->matrix[j][0];
            }

            for (size_t i = 0; i < W1->rows; ++i) {
                B1->matrix[i][0] -= LRATE * d_err_1[i] * d_s(Z1->matrix[i][0]);
            }

            free(d_err_2); free(d_err_1);
            mtrx_free(X); 
            mtrx_free(W1_X); mtrx_free(Z1); mtrx_free(A1);
            mtrx_free(W2_A1); mtrx_free(Z2); mtrx_free(A2);
            mtrx_free(W3_A2); mtrx_free(Z3); mtrx_free(A3);
        }
    }
}

long double predict(struct Matrix* W1, struct Matrix* B1, struct Matrix* W2,
                    struct Matrix* B2, struct Matrix* W3, struct Matrix* B3,
                    struct Matrix* X)
{
    struct Matrix* W1_X = mtrx_mult(W1, X);
    struct Matrix* Z1 = mtrx_add(W1_X, B1);
    struct Matrix* A1 = mtrx_sigmoid(Z1);

    struct Matrix* W2_A1 = mtrx_mult(W2, A1);
    struct Matrix* Z2 = mtrx_add(W2_A1, B2);
    struct Matrix* A2 = mtrx_sigmoid(Z2);

    struct Matrix* W3_A2 = mtrx_mult(W3, A2);
    struct Matrix* Z3 = mtrx_add(W3_A2, B3);
    struct Matrix* A3 = mtrx_sigmoid(Z3);

    long double y_hat = A3->matrix[0][0];
    mtrx_free(W1_X); mtrx_free(Z1); mtrx_free(A1);
    mtrx_free(W2_A1); mtrx_free(Z2); mtrx_free(A2);
    mtrx_free(W3_A2); mtrx_free(Z3); mtrx_free(A3);
    return y_hat;
}

int main(void)
{
    const size_t m = 2; // input size
    size_t n = 3; // neurons in 1st hidden layer
    size_t o = 2; // neurons in 2nd hidden layer
    const double inputs[4][2] = {{0.01, 0.01}, {0.01, 1.01}, {1.01, 0.01}, {1.01, 1.01}};
    const int outputs[4] = {0, 1, 1, 0};

    struct Matrix* W1 = mtrx_init(n, m); // first hidden
    struct Matrix* B1 = mtrx_init(n, 1);
    struct Matrix* W2 = mtrx_init(o, n); // second hidden
    struct Matrix* B2 = mtrx_init(o, 1);
    struct Matrix* W3 = mtrx_init(1, o); // output layer
    struct Matrix* B3 = mtrx_init(1, 1);

    train(W1, B1, W2, B2, W3, B3, inputs, outputs);
    for (int i = 0; i < 4; ++i)
    {
        struct Matrix* X = mtrx_init(2, 1);
        X->matrix[0][0] = inputs[i][0];
        X->matrix[1][0] = inputs[i][1];

        printf("Input: {%f, %f}; Expected: %d; Predicted: %Lf\n", 
        inputs[i][0], inputs[i][1], outputs[i], predict(W1, B1, W2, B2, W3, B3, X));

        mtrx_free(X);
    }
    mtrx_free(W1); mtrx_free(B1);
    mtrx_free(W2); mtrx_free(B2);
    mtrx_free(W3); mtrx_free(B3);
    return 0;
}
