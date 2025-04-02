#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct Matrix {
    size_t rows;
    size_t cols;
    long double** matrix;
};

void mtrx_fill_rand(struct Matrix* m)
{
    for (size_t i = 0; i < m->cols; ++i) {
        for (size_t j = 0; j < m->rows; ++j)
            m->matrix[j][i] = (long double)rand() / RAND_MAX;
    }
}

struct Matrix* mtrx_init(size_t rows, size_t cols)
{
    struct Matrix* p = malloc(sizeof(struct Matrix));
    p->cols = cols;
    p->rows = rows;
    p->matrix = malloc(rows * sizeof(long double*));
    for (int i = 0; (size_t) i < rows; ++i)
        p->matrix[i] = malloc(cols * sizeof(long double));
    mtrx_fill_rand(p);
    return p;
}

struct Matrix* mtrx_mult(struct Matrix* a, struct Matrix* b)
{
    struct Matrix* new = mtrx_init(a->rows, b->cols);
    for (size_t row = 0; row < a->rows; ++row)
    {
        long double res = 0;
        for (size_t col = 0; col < a->cols; ++col)
            res += a->matrix[row][col] * b->matrix[col][0];
        new->matrix[row][0] = res;
    }
    return new;
}

struct Matrix* mtrx_add(struct Matrix* a, struct Matrix* b)
{
    struct Matrix* new = mtrx_init(a->rows, a->cols);
    for (size_t i = 0; i < a->rows; ++i)
        new->matrix[i][0] = a->matrix[i][0] + b->matrix[i][0];
    return new;
}

void mtrx_free(struct Matrix* a)
{
    for (size_t rows = 0; rows < a->rows; ++rows)
        free(a->matrix[rows]);
    free(a->matrix);
    free(a);
}

long double s(long double x)
{
    return (long double) 1.0 / (1.0 + exp(-x));
}

long double d_s(long double x)
{
    return s(x) * (1 - s(x));
}

struct Matrix* mtrx_sigmoid(struct Matrix* a)
{
    struct Matrix* new = mtrx_init(a->rows, a->cols);
    for (size_t i = 0; i < a->rows; ++i)
        new->matrix[i][0] = s(a->matrix[i][0]);
    return new;
}

void mtrx_print(struct Matrix* a)
{
    for (size_t row = 0; row < a->rows; ++row) {
        for (size_t col = 0; col < a->cols; ++col)
            printf("%Lf ", a->matrix[row][col]);
        printf("\n");
    }
    printf("\n");
}
