#ifndef KEDOS_H
#define KEDOS_H

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef K_MALLOC
#include <stdlib.h>
#define K_MALLOC malloc
#endif

#ifndef K_REALLOC
#include <stdlib.h>
#define K_REALLOC realloc
#endif

#ifndef K_FREE
#include <stdlib.h>
#define K_FREE free
#endif

#ifndef K_ASSERT
#include <assert.h>
#define K_ASSERT assert
#endif

#ifdef KEDOS_USE_EXTRA_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

typedef struct Matrix
{
	size_t rows;
	size_t cols;
	size_t stride;
	real_t* elems;
} Matrix;

typedef real_t (*MatrixFn)(real_t);

Matrix kedos_matrix_alloc(size_t rows, size_t cols);
void kedos_matrix_free(Matrix mat);
void kedos_matrix_copy(Matrix dst, Matrix src);
void kedos_matrix_zero_mem(Matrix mat);
void kedos_matrix_fill(Matrix mat, real_t x);
Matrix kedos_matrix_row(Matrix mat, size_t row);
real_t kedos_matrix_get(Matrix mat, size_t row, size_t col);
void kedos_matrix_set(Matrix mat, size_t row, size_t col, real_t x);
void kedos_matrix_rand(Matrix mat, real_t low, real_t high);
void kedos_matrix_sum(Matrix dst, Matrix other);
void kedos_matrix_dot(Matrix dst, Matrix lhs, Matrix rhs);
void kedos_matrix_for_each(Matrix mat, MatrixFn func);
void kedos_matrix_display(Matrix mat, const char* name);

#define K_MATRIX_AT(m, r, c) (m).elems[((r) * (m).stride) + (c)]
#define K_MATRIX_PRINT(m) kedos_matrix_display(m, #m)


typedef struct NeuralNetwork
{
	size_t layer_count;
	Matrix *weights;
	Matrix *biases;
	Matrix *activations;
} NeuralNetwork;

NeuralNetwork kedos_neural_network_alloc(size_t *arch, size_t arch_count);
void kedos_neural_network_free(NeuralNetwork nn);
void kedos_neural_network_display(NeuralNetwork nn, const char* name);

#define K_NETWORK_PRINT(nn) kedos_neural_network_display(nn, #nn)

// Misc

#define ARRAY_COUNT(arr) (sizeof((arr)) / sizeof((arr[0])))

real_t kedos_rand_real(void);

// Activation functions

real_t kedos_sigmoid(real_t x);
real_t kedos_relu(real_t x);

#endif

#ifdef KEDOS_IMPLEMENTATION

Matrix kedos_matrix_alloc(size_t rows, size_t cols)
{
	Matrix mat;
	mat.rows = rows;
	mat.cols = cols;
	mat.stride = cols;
	mat.elems = (real_t*)K_MALLOC(sizeof(real_t) * rows * cols);
	K_ASSERT(mat.elems != NULL);
	return mat;
}

void kedos_matrix_free(Matrix mat)
{
	K_FREE(mat.elems);
}

void kedos_matrix_copy(Matrix dst, Matrix src)
{
	K_ASSERT(dst.rows == src.rows);
	K_ASSERT(dst.cols == src.cols);

	for (size_t row = 0; row < dst.rows; row++) {
		for (size_t col = 0; col < dst.cols; col++) {
			K_MATRIX_AT(dst, row, col) = K_MATRIX_AT(src, row, col);
		}
	}
}

void kedos_matrix_zero_mem(Matrix mat)
{
	kedos_matrix_fill(mat, 0);
}

void kedos_matrix_fill(Matrix mat, real_t x)
{
	for (size_t row = 0; row < mat.rows; row++) {
		for (size_t col = 0; col < mat.cols; col++) {
			K_MATRIX_AT(mat, row, col) = x;
		}
	}
}

Matrix kedos_matrix_row(Matrix mat, size_t row)
{
	Matrix r = {
		.rows = 1,
		.cols = mat.cols,
		.stride = mat.cols,
		.elems = &K_MATRIX_AT(mat, row, 0)
	};
	return r;
}

real_t kedos_matrix_get(Matrix mat, size_t row, size_t col)
{
	return K_MATRIX_AT(mat, row, col);
}

void kedos_matrix_set(Matrix mat, size_t row, size_t col, real_t x)
{
	K_MATRIX_AT(mat, row, col) = x;
}

void kedos_matrix_rand(Matrix mat, real_t low, real_t high)
{
	for (size_t row = 0; row < mat.rows; row++) {
		for (size_t col = 0; col < mat.cols; col++) {
			real_t v = kedos_rand_real() * (high - low) + low;
			K_MATRIX_AT(mat, row, col) = v;
		}
	}
}

void kedos_matrix_sum(Matrix dst, Matrix other)
{
	K_ASSERT(dst.rows == other.rows);
	K_ASSERT(dst.cols == other.cols);

	for (size_t row = 0; row < dst.rows; row++) {
		for (size_t col = 0; col < dst.cols; col++) {
			K_MATRIX_AT(dst, row, col) += K_MATRIX_AT(other, row, col);
		}
	}
}

void kedos_matrix_dot(Matrix dst, Matrix lhs, Matrix rhs)
{
	K_ASSERT(lhs.cols == rhs.rows);
	K_ASSERT(dst.rows == lhs.rows);
	K_ASSERT(dst.cols == rhs.cols);

	size_t n = lhs.cols;

	kedos_matrix_zero_mem(dst);

	for (size_t row = 0; row < dst.rows; row++) {
		for (size_t col = 0; col < dst.cols; col++) {
			for (size_t k = 0; k < n; k++) {
				K_MATRIX_AT(dst, row, col) += K_MATRIX_AT(lhs, row, k) * K_MATRIX_AT(rhs, k, col);
			}
		}
	}
}

void kedos_matrix_for_each(Matrix mat, MatrixFn func)
{
	for (size_t row = 0; row < mat.rows; row++) {
		for (size_t col = 0; col < mat.cols; col++) {
			real_t v = K_MATRIX_AT(mat, row, col);
			K_MATRIX_AT(mat, row, col) = (*func)(v);
		}
	}
}

void kedos_matrix_display(Matrix mat, const char* name)
{
	printf("%s = [\n", name);
	for (size_t row = 0; row < mat.rows; row++) {
		for (size_t col = 0; col < mat.cols; col++) {
			printf("\t%f", K_MATRIX_AT(mat, row, col));
		}

		printf("\n");
	}
	printf("]\n");

	(void) mat;
}

NeuralNetwork kedos_neural_network_alloc(size_t *arch, size_t arch_count)
{
	K_ASSERT(arch_count > 0);
	NeuralNetwork nn;
	nn.layer_count = arch_count - 1;

	nn.weights = (Matrix*)K_MALLOC(sizeof(*nn.weights) * nn.layer_count);
	K_ASSERT(nn.weights != NULL);
	nn.biases = (Matrix*)K_MALLOC(sizeof(*nn.biases) * nn.layer_count);
	K_ASSERT(nn.biases != NULL);
	nn.activations = (Matrix*)K_MALLOC(sizeof(*nn.activations) * nn.layer_count + 1);
	K_ASSERT(nn.activations != NULL);

	nn.activations[0] = kedos_matrix_alloc(1, arch[0]);

	for (size_t i = 1; i < arch_count; i++)
	{
		nn.weights[i - 1] = kedos_matrix_alloc(nn.activations[i - 1].cols, arch[i]);
		nn.biases[i - 1] = kedos_matrix_alloc(1, arch[i]);
		nn.activations[i] = kedos_matrix_alloc(1, arch[i]);
	}

	return nn;
}

void kedos_neural_network_free(NeuralNetwork nn)
{

}

void kedos_neural_network_display(NeuralNetwork nn, const char *name)
{
	
}

real_t kedos_rand_real(void)
{
	return (real_t)rand() / (real_t)RAND_MAX;
}

real_t kedos_sigmoid(real_t x)
{
	return 1 / (1 + (real_t)exp(-x));
}

real_t kedos_relu(real_t x)
{
	return (real_t)(x < 0 ? 0 : x);
}

#endif