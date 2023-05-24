#ifndef KEDOS_H
#define KEDOS_H

/*
 * MIT License
 * 
 * Copyright (c) 2023 Jshuk-7
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/

/// @file kedos.h

#include <stddef.h>
#include <stdint.h>
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

/// @brief Heap allocated array of 'real_t'.
typedef struct Matrix
{
	size_t rows;
	size_t cols;
	size_t stride;
	real_t* elems;
} Matrix;

/// @brief Activation function that modifies the input.
typedef real_t (*ActivationFn)(real_t);

/// @brief Creates a new matrix with 'rows' and 'cols'.
/// @param rows the rows of the matrix
/// @param cols the columns of the matrix
/// @return a new matrix
Matrix kedos_matrix_alloc(size_t rows, size_t cols);

/// @brief Frees the matrix's buffer.
/// @param mat the matrix to free
void kedos_matrix_free(Matrix mat);

/// @brief Copies the elements of 'src' into 'dst'.
/// @param dst the destination matrix
/// @param src the source matrix
void kedos_matrix_copy(Matrix dst, Matrix src);

/// @brief Zeros out all elements of the matrix.
/// @param mat the matrix
void kedos_matrix_zero_mem(Matrix mat);

/// @brief Fills a matrix's elements with 'x'.
/// @param mat the matrix to fill
/// @param x the element to fill the matrix with
void kedos_matrix_fill(Matrix mat, real_t x);

/// @brief Extracts a row from a matrix as a separate matrix. The original matrix will be unmodified.
/// @param mat the matrix to extract from
/// @param row the row to extract from
/// @return row as a new matrix from the original
Matrix kedos_matrix_row(Matrix mat, size_t row);

/// @brief Gets an element from a matrix.
/// @param mat the matrix
/// @param row the row
/// @param col the column
/// @return the element in the matrix at 'row' and 'column'
real_t kedos_matrix_get(Matrix mat, size_t row, size_t col);

/// @brief Sets an element in a matrix.
/// @param mat the matrix
/// @param row the row
/// @param col the column
/// @param x the element to set
void kedos_matrix_set(Matrix mat, size_t row, size_t col, real_t x);

/// @brief Randomizes all elements in the matrix.
/// @param mat the matrix to randomize
/// @param low the lowest element value possible
/// @param high the highest element value possible
void kedos_matrix_rand(Matrix mat, real_t low, real_t high);

/// @brief Adds two matrices together. The rows and columns of both matrices must be equal to perform this operation.
/// @param dst the destination
/// @param other the matrix to add to 'dst'
void kedos_matrix_sum(Matrix dst, Matrix other);

/// @brief Gets the dot product of two matrices. The columns of 'lhs' must be equal to the rows of 'rhs' to perform this operation. The shape of 'dst' must also match the shape of 'lhs'.
/// @param dst the destination matrix
/// @param lhs the left matrix
/// @param rhs the right matrix
void kedos_matrix_dot(Matrix dst, Matrix lhs, Matrix rhs);

/// @brief Performs an activation function on each element of the matrix.
/// @param mat the matrix to modify
/// @param func the activation function to perform on each element
void kedos_matrix_for_each(Matrix mat, ActivationFn func);

/// @brief Prints out all elements of the matrix nicely formatted to stdout.
/// @param mat the matrix to print
/// @param name the name of the matrix
void kedos_matrix_display(Matrix mat, const char* name, uint32_t padding);

#define K_MATRIX_PRINT(m) kedos_matrix_display(m, #m, 0)
#define K_MATRIX_AT(m, r, c) (m).elems[((r) * (m).stride) + (c)]

/// @brief A collections of intertwined matrices, such as weights, biases, inputs and outputs.
typedef struct NeuralNetwork
{
	size_t layer_count;
	Matrix* weights;
	Matrix* biases;
	Matrix* activations;
} NeuralNetwork;

/// @brief Creates a new neural network.
/// @param arch the architecture of the network
/// @param arch_count the count
/// @return a new neural network
NeuralNetwork kedos_neural_network_alloc(size_t* arch, size_t arch_count);

/// @brief Frees a neural network from memory
/// @param nn the neural network to free
void kedos_neural_network_free(NeuralNetwork nn);

/// @brief Forwards the inputs through the neural network.
/// @param nn the neural network
void kedos_neural_network_forward(NeuralNetwork nn);

/// @brief Determines the effectiveness of a neural network based on the expected output
/// @param nn the neural network to test
/// @param input the input training data
/// @param output the output training data
/// @return the cost of the neural network
real_t kedos_neural_network_cost(NeuralNetwork nn, Matrix input, Matrix output);

/// @brief Computes the gradient or 'slope' of the neural network
/// @param nn the neural network to test
/// @param gradient the slope of the model
/// @param epsilon the amount to tweak the weights and biases by
/// @param training_in the training input data
/// @param training_out the training output data
void kedos_neural_network_finite_difference(NeuralNetwork nn, NeuralNetwork gradient, real_t epsilon, Matrix training_in, Matrix training_out);

/// @brief Applies the gradient to the neural network
/// @param nn the neural network to apply the gradient to
/// @param gradient the computed gradient of the neural network
/// @param learning_rate the rate that the gradient will be applied to the neural network
void kedos_neural_network_learn(NeuralNetwork nn, NeuralNetwork gradient, real_t learning_rate);

/// @brief Randomizes all matrices in the neural network.
/// @param nn the neural network to ranomize
/// @param low the lowest element value possible
/// @param high the highest element value possible
void kedos_neural_network_rand(NeuralNetwork nn, real_t low, real_t high);

/// @brief Prints out a nueral network nicely formatted to stdout.
/// @param nn the neural network to print
/// @param name the name of the neural network
void kedos_neural_network_display(NeuralNetwork nn, const char* name);

#define K_NETWORK_PRINT(nn) kedos_neural_network_display(nn, #nn)
#define K_NETWORK_INPUT(nn) (nn).activations[0]
#define K_NETWORK_OUTPUT(nn) (nn).activations[(nn).layer_count]

// Misc

#define K_ARRAY_COUNT(arr) (sizeof((arr)) / sizeof((arr[0])))

/// @brief Gets a random 'real_t'
/// @return a random 'real_t'
real_t kedos_rand_real(void);

// Activation functions

/// @brief Sigmoid activation function
/// @param x the input
/// @return 'x' mapped from 0 -> 1
real_t kedos_activation_sigmoid(real_t x);

/// @brief Relu activation function
/// @param x the input
/// @return 0 if x is less than 0 otherwise x
real_t kedos_activation_relu(real_t x);

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

void kedos_matrix_for_each(Matrix mat, ActivationFn func)
{
	for (size_t row = 0; row < mat.rows; row++) {
		for (size_t col = 0; col < mat.cols; col++) {
			real_t v = K_MATRIX_AT(mat, row, col);
			K_MATRIX_AT(mat, row, col) = (*func)(v);
		}
	}
}

void kedos_matrix_display(Matrix mat, const char* name, uint32_t padding)
{
	printf("%*s%s = [\n", padding, "", name);
	for (size_t row = 0; row < mat.rows; row++) {
		printf("%*s\t", padding, "");
		for (size_t col = 0; col < mat.cols; col++) {
			printf("%f ", K_MATRIX_AT(mat, row, col));
		}

		printf("\n");
	}

	printf("%*s]\n", padding, "");

	(void) mat;
}

NeuralNetwork kedos_neural_network_alloc(size_t* arch, size_t arch_count)
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
		Matrix* weight = &nn.weights[i - 1];
		*weight = kedos_matrix_alloc(nn.activations[i - 1].cols, arch[i]);
		kedos_matrix_zero_mem(*weight);

		Matrix* bias = &nn.biases[i - 1];
		*bias = kedos_matrix_alloc(1, arch[i]);
		kedos_matrix_zero_mem(*bias);
		
		Matrix* activation = &nn.activations[i];
		*activation = kedos_matrix_alloc(1, arch[i]);
		kedos_matrix_zero_mem(*activation);
	}

	return nn;
}

void kedos_neural_network_free(NeuralNetwork nn)
{
	for (size_t i = 0; i < nn.layer_count; i++) {
		kedos_matrix_free(nn.weights[i]);
		kedos_matrix_free(nn.biases[i]);
		kedos_matrix_free(nn.activations[i]);
	}

	// destroy the input
	kedos_matrix_free(nn.activations[0]);
}

void kedos_neural_network_forward(NeuralNetwork nn)
{
	for (size_t i = 0; i < nn.layer_count; i++) {
		Matrix* next = &nn.activations[i + 1];
		kedos_matrix_dot(*next, nn.activations[i], nn.weights[i]);
		kedos_matrix_sum(*next, nn.biases[i]);
		kedos_matrix_for_each(*next, kedos_activation_sigmoid);
	}
}

real_t kedos_neural_network_cost(NeuralNetwork nn, Matrix input, Matrix output)
{
	K_ASSERT(input.rows == output.rows);
	size_t network_output_cols = K_NETWORK_OUTPUT(nn).cols;
	K_ASSERT(output.cols == network_output_cols);

	// Mean Squared Error

	size_t input_samples = input.rows;
	real_t cost = 0;

	for (size_t row = 0; row < input_samples; row++) {
		Matrix x = kedos_matrix_row(input, row);
		Matrix y = kedos_matrix_row(output, row);

		kedos_matrix_copy(K_NETWORK_INPUT(nn), x);
		kedos_neural_network_forward(nn);

		for (size_t col = 0; col < network_output_cols; col++) {
			real_t difference = K_MATRIX_AT(K_NETWORK_OUTPUT(nn), 0, col) - K_MATRIX_AT(y, 0, col);
			cost += difference * difference;
		}
	}

	return (real_t)(cost / input_samples);
}

void kedos_neural_network_finite_difference(NeuralNetwork nn, NeuralNetwork gradient, real_t epsilon, Matrix training_in, Matrix training_out)
{
	real_t cost = kedos_neural_network_cost(nn, training_in, training_out);
	real_t saved;

	for (size_t i = 0; i < nn.layer_count; i++) {
		for (size_t row = 0; row < nn.weights[i].rows; row++) {
			for (size_t col = 0; col < nn.weights[i].cols; col++) {
				saved = K_MATRIX_AT(nn.weights[i], row, col);
				K_MATRIX_AT(nn.weights[i], row, col) += epsilon;
				K_MATRIX_AT(gradient.weights[i], row, col) = (kedos_neural_network_cost(nn, training_in, training_out) - cost) / epsilon;
				K_MATRIX_AT(nn.weights[i], row, col) = saved;
			}
		}

		for (size_t row = 0; row < nn.biases[i].rows; row++) {
			for (size_t col = 0; col < nn.biases[i].cols; col++) {
				saved = K_MATRIX_AT(nn.biases[i], row, col);
				K_MATRIX_AT(nn.biases[i], row, col) += epsilon;
				K_MATRIX_AT(gradient.biases[i], row, col) = (kedos_neural_network_cost(nn, training_in, training_out) - cost) / epsilon;
				K_MATRIX_AT(nn.biases[i], row, col) = saved;
			}
		}
	}
}

void kedos_neural_network_learn(NeuralNetwork nn, NeuralNetwork gradient, real_t learning_rate)
{
	for (size_t i = 0; i < nn.layer_count; i++) {
		for (size_t row = 0; row < nn.weights[i].rows; row++) {
			for (size_t col = 0; col < nn.weights[i].cols; col++) {
				K_MATRIX_AT(nn.weights[i], row, col) -= learning_rate * K_MATRIX_AT(gradient.weights[i], row, col);
			}
		}

		for (size_t row = 0; row < nn.biases[i].rows; row++) {
			for (size_t col = 0; col < nn.biases[i].cols; col++) {
				K_MATRIX_AT(nn.biases[i], row, col) -= learning_rate * K_MATRIX_AT(gradient.biases[i], row, col);
			}
		}
	}
}

void kedos_neural_network_rand(NeuralNetwork nn, real_t low, real_t high)
{
	for (size_t i = 0; i < nn.layer_count; i++) {
		kedos_matrix_rand(nn.weights[i], low, high);
		kedos_matrix_rand(nn.biases[i], low, high);
	}
}

void kedos_neural_network_display(NeuralNetwork nn, const char* name)
{
	printf("%s =[\n", name);
	Matrix* ws = nn.weights;
	Matrix* bs = nn.biases;
	Matrix* as = nn.activations;

	char name_buffer[16];

	for (size_t i = 0; i < nn.layer_count; i++) {
		snprintf(name_buffer, 16, "ws%zu", i + 1);
		kedos_matrix_display(ws[i], name_buffer, 4);
		snprintf(name_buffer, 16, "bs%zu", i + 1);
		kedos_matrix_display(bs[i], name_buffer, 4);
	}

	printf("]\n");
}	

real_t kedos_rand_real(void)
{
	return (real_t)rand() / (real_t)RAND_MAX;
}

real_t kedos_activation_sigmoid(real_t x)
{
	return 1 / (1 + (real_t)exp(-x));
}

real_t kedos_activation_relu(real_t x)
{
	return (real_t)(x < 0 ? 0 : x);
}

#endif