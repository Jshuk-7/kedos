#ifndef KEDOS_H
#define KEDOS_H

#include <stddef.h>
#include <stdio.h>

#ifndef K_MALLOC
#include <stdlib.h>
#define K_MALLOC malloc
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

#define MATRIX_AT(m, r, c) (m).elems[((r) * (m).cols) + (c)]

typedef struct
{
	size_t rows;
	size_t cols;
	real_t* elems;
} Matrix;

Matrix kedos_matrix_alloc(size_t rows, size_t cols);
void kedos_matrix_free(Matrix mat);
void kedos_matrix_zero_mem(Matrix mat);
void kedos_matrix_fill(Matrix mat, real_t v);
real_t kedos_matrix_get(Matrix m, size_t row, size_t col);
void kedos_matrix_set(Matrix m, size_t row, size_t col, real_t v);
void kedos_matrix_rand(Matrix mat, real_t low, real_t high);
void kedos_matrix_sum(Matrix dst, Matrix other);
void kedos_matrix_dot(Matrix dst, Matrix lhs, Matrix rhs);
void kedos_matrix_display(Matrix mat);

real_t kedos_rand_real();

#endif

#ifdef KEDOS_IMPLEMENTATION

Matrix kedos_matrix_alloc(size_t rows, size_t cols)
{
	Matrix mat;
	mat.rows = rows;
	mat.cols = cols;
	mat.elems = (real_t*)K_MALLOC(sizeof(real_t) * rows * cols);
	K_ASSERT(mat.elems != NULL);
	return mat;
}

void kedos_matrix_free(Matrix mat)
{
	free(mat.elems);
}

void kedos_matrix_zero_mem(Matrix mat)
{
	kedos_matrix_fill(mat, 0);
}

void kedos_matrix_fill(Matrix mat, real_t v)
{
	for (size_t row = 0; row < mat.rows; row++) {
		for (size_t col = 0; col < mat.cols; col++) {
			MATRIX_AT(mat, row, col) = v;
		}
	}
}

real_t kedos_matrix_get(Matrix m, size_t row, size_t col)
{
	return MATRIX_AT(m, row, col);
}

void kedos_matrix_set(Matrix m, size_t row, size_t col, real_t v)
{
	MATRIX_AT(m, row, col) = v;
}

void kedos_matrix_rand(Matrix mat, real_t low, real_t high)
{
	for (size_t row = 0; row < mat.rows; row++) {
		for (size_t col = 0; col < mat.cols; col++) {
			real_t v = kedos_rand_real() * (high - low) + low;
			MATRIX_AT(mat, row, col) = v;
		}
	}
}

void kedos_matrix_sum(Matrix dst, Matrix other)
{
	K_ASSERT(dst.rows == other.rows);
	K_ASSERT(dst.cols == other.cols);

	for (size_t row = 0; row < dst.rows; row++) {
		for (size_t col = 0; col < dst.cols; col++) {
			MATRIX_AT(dst, row, col) += MATRIX_AT(other, row, col);
		}
	}
}

void kedos_matrix_dot(Matrix dst, Matrix lhs, Matrix rhs)
{
	K_ASSERT(lhs.cols == rhs.rows);
	K_ASSERT(dst.rows == lhs.rows);
	K_ASSERT(dst.cols == lhs.cols);

	size_t n = lhs.cols;

	kedos_matrix_zero_mem(dst);

	for (size_t row = 0; row < dst.rows; row++) {
		for (size_t col = 0; col < dst.cols; col++) {
			for (size_t k = 0; k < n; k++) {
				MATRIX_AT(dst, row, col) += MATRIX_AT(lhs, row, k) * MATRIX_AT(rhs, k, col);
			}
		}
	}
}

void kedos_matrix_display(Matrix mat)
{
	for (size_t row = 0; row < mat.rows; row++) {
		for (size_t col = 0; col < mat.cols; col++) {
			printf("%f, ", MATRIX_AT(mat, row, col));
		}

		printf("\n");
	}

	(void) mat;
}

real_t kedos_rand_real()
{
	return (real_t)rand() / (real_t)RAND_MAX;
}

#endif