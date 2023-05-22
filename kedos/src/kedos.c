#define KEDOS_IMPLEMENTATION
#include "kedos.h"

#include <time.h>

typedef struct XorModel
{
	Matrix a0;
	Matrix w1, b1, a1;
	Matrix w2, b2, a2;
} XorModel;

XorModel xor_model_alloc(void)
{
	XorModel m;
	m.a0 = kedos_matrix_alloc(1, 2);
	m.w1 = kedos_matrix_alloc(2, 2);
	m.b1 = kedos_matrix_alloc(1, 2);
	m.a1 = kedos_matrix_alloc(1, 2);
	m.w2 = kedos_matrix_alloc(2, 1);
	m.b2 = kedos_matrix_alloc(1, 1);
	m.a2 = kedos_matrix_alloc(1, 1);
	return m;
}

void forward_xor(XorModel m)
{
	kedos_matrix_dot(m.a1, m.a0, m.w1);
	kedos_matrix_sum(m.a1, m.b1);
	kedos_matrix_for_each(m.a1, kedos_sigmoid);

	kedos_matrix_dot(m.a2, m.a1, m.w2);
	kedos_matrix_sum(m.a2, m.b2);
	kedos_matrix_for_each(m.a2, kedos_sigmoid);
}

real_t xor_model_cost(XorModel m, Matrix input, Matrix output)
{
	K_ASSERT(input.rows == output.rows);
	K_ASSERT(output.cols == m.a2.cols);

	size_t r = input.rows;
	real_t cost = 0;

	for (size_t row = 0; row < r; row++) {
		Matrix x = kedos_matrix_row(input, row);
		Matrix y = kedos_matrix_row(output, row);

		kedos_matrix_copy(m.a0, x);
		forward_xor(m);

		size_t c = output.cols;
		for (size_t col = 0; col < c; col++) {
			real_t diff = K_MATRIX_AT(m.a2, 0, col) - K_MATRIX_AT(y, 0, col);
			real_t diff_sq = diff * diff;
			cost += diff_sq;
		}
	}

	real_t samples = r;
	real_t avg = cost / samples;

	return avg;
}

void finite_difference(XorModel m, XorModel g, real_t eps, Matrix in, Matrix out)
{
	real_t saved;

	real_t c = xor_model_cost(m, in, out);

	for (size_t i = 0; i < m.w1.rows; i++) {
		for (size_t j = 0; j < m.w1.cols; j++) {
			saved = K_MATRIX_AT(m.w1, i, j);
			K_MATRIX_AT(m.w1, i, j) += eps;
			K_MATRIX_AT(g.w1, i, j) = (xor_model_cost(m, in, out) - c) / eps;
			K_MATRIX_AT(m.w1, i, j) = saved;
		}
	}

	for (size_t i = 0; i < m.b1.rows; i++) {
		for (size_t j = 0; j < m.b1.cols; j++) {
			saved = K_MATRIX_AT(m.b1, i, j);
			K_MATRIX_AT(m.b1, i, j) += eps;
			K_MATRIX_AT(g.b1, i, j) = (xor_model_cost(m, in, out) - c) / eps;
			K_MATRIX_AT(m.b1, i, j) = saved;
		}
	}

	for (size_t i = 0; i < m.w2.rows; i++) {
		for (size_t j = 0; j < m.w2.cols; j++) {
			saved = K_MATRIX_AT(m.w2, i, j);
			K_MATRIX_AT(m.w2, i, j) += eps;
			K_MATRIX_AT(g.w2, i, j) = (xor_model_cost(m, in, out) - c) / eps;
			K_MATRIX_AT(m.w2, i, j) = saved;
		}
	}

	for (size_t i = 0; i < m.b2.rows; i++) {
		for (size_t j = 0; j < m.b2.cols; j++) {
			saved = K_MATRIX_AT(m.b2, i, j);
			K_MATRIX_AT(m.b2, i, j) += eps;
			K_MATRIX_AT(g.b2, i, j) = (xor_model_cost(m, in, out) - c) / eps;
			K_MATRIX_AT(m.b2, i, j) = saved;
		}
	}
}

void xor_model_learn(XorModel m, XorModel g, real_t rate)
{
	for (size_t i = 0; i < m.w1.rows; i++) {
		for (size_t j = 0; j < m.w1.cols; j++) {
			K_MATRIX_AT(m.w1, i, j) -= rate * K_MATRIX_AT(g.w1, i, j);
		}
	}

	for (size_t i = 0; i < m.b1.rows; i++) {
		for (size_t j = 0; j < m.b1.cols; j++) {
			K_MATRIX_AT(m.b1, i, j) -= rate * K_MATRIX_AT(g.b1, i, j);
		}
	}

	for (size_t i = 0; i < m.w2.rows; i++) {
		for (size_t j = 0; j < m.w2.cols; j++) {
			K_MATRIX_AT(m.w2, i, j) -= rate * K_MATRIX_AT(g.w2, i, j);
		}
	}

	for (size_t i = 0; i < m.b2.rows; i++) {
		for (size_t j = 0; j < m.b2.cols; j++) {
			K_MATRIX_AT(m.b2, i, j) -= rate * K_MATRIX_AT(g.b2, i, j);
		}
	}
}

real_t DATA[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 0,
};

int main(void) {
	srand(time(0));

	Matrix input = { .rows = 4, .cols = 2, .stride = 3, .elems = DATA };
	Matrix output = { .rows = 4, .cols = 1, .stride = 3, .elems = &DATA[2] };

	XorModel m = xor_model_alloc();
	XorModel g = xor_model_alloc();

	kedos_matrix_rand(m.w1, 0, 1);
	kedos_matrix_rand(m.b1, 0, 1);
	kedos_matrix_rand(m.w2, 0, 1);
	kedos_matrix_rand(m.b2, 0, 1);

	real_t eps = 1e-1;
	real_t rate = 1e-1;

	printf("cost: %f\n", xor_model_cost(m, input, output));
	for (size_t i = 0; i < 10000; i++) {
		finite_difference(m, g, eps, input, output);
		xor_model_learn(m, g, rate);
		printf("%zu: cost: %f\n", i, xor_model_cost(m, input, output));
	}

	printf("-------------------------------\n");

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 2; j++) {
			K_MATRIX_AT(m.a0, 0, 0) = i;
			K_MATRIX_AT(m.a0, 0, 1) = j;
			forward_xor(m);
			real_t y = *m.a2.elems;

			printf("%zu ^ %zu = %f\n", i, j, y);
		}
	}

	return 0;
}