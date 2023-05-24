#define KEDOS_IMPLEMENTATION
#include "../include/kedos.h"

#include <time.h>

real_t SUM_DATA[] = {
	0, 0,  0, 0,  0, 0,
	0, 0,  0, 1,  0, 1,
	0, 1,  0, 1,  1, 0,
	0, 1,  1, 0,  1, 1,
};

real_t XOR_DATA[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 0,
};

real_t OR_DATA[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 1,
};

int main(void) {
	srand(time(0));

	real_t* data = XOR_DATA;
	Matrix input = { .rows = 4, .cols = 2, .stride = 3, .elems = data };
	Matrix output = { .rows = 4, .cols = 1, .stride = 3, .elems = &data[2] };

	size_t arch[] = { 2, 4, 1 };
	NeuralNetwork nn = kedos_neural_network_alloc(arch, K_ARRAY_COUNT(arch));
	NeuralNetwork g = kedos_neural_network_alloc(arch, K_ARRAY_COUNT(arch));
	kedos_neural_network_rand(nn, 0, 1);

	real_t epsilon = 1e-1;
	real_t learning_rate = 1e-1;

	printf("cost %f\n", kedos_neural_network_cost(nn, input, output));
	for (size_t i = 0; i < 20*1000; i++){
		kedos_neural_network_finite_difference(nn, g, epsilon, input, output);
		kedos_neural_network_learn(nn, g, learning_rate);
		printf("%zu cost %f\n", i, kedos_neural_network_cost(nn, input, output));
	}

	printf("--------------------------------------\n");

	K_NETWORK_PRINT(nn);

	printf("--------------------------------------\n");

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 2; j++) {
			K_MATRIX_AT(K_NETWORK_INPUT(nn), 0, 0) = i;
			K_MATRIX_AT(K_NETWORK_INPUT(nn), 0, 1) = j;
			kedos_neural_network_forward(nn);
			printf("%zu ^ %zu = %f\n", i, j, K_MATRIX_AT(K_NETWORK_OUTPUT(nn), 0, 0));
		}
	}

	kedos_neural_network_free(nn);
	kedos_neural_network_free(g);
	return 0;
}