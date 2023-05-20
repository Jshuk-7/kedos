#define KEDOS_IMPLEMENTATION
#include "kedos.h"

int main(void) {
	srand(time(0));
	Matrix mat = kedos_matrix_alloc(1, 2);
	kedos_matrix_rand(mat, 5, 10);
	Matrix mat2 = kedos_matrix_alloc(2, 2);
	kedos_matrix_fill(mat2, 0);
	Matrix res = kedos_matrix_alloc(1, 2);
	kedos_matrix_dot(res, mat, mat2);
	kedos_matrix_display(mat);
	printf("--------------------------\n");
	kedos_matrix_display(res);
	return 0;
}