#ifndef __ERRORS_H__
#define __ERRORS_H__
#include <cstdio>

static void handleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))
#endif // __ERRORS_H__
