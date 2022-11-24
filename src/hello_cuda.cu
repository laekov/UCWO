#include "ucwo.hh"
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <assert.h>

int main() {
    MPI_Init(0, 0);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    UCWO::World world(MPI_COMM_WORLD);
    char* x;
    cudaMalloc(&x, 1024);
    (char*)world.expose(x, 1024, UCS_MEMORY_TYPE_CUDA);
    MPI_Barrier(MPI_COMM_WORLD);

    auto w = world.newWorker();

    char data[10];
    data[0] = 'a' + rank;
    w->put(rank ^ 1, 0, 0, data, 1).wait();

    MPI_Barrier(MPI_COMM_WORLD);

    char y[1024];
    cudaMemcpy(y, x, 1024, cudaMemcpyDeviceToHost);
    fprintf(stderr, "Rank %d got %c\n", rank, y[0]);

    MPI_Finalize();
}
