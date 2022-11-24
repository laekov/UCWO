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
    char *x, *h;
    cudaMalloc(&x, 1024);
    world.expose(x, 1024, UCS_MEMORY_TYPE_CUDA);
    // x = (char*)world.expose(0, 1024, UCS_MEMORY_TYPE_CUDA);
    h = (char*)world.expose(0, 1024);
    h[0] = 'A' + rank;
    // MPI_Barrier(MPI_COMM_WORLD);

    auto w = world.newWorker();

    char data[10];
    char y[1024];
    data[0] = 'a' + rank;
    w->put(rank ^ 1, 0, 0, data, 1).wait();
    w->get(rank ^ 1, 1, 0, x + 1, 1).wait();
    // w->get(rank ^ 1, 1, 0, y, 1).wait();

    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(y, x, 1024, cudaMemcpyDeviceToHost);
    fprintf(stderr, "Rank %d got %c %c\n", rank, y[0], y[1]);

    MPI_Finalize();
}
