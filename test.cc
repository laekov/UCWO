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
    x = (char*)world.expose(0, 1024);
    x[0] = 0;
    MPI_Barrier(MPI_COMM_WORLD);

    auto w = world.newWorker();

    char data[10];
    data[0] = 'a' + rank;
    auto rp = w->put(rank ^ 1, 0, 0, data, 1);
    while (!x[0]) {
        w->yield();
    }
    fprintf(stderr, "Rank %d Got %c\n", rank, x[0]);
    w->wait(rp);

    MPI_Finalize();
}
