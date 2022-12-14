#include "ucwo.hh"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <assert.h>
#include <thread>
#include "timer.hh"

const size_t n = 1 << 25;
const size_t nt = 50;

int main() {
    MPI_Init(0, 0);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(world_size == 2);

    UCWO::World world(MPI_COMM_WORLD);

    void* addr = 0;
    size_t len = n * sizeof(int);
    if (rank == 0) {
        addr = world.expose(addr, len);
        int* ptr = (int*)addr;

        for (size_t i = 0; i < nt; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            fprintf(stderr, "Put %d output %x\n", i, ptr[0]);
        }
    } else {
        auto worker = world.worker(0);

        int* p = new int[n * nt];
        memset(p, 12, n * nt * sizeof(int));
        for (int i = 0; i < nt; ++i) {
            p[i * n] = (i + 1) * 16;
        }
        double tott = 0;
        fprintf(stderr, "Starting bench\n");
        for (size_t i = 0; i < nt; ++i) {
            timestamp(tb);
            worker->put(0, 0, 0, p + i * n, n * sizeof(int)).wait();
            timestamp(te);

            MPI_Barrier(MPI_COMM_WORLD);
            auto dur = getDuration(tb, te);
            auto bw = (double)n * sizeof(int) / dur;
            fprintf(stderr, "Put %d time %.3lf ms bw %.3lf GBps\n",
                i, dur * 1e3, bw * 1e-9);
            if (i) {
                tott += dur;
            }
        }
        auto meant = tott / (nt - 1);
        auto bw = n * sizeof(int) / meant;
        fprintf(stderr, "Mean time %.3lf bw %.3lf GBps\n", meant * 1e3, bw * 1e-9);
    }
    sleep(1);
    MPI_Finalize();
}
