#include "ucwo.hh"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <mpi.h>
#include <assert.h>
#include <thread>
#include <omp.h>
#include "timer.hh"

const size_t n = 1 << 20;
const size_t bs = 64;
const size_t nt = 10;
const int nth = 4;

int main() {
    MPI_Init(0, 0);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(world_size == 2);

    UCWO::World world(MPI_COMM_WORLD);
    std::vector<UCWO::Worker*> workers;
    for (int i = 0; i < nth; ++i) {
        workers.push_back(world.newWorker());
    }

    void* addr = 0;
    size_t len = n * sizeof(int);
    if (rank == 0) {
        addr = world.expose(addr, len);
        MPI_Barrier(MPI_COMM_WORLD);

        int* ptr = (int*)addr;
        for (size_t i = 0; i < nt; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            fprintf(stderr, "Put %d output %x\n", i, ptr[0]);
        }
    } else {
        int* p = new int[n * nt];
        memset(p, 12, n * nt * sizeof(int));
        for (int i = 0; i < nt; ++i) {
            p[i * n] = (i + 1) * 16;
        }
        int* x = new int[n / bs];
        for (int i = 0; i < n / bs; ++i) {
            x[i] = i;
        }
        std::random_shuffle(x, x + n / bs);

        MPI_Barrier(MPI_COMM_WORLD);
        double tott = 0;
        fprintf(stderr, "Starting bench\n");
        for (size_t i = 0; i < nt; ++i) {
            timestamp(tb);
#pragma omp parallel for num_threads(nth)
            for (int j = 0; j < n / bs; ++j) {
                int thi = omp_get_thread_num();
                workers[thi]->put(0, 0, x[j] * bs * sizeof(int),
                            p + i * n + j * bs, bs * sizeof(int)).wait();
            }
            for (auto& w: workers) {
                w->flush();
            }
            timestamp(te);

            MPI_Barrier(MPI_COMM_WORLD);
            auto dur = getDuration(tb, te);
            auto bw = n * sizeof(int) / dur;
            fprintf(stderr, "Put %d time %.3lf ms bw %.3lf GBps\n",
                    i, dur * 1e3, bw * 1e-9);
            if (i) {
                tott += dur;
            }
        }
        auto meant = tott / (nt - 1);
        auto bw = n * sizeof(int) / meant;
        fprintf(stderr, "Mean time %.3lf bw %.3lf GBps\n",
                meant * 1e3, bw * 1e-9);
    }
    sleep(1);
    MPI_Finalize();
}
