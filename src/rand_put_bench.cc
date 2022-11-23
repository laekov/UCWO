#include "ucxctrl.hh"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <mpi.h>
#include <assert.h>
#include <thread>
#include <omp.h>
#include "timer.hh"

const size_t n = 1 << 22;
const size_t bs = 64;
const size_t nt = 100;

int main() {
    MPI_Init(0, 0);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(world_size == 2);

    fprintf(stderr, "Rank %d init\n", rank);
    ucxCtrl::init();
    fprintf(stderr, "Rank %d connect\n", rank);
    ucxCtrl::connect();
    fprintf(stderr, "Rank %d connected\n", rank);

    void* addr = 0;
    size_t len = n * sizeof(int);
    if (rank == 0) {
        auto rkb = ucxCtrl::mmap(addr, len);
        int* ptr = (int*)addr;
        ucxCtrl::exposeMemory(rkb, addr, 1);

        int join_worker = 0;
        std::vector<std::thread> ths;
        for (int i = 0; i < ucxCtrl::nth; ++i) {
            ths.emplace_back([&join_worker, i]() {
                while (!join_worker) {
                    ucxCtrl::yield(i);
                }
            });
        }
        for (size_t i = 0; i < nt; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            fprintf(stderr, "Put %d output %x\n", i, ptr[0]);
        }
        join_worker = 1;
        for (auto& th: ths) {
            th.join();
        }
    } else {
        auto rm = ucxCtrl::peepMemory(0);
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

        double tott = 0;
        fprintf(stderr, "Starting bench\n");
        for (size_t i = 0; i < nt; ++i) {
            timestamp(tb);
#pragma omp parallel for num_threads(ucxCtrl::nth)
            for (int j = 0; j < n / bs; ++j) {
                int thi = omp_get_thread_num();
                ucxCtrl::putAsync(0, rm[thi], x[j] * bs * sizeof(int),
                        p + i * n + j * bs, bs * sizeof(int));
            }
            ucxCtrl::flush();
            timestamp(te);

            MPI_Barrier(MPI_COMM_WORLD);
            auto dur = getDuration(tb, te);
            auto bw = n * sizeof(int) / dur;
            fprintf(stderr, "Put %d time %.3lf ms bw %.3lf GBps\n", i, dur * 1e3, bw * 1e-9);
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
