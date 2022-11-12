#include "ucxctrl.hh"
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <assert.h>

int main() {
    MPI_Init(0, 0);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(world_size == 2);

    ucxCtrl::init();
    ucxCtrl::connect();

    void* addr = 0;
    size_t len = 1024;
    if (rank == 0) {
        auto rkb = ucxCtrl::mmap(addr, len);

        int* ptr = (int*)addr;
        ptr[0] = 111;

        ucxCtrl::exposeMemory(rkb, addr, 1);

        while (ptr[0] == 111) {
            ucxCtrl::yield();
        }
        fprintf(stderr, "Seen %d\n", ptr[0]);
    } else {
        auto rm = ucxCtrl::peepMemory(0);
        int a[10];
        ucxCtrl::getSync(0, rm, 0, a, sizeof(int));
        fprintf(stderr, "Fetched %d\n", a[0]);
        a[0] += 1;
        ucxCtrl::putSync(0, rm, 0, a, sizeof(int));
    }
    sleep(1);
    MPI_Finalize();
}
