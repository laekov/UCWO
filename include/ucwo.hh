// UCWO: UCX Wrapper for One-sided communication

#ifndef UCW_HH
#define UCW_HH

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <mpi.h>
#include <vector>
#include <thread>
#include <mutex>


namespace UCWO {

struct RemoteMemory {
    ucp_rkey_h rkey;
    void* addr;
};

struct Buffer {
    void* buf;
    size_t size;
};

class Worker {
protected:
    ucp_worker_h h;
    std::vector<ucp_ep_h> eps;
    std::thread* th;
    bool end_work;
    class World* world;
    std::vector<std::vector<RemoteMemory>> remote_blocks;

    friend class World;

protected:
    void ensureBlock(int target, int block_idx);
public:
    Worker(int world_size): eps(world_size), th(0), remote_blocks(world_size) {}
    ~Worker() {
        this->stop();
    }
    void work();
    void yield();
    void stop();
    ucs_status_ptr_t get(
            int target, size_t block_idx, size_t offset, void* data, size_t);
    ucs_status_ptr_t put(
            int target, size_t block_idx, size_t offset, void* data, size_t);
    void getSync(int target, RemoteMemory, size_t offset, void* out, size_t size);
    ucs_status_t wait(ucs_status_ptr_t request);
    void flush();
};

struct RemoteBlocks {
    std::vector<Buffer> bufs;
    std::vector<RemoteMemory> metakeys;
    std::mutex mtx;
    size_t rki, rko;
    RemoteBlocks(): rki(0), rko(0) {}
    void extendBlocks(Worker*, int target);
};

class World {
protected:
    int rank, world_size;
    MPI_Comm comm;
    ucp_context_h ctx;
    std::vector<Worker*> workers;
    std::vector<Buffer> rkbufs;
    std::vector<RemoteBlocks> remote_blks;
    std::mutex rkb_mtx;
public:
    World(MPI_Comm comm_): comm(comm_) {
        this->init();
        remote_blks = std::vector<RemoteBlocks>(world_size);
        this->connect();
    }
    ~World() {
        for (auto& w: workers) {
            delete w;
        }
    }
    void init();
    Worker* newWorker();
    void connect();
    Buffer mmap(void*&, size_t);
    void* expose(void*, size_t);
    Worker* worker(int idx);
    Buffer getBlockRkey(int target, int idx);
};

};  // namespace UCWO

#endif  // UCWO_HH
