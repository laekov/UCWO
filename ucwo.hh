// UCWO: UCX Wrapper for One-sided communication

#ifndef UCW_HH
#define UCW_HH

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <mpi.h>
#include <vector>
#include <thread>


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

    friend class World;
public:
    Worker(int world_size): eps(world_size), th(0) {}
    ~Worker() {
        this->stop();
    }
    void work();
    void stop();
    void get(int target, size_t chunk_idx, size_t offset, void* data, size_t);
};

class World {
protected:
    int rank, world_size;
    MPI_Comm comm;
    ucp_context_h ctx;
    std::vector<Worker*> workers;
    std::vector<Buffer> rkbufs;
    std::vector<std::vector<RemoteMemory>> remote_rkeys;
public:
    World(MPI_Comm comm_): comm(comm_) {
        this->init();
        remote_rkeys.resize(world_size);
        this->connect();
    }
    ~World() {
        for (auto& w: workers) {
            delete w;
        }
    }
    void init();
    void newWorker();
    void connect();
    Buffer mmap(void*&, size_t);
    void expose(void*&, size_t);
    Buffer getBlockRkey(int target, int idx);
};
/*
buf_t mmap(void* &addr, size_t length);
void exposeMemory(buf_t rkey, void* addr, int target);
std::vector<rmem_t> peepMemory(int source);
void yield(int i);

void putSync(int target, rmem_t mem, size_t offset, const void* data, size_t length);
void putAsync(int target, rmem_t mem, size_t offset, const void* data, size_t length);
void getSync(int target, rmem_t mem, size_t offset, void* data, size_t length);

void flush();
*/

};  // namespace UCWO

#endif  // UCWO_HH
