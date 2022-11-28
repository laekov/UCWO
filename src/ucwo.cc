#include <cassert>
#include <cstring>
#include "ucwo.hh"

namespace UCWO {

static void send_handler(void *request, ucs_status_t status) {
}

void Worker::yield() {
    ucp_worker_progress(this->h);
}

void Worker::flush() {
    auto request = ucp_worker_flush_nb(this->h, 0, send_handler);
    this->wait(request);
}

void Worker::work() {
    this->end_work = 0;
    this->th = new std::thread([=]() {
        while (!this->end_work) {
            yield();
        }
    });
}

void Worker::stop() {
    if (th) {
        this->end_work = 1;
        this->th->join();
        delete this->th;
        this->th = 0;
    }
}

void World::init() {
    MPI_Comm_rank(comm, &this->rank);
    MPI_Comm_size(comm, &this->world_size);

    ucp_config_t *config;
    auto status = ucp_config_read(NULL, NULL, &config);
    assert(status == UCS_OK);
    status = ucp_config_modify(config, "PROTO_ENABLE", "y");
    assert(status == UCS_OK);

    ucp_params_t ucp_params;
    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features     = UCP_FEATURE_RMA | UCP_FEATURE_TAG;

    status = ucp_init(&ucp_params, config, &ctx);
    assert(status == UCS_OK);
    ucp_config_release(config);
}

Worker* World::newWorker(bool mt) {
    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = mt ? UCS_THREAD_MODE_MULTI : UCS_THREAD_MODE_SINGLE;

    auto worker = new Worker(world_size);
    auto status = ucp_worker_create(ctx, &worker_params, &worker->h);
    assert(status == UCS_OK);

    worker->world = this;
    workers.push_back(worker);
    if (mt) {
        worker->work();
    }
    if (this->remote_addrs.size() == world_size) {
        worker->connect(this->remote_addrs);
    }
    return worker;
}

void Worker::connect(const std::vector<Buffer>& remote_addrs) {
    int world_size = this->world->worldSize();
    assert(remote_addrs.size() == world_size);
    for (int i = 0; i < world_size; ++i) {
        ucp_ep_params_t ep_params;
        ucs_status_t ep_status   = UCS_OK;
        ep_params.field_mask     = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                   UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        ep_params.address        = (ucp_address_t*)remote_addrs[i].buf;
        ep_params.err_mode       = UCP_ERR_HANDLING_MODE_NONE;

        auto status = ucp_ep_create(h, &ep_params, &eps[i]);
        assert(status == UCS_OK);
    }
}

Buffer World::mmap(void* &addr, size_t length, ucs_memory_type_t memtype) {
    ucp_mem_map_params_t mmap_params;
    ucp_mem_h mem_h;
    mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                             UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                             UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                             UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_params.length = length;
    mmap_params.address = addr;
    mmap_params.flags = 0;
    if (!addr) {
        mmap_params.flags |= UCP_MEM_MAP_ALLOCATE;
    }
    // mmap_params.memory_type = UCS_MEMORY_TYPE_HOST;
    mmap_params.memory_type = memtype;
    auto status = ucp_mem_map(this->ctx, &mmap_params, &mem_h);
    assert(status == UCS_OK);

    Buffer rk;
    status = ucp_rkey_pack(this->ctx, mem_h, &rk.buf, &rk.size);
    assert(status == UCS_OK);

    ucp_mem_attr_t mem_attrs;
    mem_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS
        | UCP_MEM_ATTR_FIELD_LENGTH
        | UCP_MEM_ATTR_FIELD_MEM_TYPE;
    status = ucp_mem_query(mem_h, &mem_attrs);
    assert(status == UCS_OK);
    addr = (void*)mem_attrs.address;
    /*
    if (mem_attrs.mem_type == UCS_MEMORY_TYPE_CUDA) {
        fprintf(stderr, "Mapping cuda memory\n");
    } else if (mem_attrs.mem_type == UCS_MEMORY_TYPE_HOST) {
        fprintf(stderr, "Mapping host memory\n");
    } else if (mem_attrs.mem_type == UCS_MEMORY_TYPE_UNKNOWN) {
        fprintf(stderr, "Mapping unknown memory\n");
    }
    */
    return rk;
}

const int rkey_cnk_size = 256 << 10;  // 256K

void World::connect() {
    Buffer b0;
    b0.buf = 0;
    auto rkey = this->mmap(b0.buf, rkey_cnk_size, UCS_MEMORY_TYPE_HOST);
    b0.size = 0;
    this->rkbufs.push_back(b0);

    auto worker = this->newWorker();
    ucp_address_t *local_addr;
    size_t local_addr_len;
    auto status = ucp_worker_get_address(worker->h, &local_addr, &local_addr_len);
    assert(status == UCS_OK);
    Buffer peer_addr;
    for (int i = 0; i < world_size; ++i) {
        if (i == rank) {
            peer_addr.buf = local_addr;
            peer_addr.size = local_addr_len;
        }
        MPI_Bcast(&peer_addr.size, sizeof(size_t), MPI_CHAR, i, comm);
        if (i != rank) {
            peer_addr.buf = (ucp_address_t*)malloc(peer_addr.size);
        }
        MPI_Bcast(peer_addr.buf, peer_addr.size, MPI_CHAR, i, comm);
        this->remote_addrs.push_back(peer_addr);
    }
    worker->connect(this->remote_addrs);

    for (int i = 0; i < world_size; ++i) {
        void *rkey_buffer, *addr;
        size_t rkey_buffer_size;
        if (i == rank) {
            addr = b0.buf;
            rkey_buffer = rkey.buf;
            rkey_buffer_size = rkey.size;
        }
        MPI_Bcast(&rkey_buffer_size, sizeof(size_t), MPI_CHAR, i, comm);
        if (i != rank) {
            rkey_buffer = malloc(rkey_buffer_size);
        }
        MPI_Bcast(rkey_buffer, rkey_buffer_size, MPI_CHAR, i, comm);
        MPI_Bcast(&addr, sizeof(void*), MPI_CHAR, i, comm);

        RemoteMemory rm;
        rm.addr = addr;
        if (i != rank) {
            auto status = ucp_ep_rkey_unpack(worker->eps[i],
                    rkey_buffer, &rm.rkey);
            assert(status == UCS_OK);
            free(rkey_buffer);
        }
        remote_blks[i].metakeys.push_back(rm);
    }
}

void Worker::ensureBlock(int target, int block_idx) {
    while (this->remote_blocks[target].size() <= block_idx) {
        int i = this->remote_blocks[target].size();
        auto rkey_buf = world->getBlockRkey(target, i);
        RemoteMemory rm;
        auto status = ucp_ep_rkey_unpack(eps[target], rkey_buf.buf, &rm.rkey);
        assert(status == UCS_OK);
        memcpy(&rm.addr, (char*)rkey_buf.buf + rkey_buf.size, sizeof(size_t));
        this->remote_blocks[target].push_back(rm);
    }
}

Worker::Request Worker::get(int target, size_t block_idx, size_t offset,
        void* out, size_t length) {
    this->ensureBlock(target, block_idx);
    auto mem = this->remote_blocks[target][block_idx];
    auto request = ucp_get_nb(this->eps[target], out, length,
            (size_t)mem.addr + offset, mem.rkey, send_handler);
    return Request(this, request);
}

Worker::Request Worker::put(int target, size_t block_idx, size_t offset,
        void* in, size_t length) {
    this->ensureBlock(target, block_idx);
    auto mem = this->remote_blocks[target][block_idx];
    auto request = ucp_put_nb(this->eps[target], in, length,
            (size_t)mem.addr + offset, mem.rkey, send_handler);
    return Request(this, request);
}

ucs_status_t Worker::wait(ucs_status_ptr_t request) {
    if (request == NULL) {
        return UCS_OK;
    } else if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    } else {
        ucs_status_t status;
        do {
            ucp_worker_progress(h);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(request);
        return status;
    }
}

void Worker::getSync(int target, RemoteMemory mem, size_t offset,
        void* out, size_t size) {
    auto request = ucp_get_nb(this->eps[target], out, size,
            (size_t)mem.addr + offset, mem.rkey, send_handler);
    wait(request);
}

void RemoteBlocks::extendBlocks(Worker* w, int target) {
    Buffer rkey;
    size_t meta[2];
    w->getSync(target, metakeys[this->rki], this->rko, meta, 2 * sizeof(size_t));
    assert(meta[1] != -1ull); // assume that there is only one meta block
    rkey.size = meta[0];
    rkey.buf = malloc(rkey.size + sizeof(size_t));
    w->getSync(target, metakeys[this->rki], this->rko + 2 * sizeof(size_t),
            rkey.buf, rkey.size);
    // Put address at the end of rkey buffer
    memcpy((char*)rkey.buf + rkey.size, &meta[1], sizeof(size_t));
    this->bufs.push_back(rkey);
    this->rko += rkey.size + 2 * sizeof(size_t);
}

void* World::expose(void* addr, size_t length, ucs_memory_type_t memtype) {
    Buffer rkey = this->mmap(addr, length, memtype);
    std::lock_guard<std::mutex> lck(rkb_mtx);
    // Malloc rkbufs if needed 
    // This seems to be unnccesssary due to the limit of shmem segments
    Buffer rkbuf = *rkbufs.rbegin();
    if (rkbuf.size + rkey.size * 2 >= rkey_cnk_size) {
        Buffer new_rkbuf;
        new_rkbuf.buf = 0;
        Buffer new_rk = this->mmap(new_rkbuf.buf, rkey_cnk_size, UCS_MEMORY_TYPE_HOST);
        if (new_rk.size + 2 * sizeof(size_t) +  rkbuf.size > rkey_cnk_size) {
            fprintf(stderr, "Rkey linked list prolong failed\n");
            assert(false);
        }
        char* p = (char*)rkbuf.buf + rkbuf.size;
        memcpy(p, &new_rk.size, sizeof(size_t));
        memset(p + sizeof(size_t), 0xff, sizeof(size_t));
        memcpy(p + sizeof(size_t) * 2, new_rk.buf, new_rk.size);
        new_rkbuf.size = 0;
        rkbufs.push_back(new_rkbuf);
    }
    // Copy rkey to the rkbuf
    char* p = (char*)rkbufs.rbegin()->buf + rkbufs.rbegin()->size;
    memcpy(p, &rkey.size, sizeof(size_t));
    // memset(p + sizeof(size_t), 0, sizeof(size_t));
    memcpy(p + sizeof(size_t), &addr, sizeof(size_t));
    memcpy(p + sizeof(size_t) * 2, rkey.buf, rkey.size);
    rkbufs.rbegin()->size += rkey.size + 2 * sizeof(size_t);
    // TODO: Register the address in some local array
    return addr;
}

Buffer World::getBlockRkey(int target, int idx) {
    std::lock_guard<std::mutex> lck(remote_blks[target].mtx);
    while (remote_blks[target].bufs.size() <= idx) {
        remote_blks[target].extendBlocks(this->workers[0], target);
    }
    return remote_blks[target].bufs[idx];
}

Worker* World::worker(int idx) {
    assert(idx < workers.size());
    return workers[idx];
}

};
