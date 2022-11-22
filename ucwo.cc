#include <cassert>
#include <cstring>
#include "ucwo.hh"

namespace UCWO {

void Worker::work() {
    this->end_work = 0;
    this->th = new std::thread([=]() {
        while (!this->end_work) {
            ucp_worker_progress(this->h);
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

    ucp_params_t ucp_params;
    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features     = UCP_FEATURE_RMA | UCP_FEATURE_TAG;

    status = ucp_init(&ucp_params, config, &ctx);
    assert(status == UCS_OK);
    ucp_config_release(config);
}

void World::newWorker() {
    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    auto worker = new Worker(world_size);
    auto status = ucp_worker_create(ctx, &worker_params, &worker->h);
    assert(status == UCS_OK);

    ucp_address_t *local_addr;
    size_t local_addr_len;
    status = ucp_worker_get_address(worker->h, &local_addr, &local_addr_len);
    assert(status == UCS_OK);

    ucp_address_t *peer_addr;
    size_t peer_addr_len;
    for (int i = 0; i < world_size; ++i) {
        if (i == rank) {
            peer_addr = local_addr;
            peer_addr_len = local_addr_len;
        }
        MPI_Bcast(&peer_addr_len, sizeof(size_t), MPI_CHAR, i, comm);
        if (i != rank) {
            peer_addr = (ucp_address_t*)malloc(peer_addr_len);
        }
        MPI_Bcast(peer_addr, peer_addr_len, MPI_CHAR, i, comm);
        if (i != rank) {
            ucp_ep_params_t ep_params;
            ucs_status_t ep_status   = UCS_OK;
            ep_params.field_mask     = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                       UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
            ep_params.address        = peer_addr;
            ep_params.err_mode       = UCP_ERR_HANDLING_MODE_NONE;

            auto status = ucp_ep_create(worker->h, &ep_params, &worker->eps[i]);
            assert(status == UCS_OK);

            free(peer_addr);
        }
    }
    workers.push_back(worker);
    worker->work();
}

RKey World::mmap(void* &addr, size_t length) {
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
    mmap_params.memory_type = UCS_MEMORY_TYPE_HOST;
    auto status = ucp_mem_map(this->ctx, &mmap_params, &mem_h);
    assert(status == UCS_OK);

    RKey rk;
    status = ucp_rkey_pack(this->ctx, mem_h, &rk.buf, &rk.size);
    assert(status == UCS_OK);

    ucp_mem_attr_t mem_attrs;
    mem_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH;
    status = ucp_mem_query(mem_h, &mem_attrs);
    assert(status == UCS_OK);
    addr = (void*)mem_attrs.address;
    return rk;
}

const int rkey_cnk_size = 256 << 10;  // 256K

void World::connect() {
    RKeyBuffer b0;
    auto rkey = this->mmap(b0.buf, rkey_cnk_size);
    memcpy(b0.buf, &rkey.size, sizeof(size_t));
    memset((char*)b0.buf + sizeof(size_t), 0, sizeof(size_t));
    memcpy((char*)b0.buf + sizeof(size_t) * 2, rkey.buf, rkey.size);
    b0.size = rkey.size + sizeof(size_t) * 2;
    this->rkbufs.push_back(b0);

    this->newWorker();
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
        if (i != rank) {
            RemoteMemory rm;
            auto status = ucp_ep_rkey_unpack(this->workers[0]->eps[i],
                    rkey_buffer, &rm.rkey);
            assert(status == UCS_OK);
            rm.addr = addr;
            remote_rkeys[i].push_back(rm);
            free(rkey_buffer);
        } else {
            RemoteMemory rm;
            rm.addr = b0.buf;
            remote_rkeys[i].push_back(rm);
        }
    }
}

};
