#include <mpi.h>

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include <omp.h>

#include "ucxctrl.hh"


namespace ucxCtrl {

ucp_context_h ucp_context;
ucp_worker_h ucp_worker[nth];

ucp_address_t *local_addr[nth];
size_t local_addr_len[nth];

ucp_ep_h* eps;

int rank, world_size;

static void request_init(void* r) {
    *(int*)r = 0;
}

void init() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ucp_config_t *config;
    auto status = ucp_config_read(NULL, NULL, &config);
    assert(status == UCS_OK);

    ucp_params_t ucp_params;
    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                              UCP_PARAM_FIELD_REQUEST_SIZE |
                              UCP_PARAM_FIELD_REQUEST_INIT;
    ucp_params.features     = UCP_FEATURE_RMA | UCP_FEATURE_TAG;
    ucp_params.request_size    = sizeof(int);
    ucp_params.request_init    = request_init;

    status = ucp_init(&ucp_params, config, &ucp_context);
    assert(status == UCS_OK);
    ucp_config_release(config);

    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    for (int i = 0; i < nth; ++i) {
        status = ucp_worker_create(ucp_context, &worker_params, ucp_worker + i);
        assert(status == UCS_OK);

        status = ucp_worker_get_address(ucp_worker[i], local_addr + i,
                local_addr_len + i);
        assert(status == UCS_OK);
    }
}

void connect() {
    ucp_address_t *peer_addr;
    size_t peer_addr_len;
    eps = new ucp_ep_h[world_size * nth];
    for (int i = 0; i < world_size; ++i) {
        for (int j = 0; j < nth; ++j) {
            if (i == rank) {
                peer_addr = local_addr[j];
                peer_addr_len = local_addr_len[j];
            }
            MPI_Bcast(&peer_addr_len, sizeof(size_t), MPI_CHAR, i, MPI_COMM_WORLD);
            if (i != rank) {
                peer_addr = (ucp_address_t*)malloc(peer_addr_len);
            }
            MPI_Bcast(peer_addr, peer_addr_len, MPI_CHAR, i, MPI_COMM_WORLD);
            if (i != rank) {
                ucp_ep_params_t ep_params;
                ucs_status_t ep_status   = UCS_OK;
                ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
                ep_params.address         = peer_addr;
                ep_params.err_mode        = UCP_ERR_HANDLING_MODE_NONE;
                for (int j = 0; j < nth; ++j) {
                    auto status = ucp_ep_create(ucp_worker[j], &ep_params, eps + i * nth + j);
                    assert(status == UCS_OK);
                }
                free(peer_addr);
            }
        }
    }
}

buf_t mmap(void* &addr, size_t length) {
    ucp_mem_map_params_t mmap_params;
    ucp_mem_h mem_h;
    mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                             UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                             UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                             UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_params.length = length;
    mmap_params.address = addr;
    if (!addr) {
        mmap_params.flags = UCP_MEM_MAP_ALLOCATE;
    }
    mmap_params.memory_type = UCS_MEMORY_TYPE_HOST;
    auto status = ucp_mem_map(ucp_context, &mmap_params, &mem_h);
    assert(status == UCS_OK);

    void* rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(ucp_context, mem_h, &rkey_buffer, &rkey_buffer_size);
    assert(status == UCS_OK);

    ucp_mem_attr_t mem_attrs;
    mem_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH;
    status = ucp_mem_query(mem_h, &mem_attrs);
    assert(status == UCS_OK);
    addr = (void*)mem_attrs.address;
    return buf_t(rkey_buffer_size, rkey_buffer);
}

void exposeMemory(buf_t rkey, void* addr, int target) {
    MPI_Send(&rkey.first, sizeof(size_t), MPI_CHAR, target, 125, MPI_COMM_WORLD);
    MPI_Send(rkey.second, rkey.first, MPI_CHAR, target, 126, MPI_COMM_WORLD);
    MPI_Send(&addr, sizeof(void*), MPI_CHAR, target, 127, MPI_COMM_WORLD);
}


std::vector<rmem_t> peepMemory(int source) {
    void* rkey_buffer;
    size_t rkey_buffer_size;
    MPI_Recv(&rkey_buffer_size, sizeof(size_t), MPI_CHAR, 0, 125,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    rkey_buffer = malloc(rkey_buffer_size);
    MPI_Recv(rkey_buffer, rkey_buffer_size, MPI_CHAR, 0, 126,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    size_t remote_addr;
    MPI_Recv(&remote_addr, sizeof(size_t), MPI_CHAR, 0, 127,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<rmem_t> out;
    for (int i = 0; i < nth; ++i) {
        ucp_rkey_h rkey;
        auto status = ucp_ep_rkey_unpack(eps[source * nth + i], rkey_buffer, &rkey);
        assert(status == UCS_OK);
        out.push_back(rmem_t(rkey, (void*)remote_addr));
        fprintf(stderr, "Rank %d rkey %d: %016x\n", rank, i, rkey);
    }
    free(rkey_buffer);
    return out;
}

void yield(int i) {
    ucp_worker_progress(ucp_worker[i]);
}

static void send_handler(void *request, ucs_status_t status) {
    *(int*)request = 1;
}

void putSync(int target, rmem_t mem, size_t offset, const void* data, size_t length) {
    // auto status = ucp_put_nbi(eps[target], data, length,
            // (size_t)mem.second + offset, mem.first);
    auto status = ucp_put_nb(eps[target * nth], data, length,
            (size_t)mem.second + offset, mem.first, send_handler);
    if (UCS_PTR_IS_ERR(status)) {
        fprintf(stderr, "UCX Error\n");
    }
    while (status && *(int*)status == 0) {
        yield(0);
    }
}

void putAsync(int target, rmem_t mem, size_t offset, const void* data, size_t length) {
    // auto status = ucp_put_nbi(eps[target], data, length,
            // (size_t)mem.second + offset, mem.first);
    int thi = omp_get_thread_num();
    auto status = ucp_put_nb(eps[target * nth + thi], data, length,
            (size_t)mem.second + offset, mem.first, send_handler);
}

void flush() {
    for (int i = 0; i < nth; ++i) {
        auto status = ucp_worker_flush_nb(ucp_worker[i], 0, send_handler);
        while (status && *(int*)status == 0) {
            yield(i);
        }
    }
}

void getSync(int target, rmem_t mem, size_t offset, void* data, size_t length) {
    auto status = ucp_get_nb(eps[target * nth], data, length,
            (size_t)mem.second + offset, mem.first, send_handler);
    while (status && *(int*)status == 0) {
        yield(0);
    }
}

};  // ucxCtrl
