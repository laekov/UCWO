#ifndef UCX_CTRL_HH
#define UCX_CTRL_HH

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <vector>


namespace ucxCtrl {

const int nth = 4;

typedef std::pair<size_t, void*> buf_t;
typedef std::pair<ucp_rkey_h, void*> rmem_t;

void init();
void connect();
buf_t mmap(void* &addr, size_t length);
void exposeMemory(buf_t rkey, void* addr, int target);
std::vector<rmem_t> peepMemory(int source);
void yield(int i);

void putSync(int target, rmem_t mem, size_t offset, const void* data, size_t length);
void putAsync(int target, rmem_t mem, size_t offset, const void* data, size_t length);
void getSync(int target, rmem_t mem, size_t offset, void* data, size_t length);

void flush();

};  // namespace ucxCtrl

#endif  // UCX_CTRL_HH
