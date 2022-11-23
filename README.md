# UCWO: UCX Wrapper for One-sideds

A CPP wrapper based on MPI and UCX to conveniently conduct one-sided communication.

## Terms

A **block** of local memory on each process can be exposed so that it can be accessed remotely from another process.
On each process, blocks are sequentially numbered according to the order when they are exposed.
Other processes are able to visit these blocks using the index.

## Usage

* A `World` is a thread-safe base class to provide connection and memory mapping. It is binded to an MPI comm.
* A `World` can `expose` multiple memory blocks.
* Multiple `Worker` instances are created by a `World` to conduct parallel communication.
* Each worker independently initiate `put`, `get`, or `flush` operations.

## Notes

*  A worker itself is not thread-safe.
* `newWorker` routine of `World` is a collective operation that involves all processes to involve. Only workers in the same `newWorker` call are connected. (Maybe optimize this in the future)
