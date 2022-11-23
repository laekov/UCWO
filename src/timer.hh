#include <chrono>

using timestamp_t = std::chrono::time_point<std::chrono::system_clock>;

inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
        std::chrono::time_point<std::chrono::system_clock> b) {
    return  std::chrono::duration<double>(b - a).count();
}

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
#define SET_TS(__tss__) ((__tss__).push_back(std::chrono::system_clock::now()))

