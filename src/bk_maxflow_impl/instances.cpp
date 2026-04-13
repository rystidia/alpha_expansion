#include "graph.cpp"
#include "maxflow.cpp"
#include "bk_maxflow_impl/energy.h"
#include <cstdint>

template class Graph<int, int, int>;
template class Energy<int, int, int>;

template class Graph<float, float, float>;
template class Energy<float, float, float>;

template class Graph<double, double, double>;
template class Energy<double, double, double>;

template class Graph<int64_t, int64_t, int64_t>;
template class Energy<int64_t, int64_t, int64_t>;
