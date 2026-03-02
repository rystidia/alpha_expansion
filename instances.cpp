#include "graph.h"
#include "energy.h"
#include "graph.cpp"
#include "maxflow.cpp"

template class Graph<int, int, int>;
template class Graph<short, int, int>;
template class Graph<float, float, float>;
template class Graph<double, double, double>;

template class Energy<int, int, int>;
template class Energy<short, int, int>;
template class Energy<float, float, float>;
template class Energy<double, double, double>;
