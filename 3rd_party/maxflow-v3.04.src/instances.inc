#include "graph_mf.h"

#ifdef _MSC_VER
#pragma warning(disable: 4661)
#endif

// Instantiations: <captype, tcaptype, flowtype>
// IMPORTANT: 
//    flowtype should be 'larger' than tcaptype 
//    tcaptype should be 'larger' than captype

template class Graph_mf<int,int,int>;
template class Graph_mf<short,int,int>;
template class Graph_mf<float,float,float>;
template class Graph_mf<double,double,double>;

