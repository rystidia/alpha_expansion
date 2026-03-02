#include <iostream>
#include <cassert>
#include "energy.h"

int main() {
    // E(x, y, z) = x - 2*y + 3*(1-z) - 4*x*y + 5*|y-z|
    // x,y,z in {0,1}
    
    typedef Energy<int, int, int> EnergyT;
    EnergyT::Var varx, vary, varz;
    EnergyT *e = new EnergyT(3, 5);

    varx = e->add_variable();
    vary = e->add_variable();
    varz = e->add_variable();

    e->add_term1(varx, 0, 1); // x
    e->add_term1(vary, 0, -2); // -2y
    e->add_term1(varz, 3, 0); // 3*(1-z)

    e->add_term2(varx, vary, 0, 0, 0, -4); // -4*x*y
    
    // 5*|y-z|
    e->add_term2(vary, varz, 0, 5, 5, 0); 

    EnergyT::TotalValue min_energy = e->minimize();
    
    assert(min_energy == -5);
    assert(e->get_var(varx) == 1);
    assert(e->get_var(vary) == 1);
    assert(e->get_var(varz) == 1);

    delete e;
    std::cout << "test_basic_energy passed!" << std::endl;
    return 0;
}
