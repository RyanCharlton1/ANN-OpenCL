#include <ANN/Network.h>
#include <ANN/Dense.h>

#include <iostream>

int main(){
    Network n(1);
    n.add_layer(new Dense(2));
    n.add_layer(new Dense(2));
    n.compile(1e-2f);

    std::cout << n.to_string() << std::endl;

    std::cout << n.trace() << std::endl;
    float f = 1.0f;
    n.calc(&f, 1);
    std::cout << n.trace() << std::endl;
}