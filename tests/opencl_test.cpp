#include <CL/cl.h>

#include <ANN/CLData.h>
#include <ANN/Network.h>
#include <ANN/Dense.h>

#include <iostream>
#include <fstream>
#include <sstream>

int main(){
    Network n(1);
    n.add_layer(new Dense(16, leaky_ReLU));
    n.add_layer(new Dense(1,  leaky_ReLU));
    n.compile(1e-4f);

    std::cout << n.to_string() << std::endl;

    float in[100];
    float out[100];

    for (int i = 0; i < 100; i++){
        in[i]  = (float)i;
        out[i] = 2 * in[i] + 1.0f; 
    }

    n.fit(&in[1], 1, out, 1, 1, 1, 1);

    std::cout << n.trace() << std::endl;
    return 0;
}