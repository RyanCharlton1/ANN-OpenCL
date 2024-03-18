#include <CL/cl.h>

#include <ANN/CLData.h>
#include <ANN/Network.h>
#include <ANN/Dense.h>

#include <iostream>
#include <fstream>
#include <sstream>

#define EXAMPLES  10000
#define BATCHSIZE 5

int main(){
    Network n(1);
    n.add_layer(new Dense(2, ReLU, false));
    n.add_layer(new Dense(2, softmax, false));
    n.compile(1e-1, cross_entropy,  GrdDsc);

    srand(0xC0FFEE);

    float in[EXAMPLES];
    float out[2 * EXAMPLES];
    for (int i = 0; i < EXAMPLES; i++){
        float r = (float)rand() / (float)RAND_MAX - 0.5f;

        in[i]      = r * 2.0f;
        out[i*2]   = in[i] <  0.0f ? 1.0f : 0.0f;
        out[i*2+1] = in[i] >= 0.0f ? 1.0f : 0.0f;
    }

    n.fit(in, 1, out, 2, EXAMPLES / BATCHSIZE, BATCHSIZE, 5);

    std::cout << n.to_string() << std::endl;
    
    return 0;
}