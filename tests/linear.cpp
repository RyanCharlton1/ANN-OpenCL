#include <ANN/Network.h>

#include <iostream>

int main(){
    Network n(1);
    n.add_layer(new Dense(2, leaky_ReLU, norm1d));
    n.add_layer(new Dense(1, leaky_ReLU, none, false));
    n.compile(1e-2f, MSE, GrdDsc);

    std::cout << n.to_string() << std::endl;

    float in[100];
    float out[100];

    for (int i = 0; i < 100; i++){
        in[i]  = (float)i;
        out[i] = 2 * in[i] + 1.0f; 
    }

    n.fit(in, 1, out, 1, 3, 5, 100);
    
    //n.evaluate(in, 1, out, 1, 100);

    std::cout << n.to_string() << std::endl;
    std::cout << n.trace() << std::endl;
}