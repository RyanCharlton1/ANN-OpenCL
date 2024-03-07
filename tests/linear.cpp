#include <ANN/Network.h>
#include <ANN/Dense.h>

#include <iostream>

int main(){
    Network n(1);
    n.add_layer(new Dense(1, true));
    //n.add_layer(new Dense(2));
    n.compile(1e-4f);

    std::cout << n.to_string() << std::endl;
    std::cout << n.trace() << std::endl;

    float in[100];
    float out[100];

    for (int i = 0; i < 100; i++){
        in[i]  = (float)i;
        out[i] = 2 * in[i] + 1.0f; 
    }

    n.fit(in, 1, out, 1, 20, 5, 5);

    std::cout << n.to_string() << std::endl;
    std::cout << n.trace() << std::endl;

}