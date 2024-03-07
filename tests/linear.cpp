#include <ANN/Network.h>
#include <ANN/Dense.h>

#include <iostream>

int main(){
    Network n(1);
    n.add_layer(new Dense(1, true));
    //n.add_layer(new Dense(2));
    n.compile(1e-1f);

    std::cout << n.to_string() << std::endl;
    std::cout << n.trace() << std::endl;

    float in;
    float out;

    for (int k = 0; k < 5; k++)
    for (int i = 1; i < 6; i++){
        in  = (float) i;
        out = 2*in + 1;
        n.fit(&in, 1, &out, 1);
    }

    std::cout << n.to_string() << std::endl;
    std::cout << n.trace() << std::endl;

}