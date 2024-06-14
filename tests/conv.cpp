#include <ANN/Network.h>
#include <ANN/Layers/Conv.h>

int main(){
    Network n(9);
    n.add_layer(new Conv(3, 3, 1, 1, 2, 2, ReLU));
    n.compile(1e-3, MSE, GrdDsc);

    std::cout << n.to_string();

    float values[] = { 1, 2, 3, 
                       4, 5, 6, 
                       7, 8, 9,
                       8, 7, 6,
                       5, 4, 3,
                       2, 1, 0 };
    n.fit(values, 9, nullptr, 4, 1, 2);

    //n.calc(values, 9);
    std::cout << n.trace();

    return 0;
}