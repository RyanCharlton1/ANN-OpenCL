#include <ANN/Network.h>
#include <ANN/Layers/Conv.h>

#define BATCHES   5
#define BATCHSIZE 10
#define EPOCHS    5

int main(){
    Network n(18);
    n.add_layer(new Conv(3, 3, 2, 1, 1, 2, 2, 2, leaky_ReLU));
    n.compile(1e-3, MSE, GrdDsc);

    std::cout << n.to_string();

    float values  [9 * BATCHSIZE * BATCHES];
    float expected[9 * BATCHSIZE * BATCHES];

    for (int i = 0; i < BATCHSIZE * BATCHES; i++){
        int n = rand() % 9;

        values[i * 9 + n] = 1.0f;
        expected[i * 9 + n] = 1.0f;
    }

    n.fit(values, 9, expected, 9, BATCHES, BATCHSIZE, EPOCHS);

    //n.calc(values, 9);
    std::cout << n.trace();

    return 0;
}