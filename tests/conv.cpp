#include <ANN/Network.h>
#include <ANN/Layers/Conv.h>

#define BATCHES   1
#define BATCHSIZE 2
#define EPOCHS    1

#define INPUTSIZE 3 * 3 * 2

int main(){
    Network n(INPUTSIZE);
    n.add_layer(new Conv(3, 3, 2, 2, 2, 3, 1, 1, leaky_ReLU, true));
    n.add_layer(new Conv(2, 2, 3, 2, 2, 1, 1, 1, leaky_ReLU, true));
    n.add_layer(new Dense(1, leaky_ReLU));
    n.compile(1e-2, MSE, GrdDsc);

    std::cout << n.to_string() << std::endl;

    // float values  [INPUTSIZE * BATCHSIZE * BATCHES];
    // float expected[BATCHSIZE * BATCHES];
    // std::fill(values,   values + INPUTSIZE * BATCHSIZE * BATCHES, 0.0f);
    // //std::fill(expected, expected + 9 * BATCHSIZE * BATCHES, 0.0f);

    // for (int i = 0; i < BATCHSIZE * BATCHES; i++){
    //     int n_ = rand() % INPUTSIZE;

    //     values[i * INPUTSIZE + n_] = 1.0f;
    //     expected[i] = n_;
    // }

    float values[] = { 0, 1,    2, 3,   4, 5, 
                       6, 7,    8, 9,   8, 7,
                       6, 5,    4, 3,   2, 1,
                       
                       2, 3,    4, 5,   6, 7,
                       8, 9,    8, 7,   6, 5,
                       4, 3,    2, 1,   0, 1 };
       
    float expected[] = { 2, 3 };

    n.fit(values, INPUTSIZE, expected, 1, BATCHES, BATCHSIZE, EPOCHS);
//
    //n.calc(values, INPUTSIZE);
    //std::cout << n.trace();

    return 0;
}