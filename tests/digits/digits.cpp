#include <ANN/Network.h>
#include <ANN/Layers/Conv.h>
#include <Data/digit_load.hpp>

#define IMAGESIZE  28*28
#define IMAGECOUNT 10000
#define TRAINSIZE  9000
#define TESTSIZE   IMAGECOUNT - TRAINSIZE
#define LABELSIZE  10

#define BATCHSIZE 128
#define EPOCHS    3

int main(){ 
    Network n(IMAGESIZE);
    n.add_layer(new Conv(28, 28, 1, 7, 7, 4, 1, 1, ReLU, true));
    n.add_layer(new Conv(22, 22, 4, 7, 7, 8, 1, 1, ReLU, true));
    n.add_layer(new Conv(16, 16, 8, 7, 7, 16, 1, 1, ReLU, true));
    //n.add_layer(new Dense(512, ReLU));
    //n.add_layer(new Dense(512, ReLU));
    n.add_layer(new Dense(10, softmax));
    n.compile(1e-3, cross_entropy, adam); //l2_reg, 0.01f);

    // std::cout << n.to_string() << '\n';

    DigitData d = load_digits();

    float* label_onehot = new float[d.label_count * 10];
    for (int i = 0; i < d.label_count; i++)
        label_onehot[i * 10 + d.labels[i]] = 1.0f;
    
    // 10k instances, save 1k for test set 
    n.fit(d.data, IMAGESIZE, label_onehot, LABELSIZE, 
        TRAINSIZE / BATCHSIZE, BATCHSIZE, EPOCHS);

    // std::cout << n.to_string() << '\n';

    float* test_data   = &d.data[TRAINSIZE * IMAGESIZE];
    float* test_labels = &label_onehot[TRAINSIZE * LABELSIZE];

    n.evaluate(test_data, IMAGESIZE, test_labels,  LABELSIZE, TESTSIZE);
    n.evaluate(d.data,    IMAGESIZE, label_onehot, LABELSIZE, TRAINSIZE);

    delete[] label_onehot;
}