#include <ANN/Network.h>
#include <Data/digit_load.hpp>

#define IMAGESIZE  28*28
#define IMAGECOUNT 10000
#define TRAINSIZE  9000
#define TESTSIZE   IMAGECOUNT - TRAINSIZE
#define LABELSIZE  10

#define BATCHSIZE 100
#define EPOCHS    5 


// Model result GrDsc:
// 90/90   [====================>] loss: 0.17 batch time: 31 ms epoch time: 2.75s
// 90/90   [====================>] loss: 0.07 batch time: 34 ms epoch time: 3.26s
// 90/90   [====================>] loss: 0.05 batch time: 31 ms epoch time: 2.82s
// 90/90   [====================>] loss: 0.04 batch time: 37 ms epoch time: 3.26s
// 90/90   [====================>] loss: 0.03 batch time: 31 ms epoch time: 2.92s
// 1000/1000[====================>] acc: 0.92

// Model result adam with 1e-3 learning rate
// 90/90   [====================>] loss: 0.07 avg. batch time: 27 ms epoch time: 2.49s
// 90/90   [====================>] loss: 0.05 avg. batch time: 31 ms epoch time: 2.86s
// 90/90   [====================>] loss: 0.04 avg. batch time: 28 ms epoch time: 2.53s
// 1000/1000[====================>] acc: 0.93


int main(){ 
    Network n(IMAGESIZE);
    n.add_layer(new Dense(512, ReLU));
    n.add_layer(new Dense(512, ReLU));
    n.add_layer(new Dense(10, softmax));
    n.compile(1e-3, cross_entropy, adam);

    DigitData d = load_digits();

    float* label_onehot = new float[d.label_count * 10];
    for (int i = 0; i < d.label_count; i++)
        label_onehot[i * 10 + d.labels[i]] = 1.0f;
    
    // 10k instances, save 1k for test set 
    n.fit(d.data, IMAGESIZE, label_onehot, LABELSIZE, 
        TRAINSIZE / BATCHSIZE, BATCHSIZE, 3);

    float* test_data   = &d.data[TRAINSIZE * IMAGESIZE];
    float* test_labels = &label_onehot[TRAINSIZE * LABELSIZE];

    n.evaluate(test_data, IMAGESIZE, test_labels, LABELSIZE, TESTSIZE);

    delete[] label_onehot;
}