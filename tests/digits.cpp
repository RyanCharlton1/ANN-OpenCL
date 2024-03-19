#include <ANN/Network.h>
#include <Data/digit_load.hpp>

#define IMAGESIZE 28*28
#define BATCHSIZE 100

int main(){
    Network n(IMAGESIZE);
    n.add_layer(new Dense(512, ReLU));
    n.add_layer(new Dense(512, ReLU));
    n.add_layer(new Dense(10, softmax));
    n.compile(1e-1, cross_entropy, GrdDsc);

    DigitData d = load_digits();

    float* label_onehot = new float[d.label_count * 10];
    for (int i = 0; i < d.label_count; i++)
        label_onehot[i * 10 + d.labels[i]] = 1.0f;
    
    n.fit(d.data, d.data_size, label_onehot, 10, 
        d.data_count / BATCHSIZE, BATCHSIZE, 5);

    float* output = n.calc(d.data, IMAGESIZE);

    print_image(d.data);

    std::cout << "[";
    for (int i = 0; i < 10; i++)
        std::cout << " " << output[i];
    std::cout << "]" << std::endl;

    delete[] label_onehot;
}