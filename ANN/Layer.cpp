#include <ANN/Layer.h>

Layer::Layer(int nunits, bool bias){
    this->nunits   = nunits;
    this->has_bias = bias;

    values = new float[nunits];
}

Layer::~Layer(){
    if (values)  delete[] values;
    if (weights) delete[] weights;
    if (bias)    delete[] bias;
}

