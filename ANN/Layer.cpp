#include <ANN/Layer.h>

Layer::Layer(int nunits, bool bias){
    this->nunits   = nunits;
    this->has_bias = bias;

    values  = new float[nunits];
    pre_act = new float[nunits];
}

Layer::~Layer(){
    if (values)  delete[] values;
    if (pre_act) delete[] pre_act;
    if (weights) delete[] weights;
    if (bias)    delete[] bias;
}

