#include <ANN/Layer.h>

Layer::Layer(int nunits, Function act, bool bias){
    this->nunits   = nunits;
    this->has_bias = bias;
    this->act      = act;

    values = new float[nunits];

    if (bias) this->bias = new float[nunits];
}

Layer::~Layer(){
    if (values)  delete[] values;
    if (weights) delete[] weights;
    if (bias)    delete[] bias;
}

void Layer::update(){
    calc_pre_act_values();
    if (has_bias)
        add_bias();
    // Apply activation function and store in values_clmem
    apply_act();
}