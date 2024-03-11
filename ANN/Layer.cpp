#include <ANN/Layer.h>

Layer::Layer(int nunits, Function act, bool bias){
    this->nunits   = nunits;
    this->has_bias = bias;
    this->act      = act;

    values  = new float[nunits];
    pre_act = new float[nunits];

    if (bias) this->bias = new float[nunits];

    values_grad = new float[nunits];
    act_grad    = new float[nunits];
    loss_grad   = new float[nunits];
    //bias_grad   = new float[nunits];
}

Layer::~Layer(){
    if (values)  delete[] values;
    if (pre_act) delete[] pre_act;
    if (weights) delete[] weights;
    if (bias)    delete[] bias;

    if (values_grad)  delete[] values_grad;
    if (act_grad)     delete[] act_grad;
    if (loss_grad)    delete[] loss_grad;
    if (weights_grad) delete[] weights_grad;
    //if (bias_grad)    delete[] bias_grad;
}

void Layer::update(){
    calc_pre_act_values();
    if (has_bias)
        add_bias();
    // Apply activation function and store in values_clmem
    apply_act();
}