#pragma once

#include <string>

class Layer{
protected:
    float* values  = nullptr;
    float* pre_act = nullptr;
    float* weights = nullptr;
    float* bias    = nullptr;  
    bool   has_bias;

    float* values_grad  = nullptr;
    float* act_grad     = nullptr;
    float* loss_grad    = nullptr;
    float* weights_grad = nullptr;
    float* bias_grad    = nullptr;

    int nunits;
    int prev_nunits;
    int nweights;

    Layer *prev;

public:
    Layer(int nunits, bool bias);
    ~Layer();

    int    get_nunits()  { return nunits; }
    float* get_values()  { return values; }
    float* get_weights() { return weights; }

    float* get_values_grad() { return values_grad; }
    float* get_act_grad()    { return act_grad; }
    float* get_loss_grad()   { return loss_grad; }

    // Calc new values by feed forward
    virtual void update() {};
    // Connect Layer to prev during Network compilation
    virtual void connect(Layer* prev) {};
    virtual void optimise(float learn_rate) {};
    virtual void calc_act_grad() {};
    // Accumulate weight grads dL/dw by multilpying dL/dy and dy/dw(z)
    // And bias dL/db as dL/dy * dy/db(1)
    virtual void accumulate_weight_grad() {};
    // Calculate prev Layer's loss_grad dL/dA by multiplying dL/dy and dy/dA(w^T)
    virtual void calc_loss_grad() {};
    // Calculate value_grad by multiplying dL/dA and dA/dz
    virtual void calc_value_grad() {};
    // Clears bias and weight grad as they accumulate for batch training
    virtual void clear_accumulators() {};
    // Divide weight/bias grad by n
    virtual void average_accumulators(int n) {};
    virtual std::string to_string(){};
};
