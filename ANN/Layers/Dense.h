#pragma once 

#include <ANN/Layers/Layer.h>

class Dense : public Layer{

public:
    Dense(int nunits, Function act, bool norm=false, bool bias=true) 
    : Layer(nunits, act, norm, bias) { features = nunits; }
    
    // Past values mutliplied by weights
    void calc_pre_act_values() override;
    // Connect to prev Layer and init memory for Dense topology
    void connect(Layer* prev) override;
    // Calculate weight grad dL/dw by multilpying dL/dy * dy/dw(z)
    void calc_weight_grad(Function reg, float lambda) override;
    // Calculate prev Layer's loss_grad dL/dA by multiplying dL/dy and dy/dA(w^T)
    void calc_prev_output_grad() override;
    // Calculate value_grad by multiplying dL/dA and dA/dz
    void calc_input_grad() override;
    // Print each Layer's weights
    std::string to_string() override;
};