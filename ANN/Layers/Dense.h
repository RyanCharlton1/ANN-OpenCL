#pragma once 

#include <ANN/Layers/Layer.h>

class Dense : public Layer{

public:
    Dense(int nunits, Function act, Function norm=none, bool bias=true) 
    : Layer(nunits, act, norm, bias) {}
    
    void zero_adam_norm() override;
    // Past values mutliplied by weights
    void calc_pre_act_values() override;
    // Batch normalise values before activation function
    void normalise() override;
    // Connect to prev Layer and init memory for Dense topology
    void connect(Layer* prev) override;
    // Calculate weight grad dL/dw by multilpying dL/dy * dy/dw(z)
    void calc_weight_grad(Function reg, float lambda) override;
    // Calculate prev Layer's loss_grad dL/dA by multiplying dL/dy and dy/dA(w^T)
    void calc_loss_grad() override;
    // Calculate value_grad by multiplying dL/dA and dA/dz
    void calc_value_grad() override;
    // Calculate normalisation grad and combine in act_grad_clmem 
    // by multiplying dA/dAf * dAf/dN * dN/dz. Also calculate norm param
    // grads
    void calc_norm_grad() override;
    // Print each Layer's weights
    std::string to_string() override;
};