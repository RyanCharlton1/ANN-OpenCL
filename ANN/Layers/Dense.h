#pragma once 

#include <ANN/Layers/Layer.h>

class Dense : public Layer{

public:
    Dense(int nunits, Function act, bool bias=true) 
    : Layer(nunits, act, bias) {}


    // Create cl mem afor values and weights and store weights
    void init_cl_mem(Function opt, int bsize=1) override;
    void free_cl_mem() override;

    void zero_adam_avgs() override;

    void cl_to_host_values() override;
    void cl_to_host_weights() override;

    //void update() override;
    // Past values mutliplied by weights
    void calc_pre_act_values() override;
    // Add bias to pre act values
    void add_bias() override;
    // Apply activation funtion pre act values
    void apply_act() override;
    // Connect to prev Layer and init memory for Dense topology
    void connect(Layer* prev) override;
    void optimise(Function optimiser, float learn_rate, int instance) override;

    // Calculate weight grad dL/dw by multilpying dL/dy * dy/dw(z)
    void calc_weight_grad() override;

    // Calculate prev Layer's loss_grad dL/dA by multiplying dL/dy and dy/dA(w^T)
    void calc_loss_grad() override;
    // Calculate value_grad by multiplying dL/dA and dA/dz
    void calc_value_grad() override;
    // Calculate the activation function gradient at the pre_act_values
    void calc_act_grad() override;
    // Print each Layer's weights
    std::string to_string() override;
};