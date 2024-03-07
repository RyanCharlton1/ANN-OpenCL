#pragma once 

#include <ANN/Layer.h>

class Dense : public Layer{

public:
    Dense(int nunits, bool bias=true) : Layer(nunits, bias) {}

    void update() override;
    // Connect to prev Layer and init memory for Dense topology
    void connect(Layer* prev) override;
    void optimise(float learn_rate) override;
    void calc_act_grad() override;
    // Calculate weight grad dL/dw by multilpying dL/dy * dy/dw(z)
    void accumulate_weight_grad() override;
    // Calculate prev Layer's loss_grad dL/dA by multiplying dL/dy and dy/dA(w^T)
    void calc_loss_grad() override;
    // Calculate value_grad by multiplying dL/dA and dA/dz
    void calc_value_grad() override;
    // Set all weights/bias grad to 0.0f
    void clear_accumulators() override;
    // Divide weight/bias grad by n
    void average_accumulators(int n) override;

    // Print each Layer's weights
    std::string to_string() override;
};