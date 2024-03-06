#pragma once 

#include <ANN/Layer.h>

class Dense : public Layer{

public:
    Dense(int nunits, bool bias=true) : Layer(nunits, bias) {}

    void update() override;
    // Connect to prev Layer and init memory for Dense topology
    void connect(Layer* prev) override;
    // Print each Layer's weights
    std::string to_string() override;
};