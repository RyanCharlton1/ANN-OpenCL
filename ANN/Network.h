#pragma once

#include <ANN/Layer.h>

#include <vector>
#include <string>

class Network{
    int    ninput;
    Layer* input;
    float  learn_rate;

    std::vector<Layer*> layers;

public:
    Network(int ninput);
    ~Network();

    Layer* get_output_layer() { return layers[layers.size() - 1]; }
    float* get_output()       { return get_output_layer()->get_values(); }

    void add_layer(Layer* layer) { layers.push_back(layer); }

    void compile(float learn_rate);
    void set_input(float* data, int dsize);
    void calc(float* data, int dsize);

    std::string to_string();
    std::string trace();
};