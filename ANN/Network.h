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

    // Connect Layers, initing their memory and generating weights/bias
    void compile(float learn_rate);
    // Store data in input Layer
    void set_input(float* data, int dsize);
    // Feed forward data to calculate output, stored in the final Layer
    void calc(float* data, int dsize);
    // Clear each Layer's accumulators for weight/bias grads
    void clear_accumulators();
    void fit_batch(float* data, int dsize, float* exp, int esize, int bsize);
    void fit(float* data, int dsize, float* exp, int esize,
             int batches=1, int bsize=1, int epochs=1);

    std::string to_string();
    std::string trace();
};