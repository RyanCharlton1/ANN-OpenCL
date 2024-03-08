#pragma once

#include <ANN/Layer.h>
#include <ANN/Dense.h>
#include <ANN/CLData.h>

#include <CL/cl.h>

#include <vector>
#include <string>

class Network{
    int    ninput;
    Dense* input;
    float  learn_rate;

    std::vector<Layer*> layers;
    
public:
    CLdata cl;
    Network(int ninput);
    ~Network();

    void create_kernel(Function f);
    void create_kernels();

    Layer* get_output_layer() { return layers[layers.size() - 1]; }
    float* get_output()       { return get_output_layer()->get_values(); }

    std::vector<Layer*> get_layers() { return layers; }

    void add_layer(Layer* layer) { layers.push_back(layer); }
    
    void cl_to_host_weights();
    void cl_to_host_values();

    // Connect Layers, initing their memory and generating weights/bias
    void compile(float learn_rate);
    // Store data in input Layer
    void set_input(float* data, int dsize);
    // Feed forward data to calculate output, stored in the final Layer
    void calc(float* data, int dsize);
    // Clear each Layer's accumulators for weight/bias grads
    void clear_accumulators();
    void fit_batch(float* data, int dsize, float* exp, int esize, int bsize);
    void fit_batch_cl(float* data, int dsize, float* exp, int esize, int bsize);
    void fit(float* data, int dsize, float* exp, int esize,
             int batches=1, int bsize=1, int epochs=1);

    std::string to_string();
    std::string trace();
};