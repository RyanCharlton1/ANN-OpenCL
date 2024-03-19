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
    
    Function opt;
    Function loss;

    cl_mem expected_clmem;
    cl_mem loss_clmem;
    cl_mem loss_grad_clmem;

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

    void init_clmem(int bsize);
    void free_clmem();

    void host_to_cl_expected(float* exp, int esize);

    void cl_to_host_weights();
    void cl_to_host_values();
    void cl_to_host();

    // Connect Layers, initing their memory and generating weights/bias
    void compile(float learn_rate, Function loss, Function opt);
    // Store data in input Layer
    void set_input(float* data, int dsize);
    // Feed forward data to calculate output, stored in the final Layer
    void calc_cl(float* data, int dsize);
    // Init cl mem, feed forward, return pointer to output values
    float* calc(float* data, int dsize); 
    // Calculate dL/dA at the final layer
    float calc_loss(int dsize);
    // Calculate dL/dy by multiplying dL/dA * dA/dy
    void calc_output_value_grad(int dsize);

    float fit_batch_cl(float* data, int dsize, float* exp, int esize, int bsize);
    void fit(float* data, int dsize, float* exp, int esize,
             int batches=1, int bsize=1, int epochs=1);

    void evaluate(float* test, int tsize, float* exp, int esize, int count);

    std::string to_string();
    std::string trace();
};