#pragma once

#include <ANN/CLData.h>

#include <string>
#include <CL/cl.h>

class Layer{
protected:
    float* values  = nullptr;
    float* weights = nullptr;
    float* bias    = nullptr;  
    bool   has_bias;

    int nunits      = 0;
    int prev_nunits = 0;
    int nweights    = 0;
    int features    = 0;

    Layer* prev;

    Function act;
    
    bool norm;

    CLdata* cl;

    int bsize;

    cl_mem values_clmem;
    // Weights is a table of connections to previous layers nodes
    // node | x_1 x_2 prev nodes...
    // y_1  | w_1 w_2 ...
    // The transpose gives weights of connections from a given node in
    // the last layer to this layer. Useful for prev layers node gradients.
    cl_mem weights_clmem;       
    cl_mem bias_clmem;
    cl_mem pre_act_values_clmem;

    cl_mem input_grad_clmem;    // Input gradient, pre activation
    cl_mem output_grad_clmem;   // Output gradient, post activation
    cl_mem weights_grad_clmem;  
    cl_mem act_grad_clmem;      // Activation/value differntial
    cl_mem bias_grad_clmem;

    cl_mem softmax_sum_clmem;

    bool   use_adam = false;
    cl_mem adam_weight_avg_clmem;
    cl_mem adam_weight_square_avg_clmem;
    cl_mem adam_bias_avg_clmem;
    cl_mem adam_bias_square_avg_clmem;

    float* beta_values  = nullptr;
    float* gamma_values = nullptr;

    cl_mem norm_avg_clmem;
    cl_mem norm_var_clmem;
    cl_mem norm_beta_clmem;
    cl_mem norm_gamma_clmem;
    cl_mem norm_beta_grad_clmem;
    cl_mem norm_gamma_grad_clmem;
    cl_mem pre_norm_values_clmem;
    cl_mem pre_affine_values_clmem;

    cl_mem adam_beta_avg_clmem;
    cl_mem adam_beta_square_clmem;
    cl_mem adam_gamma_avg_clmem;
    cl_mem adam_gamma_square_clmem;

public:
    Layer(int nunits, Function act, bool norm, bool bias);
    ~Layer();

    int    get_nunits()   { return nunits; }
    float* get_values()   { return values; }
    float* get_weights()  { return weights; }
    bool   get_norm()     { return norm; }

    void set_cl(CLdata* cl) { this->cl = cl; }

    cl_mem get_values_clmem()  { return values_clmem; }
    cl_mem get_weights_clmem() { return weights_clmem; }

    cl_mem get_input_grad_clmem()     { return input_grad_clmem; }
    cl_mem get_pre_act_values_clmem() { return pre_act_values_clmem; }
    cl_mem get_output_grad_clmem()    { return output_grad_clmem; }
    cl_mem get_act_grad_clmem()       { return act_grad_clmem; }

    void cl_to_host_values();
    void cl_to_host_weights();
    void cl_to_host_norm();

    // Create cl mem afor values and weights and store weights
    virtual void init_cl_mem(Function opt, int bsize=1);
    virtual void free_cl_mem();

    void init_norm_cl_mem(Function opt);

    // Adam averages are calculated iteratively, so they need to be zeroed
    // each epoch
    void zero_adam_avgs();
    void zero_adam_norm();

    // Init weights with random values using Xavier and Glorot 
    // initialisation
    void init_weights(int nweights);

    // Calc new values by feed forward
    void update();
    // Past values mutliplied by weights
    virtual void calc_pre_act_values() {};
    // Add bias to pre act values
    void add_bias();
    // Use Layer::norm to normalise values pre activation
    virtual void normalise() {};
    // Apply activation funtion pre act values
    void apply_act();
    // Connect Layer to prev during Network compilation
    virtual void connect(Layer* prev);
    // Optimse parameters using opt optimser
    void optimise(Function opt, float learn_rate, int instance);
    // Accumulate weight grads dL/dw by multilpying dL/dy and dy/dw(z)
    // and bias dL/db as dL/dy * dy/db(1)
    virtual void calc_weight_grad(Function reg, float lambda) {};
    // Calculate prev Layer's gradient at output dL/dA by multiplying 
    // dL/dy and dy/dA(w^T)
    virtual void calc_prev_output_grad() {};
    // Calculate value_grad by multiplying dL/dA and dA/dz
    virtual void calc_input_grad() {};
    // Calculate the activation function gradient at the pre_act_values
    void calc_act_grad();
    // Combine normalisation gradient with act_grad_clmem 
    virtual void calc_norm_grad() {};
    
    virtual std::string to_string() { return ""; };
};
