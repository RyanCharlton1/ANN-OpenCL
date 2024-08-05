#pragma once 

#include <ANN/Layers/Dense.h>

#define masks(prev_size, mask_size, stride) \
    ((prev_size - mask_size) / stride + 1) 

#define nunits(prevw, prevh, maskw, maskh, stridex, stridey, features) \
    masks(prevw, maskw, stridex) * masks(prevh, maskh, stridey) * features

class Conv : public Dense{
    int prevw,    prevh,   prevc;
    int filterw,  filterh;
    int stridex,  stridey;
    
    int outw, outh;

    cl_program                    conv_program;
    std::map<Function, cl_kernel> conv_kernels;

    int    padded_values_grad_size;
    cl_mem padded_values_grad_clmem;

    int    dilated_values_grad_size;
    cl_mem dilated_values_grad_clmem;

    cl_mem reversed_weights_clmem;

public: 
    // This Layer's nunits is determined by how many kernels can applied 
    // to the last Layer using the provided stride multiplied by the 
    // number of output features. 
    // prevc is number of channels e.g. 3 for RGB image
    Conv(int prevw,    int prevh,       int prevc,
         int filterw,  int filterh,     int features,
         int stridex,  int stridey,
         Function act, bool norm=false, bool bias=true)
    : Dense(nunits(prevw, prevh, filterw, filterh, stridex, stridey, features),
            act, norm, bias){
        
        this->prevw   = prevw;   this->prevh   = prevh; this->prevc = prevc;
        this->filterw = filterw; this->filterh = filterh;

        this->stridex  = stridex; this->stridey = stridey;
        this->features = features;

        outw = masks(prevw, filterw, stridex);
        outh = masks(prevh, filterh, stridey);
    }

    void init_cl_mem(Function opt, int bsize=1) override;
    void free_cl_mem() override;

    // Load convolution kernel with all properties of this layer as
    // preprocessor defines to minimise computation for work items
    void load_kernels() override;

    // Past values mutliplied by masks
    void calc_pre_act_values() override;
    // Connect to prev Layer and init memory for masks
    void connect(Layer* prev) override;
    // Calculate weight grad dL/dw by multilpying dL/dy * dy/dw(z)
    void calc_weight_grad(Function reg, float lambda) override;
    // Applying reversed weights to padded and dilated value gradients
    // will calculate previous Layer's loss grad
    void calc_prev_output_grad() override;
    // Print each masks weights, each mask squashed to a single row
    std::string to_string() override;
};
