#pragma once 

#include <ANN/Layers/Dense.h>

#define masks(prev_size, mask_size, stride) \
    ((prev_size - mask_size) / stride + 1) 

#define nunits(prevx, prevy, maskx, masky, stridex, stridey) \
    masks(prevx, maskx, stridex) * masks(prevy, masky, stridey)

// Simplest implementation of Conv is a Dense layer with 0's as weights 
// for connections that aren't in a given output node's kernel and keeping 
// their weight gradients 0
class Conv : public Dense{
    int maskx,   masky;
    int stridex, stridey;
    int prevx,   prevy;

    float* mask = nullptr;
    cl_mem mask_clmem;

public:
    // The size of this Layer is dependant on the size of the the last and
    // the kernel used for convolution. size is the side length of the kernel
    Conv(int prevx, int prevy, int stridex, int stridey, 
         int maskx, int masky, Function act, bool bias=true) 
    : Dense(nunits(prevx, prevy, maskx, masky, stridex, stridey), act, none, bias) 
        { this->maskx   = maskx;   this->masky   = masky;
          this->stridex = stridex; this->stridey = stridey;
          this->prevx   = prevx;   this->prevy   = prevy; }

    ~Conv();
    
    void init_cl_mem(Function opt, int bsize=1) override;
    void free_cl_mem() override;

    // Connect to prev Layer and init memory for Dense topology
    void connect(Layer* prev) override;
    // Calculate weight grad dL/dw by multilpying dL/dy * dy/dw(z)
    void calc_weight_grad(Function reg, float lambda) override;
    // Print each Layer's weights
    //std::string to_string() override;
};