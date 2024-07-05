#pragma once 

#include <ANN/Layers/Dense.h>

#define masks(prev_size, mask_size, stride) \
    ((prev_size - mask_size) / stride + 1) 

#define nunits(prevx, prevy, maskx, masky, stridex, stridey, features) \
    masks(prevx, maskx, stridex) * masks(prevy, masky, stridey) * features

class Conv : public Dense{
    int prevx,   prevy, prevz;
    int maskx,   masky;
    int stridex, stridey;
    int features;

    float* mask = nullptr;
    cl_mem mask_clmem;

public: 
    // This Layer's nunits is determined by how many kernels can applied 
    // to the last Layer using the provided stride multiplied by the 
    // number of output features. 
    // prevz is number of channels e.g. 3 for RGB image
    Conv(int prevx,   int prevy,   int prevz,
         int maskx,   int masky,   int features,
         int stridex, int stridey,
         Function act, Function norm=none, bool bias=true)
    : Dense(nunits(prevx, prevy, maskx, masky, stridex, stridey, features),
            act, norm, bias){
        
        this->prevx = prevx; this->prevy = prevy; this->prevz = prevz;
        this->maskx = maskx; this->masky = masky;

        this->stridex  = stridex; this->stridey = stridey;
        this->features = features;
    }
    
    ~Conv();
    
    void init_cl_mem(Function opt, int bsize=1) override;
    void free_cl_mem() override;

    // Connect to prev Layer and init memory for Dense topology
    void connect(Layer* prev) override;
    // Calculate weight grad dL/dw by multilpying dL/dy * dy/dw(z)
    void calc_weight_grad(Function reg, float lambda) override;
};
