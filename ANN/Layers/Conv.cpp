#include <ANN/Layers/Conv.h>

Conv::~Conv(){
    Layer::~Layer();

    if (mask) delete[] mask;
}

void Conv::init_cl_mem(Function opt, int bsize){
    Dense::init_cl_mem(opt, bsize);

    mask_clmem = alloc_buffer(
        cl->context, "mask_clmem", nweights * sizeof(float),
        mask, CL_MEM_COPY_HOST_PTR);
}

void Conv::free_cl_mem() {
    Dense::free_cl_mem();
    
    clReleaseMemObject(mask_clmem);
}

void Conv::connect(Layer* prev){
    Dense::connect(prev);

    // Create mask, values init to 0.0f
    mask = new float[nweights];
    std::fill(mask, mask + nweights, 0.0f);

    // Each unit has it's own kernel
    for (int i = 0; i < nunits; i++){
        // Fill in mask area for each kernel
        for (int x = 0; x < maskx; x++){
            for (int y = 0; y < masky; y++){
                // Row and col on image 
                int irow = i / masks(prevx, maskx, stridex);
                int icol = i % masks(prevx, maskx, stridex);

                // i * prev_nunits               : index of the unit
                // (irow + y) * prevx + icol + x : index of pixel in image 
                int index = i * prev_nunits + (irow + y) * prevx + icol + x;
                
                mask[index] = 1.0f;
            }
        }
    }

    // Apply mask to weights
    for (int i = 0; i < nweights; i++)
        weights[i] *= mask[i];
}

void Conv::calc_weight_grad(Function reg, float lambda){
    Dense::calc_weight_grad(reg, lambda);

    size_t gloabl_size = nweights;
    call_kernel(cl, vec_vec_mult, 
        1, NULL, &gloabl_size, NULL, 0, NULL, NULL,// Args
        weights_grad_clmem,
        mask_clmem,
        weights_grad_clmem);
}
