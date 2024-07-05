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
    // for (int i = 0; i < nunits / features; i++){
    //     // Fill in mask area for each kernel
    //     for (int x = 0; x < maskx; x++){
    //         for (int y = 0; y < masky; y++){
    //             // Row and col on image 
    //             int irow = i / masks(prevx, maskx, stridex);
    //             int icol = i % masks(prevx, maskx, stridex);

    //             // i * prevx * prevy : index of the unit
    //             int index = i * prevx * prevy * features;
    //             // (irow + y) * prevx + icol + x : index of pixel in image 
    //             index += (irow + y) * prevx + icol + x;
                
    //             // Set mask for all previous channel's collumns and new 
    //             // faetures rows
    //             for (int z = 0; z < prevz; z++)
    //                 for (int f = 0; f < features; f++)
    //                     mask[f * prev_nunits + index * prevz + z] = 1.0f;
    //         }
    //     }
    // }

    // Each Kernel
    for (int k = 0; k < nunits / features; k++){
        
        // Each row of the weights matrix is a single new unit
        int kernel_offset = k * features * prev_nunits;

        int num_masky = k / masks(prevx, maskx, stridex);
        int num_maskx = k % masks(prevx, maskx, stridex);
    
        // Each element of the mask 
        for (int y = 0; y < masky; y++){

            int yoffset = kernel_offset + (num_masky * stridey + y) * prevx * prevz;

            for (int x = 0; x < maskx; x++){

                int xoffset = yoffset + (num_maskx * stridex + x) * prevz;

                // Number of the mask 0,0 is top left 
                
                // Each channel of the input 
                for (int z = 0; z < prevz; z++){

                    int zoffset = xoffset + z;

                    // Each feature of the output is a new unit
                    for (int f = 0; f < features; f++){

                        int feature_offset = zoffset + f * prev_nunits;

                        mask[feature_offset] = 1.0f;  
                    }
                }
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
    
    clFinish(cl->command_queue);
}
