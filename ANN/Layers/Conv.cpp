#include <ANN/Layers/Conv.h>

void Conv::init_cl_mem(Function opt, int bsize){
    Dense::init_cl_mem(opt, bsize);

    float zero = 0.0f;

    padded_values_grad_size  = filterw + (outx - 1) * stridex + filterw - 1;
    padded_values_grad_size *= filterh + (outy - 1) * stridey + filterh - 1;
    padded_values_grad_size *= features;
    //padded_values_grad_size *= bsize;

    dilated_values_grad_size  = 1 + (outx - 1) * stridex;
    dilated_values_grad_size += 1 + (outy - 1) * stridey;
    dilated_values_grad_size *= features;

    padded_values_grad_clmem = alloc_buffer(
        cl->context, "padded_values_grad_clmem", 
        bsize * padded_values_grad_size * sizeof(float));
    
    clEnqueueFillBuffer(
        cl->command_queue, padded_values_grad_clmem, &zero, sizeof(float),
        0, bsize * padded_values_grad_size * sizeof(float), 0, NULL, NULL);

    dilated_values_grad_clmem = alloc_buffer(
        cl->context, "dilated_values_grad_clmem", 
        bsize * dilated_values_grad_size * sizeof(float));

    clEnqueueFillBuffer(
        cl->command_queue, dilated_values_grad_clmem, &zero, sizeof(float),
        0, bsize * dilated_values_grad_size * sizeof(float), 0, NULL, NULL);

    clFinish(cl->command_queue);
}

void Conv::free_cl_mem(){
    Dense::free_cl_mem();

    clReleaseMemObject(padded_values_grad_clmem);
    clReleaseMemObject(dilated_values_grad_clmem);
}

void Conv::connect(Layer* prev){
    this->prev  = prev; 
    prev_nunits = prev->get_nunits();
    nweights    = features * filterh * filterw * prevc;
    weights     = new float[nweights];

    init_weights();
}

void Conv::calc_pre_act_values(){
    size_t work_size[3] = { bsize * outy, outx, features };

    call_kernel(
        cl, convolution,
        3, NULL, work_size, NULL, 0, NULL, NULL,
        // Args
        prevw,
        prevh,
        outy,
        filterw,
        filterh,
        prevc,
        stridex,
        stridey,
        weights_clmem,
        prev->get_values_clmem(),
        pre_act_values_clmem);

    clFinish(cl->command_queue);
}

void Conv::calc_loss_grad(){
    size_t pad_work_size[3]    = { bsize * outy, outx, features };
    size_t deconv_work_size[3] = { bsize * prevh, prevw, prevc};
    cl_event padded;

    call_kernel(
        cl, pad_and_dilate,
        3, NULL, pad_work_size, NULL, 0, NULL, &padded,
        // Args
        filterw,
        filterh,
        stridex,
        stridey,
        padded_values_grad_size,
        bsize,
        values_grad_clmem,
        padded_values_grad_clmem);

    call_kernel(
        cl, deconvolution,
        3, NULL, deconv_work_size, NULL, 1, &padded, NULL,
        // Args
        filterw + (outx - 1) * stridex + filterw - 1,
        filterh + (outy - 1) * stridey + filterh - 1,
        filterw,
        filterh,
        features,
        weights_clmem,
        padded_values_grad_clmem,
        prev->get_loss_grad_clmem());

    clFinish(cl->command_queue);
}

void Conv::calc_weight_grad(Function reg, float lambda){
    size_t pad_work_size[3]  = { bsize * outy, outx, features };
    size_t grad_work_size[3] = { filterh, filterw, prevc * features };
    cl_event padded;

    call_kernel(
        cl, pad_and_dilate,
        3, NULL, pad_work_size, NULL, 0, NULL, &padded,
        // Args
        1, // Don't pad
        1, // Don't pad
        stridex,
        stridey,
        dilated_values_grad_size,
        bsize,
        values_grad_clmem,
        dilated_values_grad_clmem);

    call_kernel(
        cl, convolution_weight_grads,
        3, NULL, grad_work_size, NULL, 1, &padded, NULL,
        // Args
        prev_nunits,
        prevw,
        dilated_values_grad_size,
        1 + (outx - 1) * stridex,
        1 + (outy - 1) * stridey,
        prevc,
        bsize,
        prev->get_values_clmem(),
        dilated_values_grad_clmem,
        weights_grad_clmem);

    // TODO: reg

    // TODO: bias, not used with batch norm

    clFinish(cl->command_queue);
}

// void Conv::init_cl_mem(Function opt, int bsize=1){
//     Layer::init_cl_mem(opt, bsize);
    
//     weights_grads_pre_batch_clmem = alloc_buffer(
//         cl->context, "weights_grads_pre_batch_clmem", 
//         bsize * nweights * sizeof(float));

//     loss_grads_pre_sum_clmem = alloc_buffer(
//         cl->context, "loss_grads_pre_sum_clmem", nunits * outx * outy);
// }

// void Conv::free_cl_mem(){
//     Layer::free_cl_mem();

//     clReleaseMemObject(weights_grads_pre_batch_clmem);
//     clReleaseMemObject(loss_grads_pre_sum_clmem);
// }

// void Conv::calc_pre_act_values(){
//     size_t work_size[3] = { outy * bsize,
//                             outx,
//                             features };

//     call_kernel(cl, convolution,
//         3, NULL, work_size, NULL, 0, NULL, NULL,
//         // Args
//         maskx,
//         masky,
//         prevz,
//         stridex,
//         stridey,
//         prevx,
//         nweights,
//         bsize,
//         weights_clmem,
//         prev->get_values_clmem(),
//         pre_act_values_clmem,
//         weights_grads_pre_batch_clmem);

//     clFinish(cl->command_queue);
// }

// void Conv::connect(Layer* prev){
//     this->prev  = prev; 
//     prev_nunits = prev->get_nunits();
//     nweights    = nunits * masky * maskx * prevz;
//     weights     = new float[nweights];

//     init_weights();
// }

// void Conv::calc_weight_grad(Function reg, float lambda){
//     size_t work_size[1] = { nweights };

//     call

//     call_kernel(cl, avg,
//         1, NULL, work_size, NULL, 0, NULL, NULL,
//         // Args
//         bsize,
//         weights_grads_pre_batch_clmem,
//         weights_grad_clmem);
// }

// void Conv::calc_loss_grad(){
    
// }

std::string Conv::to_string(){
    char buffer[16];
    std::string s;

    int mask_size = filterh * filterw * prevc;

    for (int i = 0; i < features; i++){
        s += "[";

        for (int j = 0; j < mask_size; j++){
            sprintf(buffer, "% .5f ", weights[i * mask_size + j]);
            s += buffer;
        }

        s.pop_back();
        s += "]\n";
    }

    if (has_bias){
        s += "bias: [";
        for (int i = 0; i < nunits; i++){
            sprintf(buffer, "% .5f ", bias[i]);
            s += buffer;
        }

        s.pop_back();
        s += "]\n";
    }


    // Print beta and gamma 
    if (norm != none){
        s += "gamma: [";

        for (int i = 0; i < features; i++){
            sprintf(buffer, "% .5f ", gamma_values[i]);
            s += buffer;
        }

        s.pop_back();
        s += "]\n";

        s += "beta:  [";

        for (int i = 0; i < features; i++){
            sprintf(buffer, "% .5f ", beta_values[i]);
            s += buffer;
        }

        s.pop_back();
        s += "]\n";
    }

    s += "\n";
    return s;
}
