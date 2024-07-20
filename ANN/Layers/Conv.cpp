#include <ANN/Layers/Conv.h>

void Conv::init_cl_mem(Function opt, int bsize){
    Dense::init_cl_mem(opt, bsize);

    float zero = 0.0f;

    // Backwards pass is calculated by convolution of reversed filters
    // over output gradients padded by filter dimension -1 and 
    // dilated by stride -1
    padded_values_grad_size  = filterw + (outx - 1) * stridex + filterw - 1;
    padded_values_grad_size *= filterh + (outy - 1) * stridey + filterh - 1;
    padded_values_grad_size *= features;

    // Weight gradients are calculated by convolution of previous Layer's
    // values with output gradients dliated by stride -1
    dilated_values_grad_size  = 1 + (outx - 1) * stridex;
    dilated_values_grad_size *= 1 + (outy - 1) * stridey;
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
    Layer::connect(prev);
    // Filters share channel dimension with previous Layer
    init_weights(features * filterh * filterw * prevc);
}

// Perform convolution using opencl kernel 
void Conv::calc_pre_act_values(){
    // outx and outy are the amount of masks that can be applied to the
    // input image forming the output image's dimensions
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

// Calculate input gradients by applying reversed filters to padded and 
// dilated output gradients
// https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
void Conv::calc_prev_output_grad(){
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
        input_grad_clmem,
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
        prev->get_output_grad_clmem());

    clFinish(cl->command_queue);
}

// Calculate weight gradients by applying previous Layer's values to
// padded and dilated output gradients
// https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
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
        input_grad_clmem,
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

std::string Conv::to_string(){
    char buffer[16];
    std::string s;

    int mask_size = filterh * filterw * prevc;

    // Print each mask on a single line for easier reading 
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
    if (norm){
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
