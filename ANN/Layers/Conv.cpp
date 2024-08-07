#include <ANN/Layers/Conv.h>

void Conv::init_cl_mem(Function opt, int bsize){
    Dense::init_cl_mem(opt, bsize);

    float zero = 0.0f;

    // Backwards pass is calculated by convolution of reversed filters
    // over output gradients padded by filter dimension -1 and 
    // dilated by stride -1
    padded_values_grad_size  = filterw + (outw - 1) * stridex + filterw - 1;
    padded_values_grad_size *= filterh + (outh - 1) * stridey + filterh - 1;
    padded_values_grad_size *= features;

    // Weight gradients are calculated by convolution of previous Layer's
    // values with output gradients dliated by stride -1
    dilated_values_grad_size  = 1 + (outw - 1) * stridex;
    dilated_values_grad_size *= 1 + (outh - 1) * stridey;
    dilated_values_grad_size *= features;

    // Buffers are padded and dilted with 0s
    // Init to all 0s and then place the values in the correct positions
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

    batch_weight_grads_clmem = alloc_buffer(
        cl->context, "batch_weight_grads_clmem", 
        bsize * nweights * sizeof(float));

    clFinish(cl->command_queue);
}

void Conv::free_cl_mem(){
    Dense::free_cl_mem();

    clReleaseProgram(conv_program);
    clReleaseMemObject(padded_values_grad_clmem);
    clReleaseMemObject(dilated_values_grad_clmem);
    clReleaseMemObject(batch_weight_grads_clmem);

    for (auto& kernel : conv_kernels)
        clReleaseKernel(kernel.second);

    conv_kernels.clear();
}

char PARAMS[] = "-DFILTERH=%d -DFILTERW=%d -DCHANNELS=%d "
    "-DSTRIDEX=%d -DSTRIDEY=%d -DPREVH=%d -DPREVW=%d -DOUTH=%d -DOUTW=%d "
    "-DFEATURES=%d -DBSIZE=%d";

void Conv::load_kernels(){
    cl_int status;
    char options[256];

    std::snprintf(options, 256, PARAMS, 
        filterh, filterw, prevc, stridex, stridey, prevh, prevw, outh, 
        outw, features, bsize);

    conv_program = create_program(cl->context, cl->device_list, 
        "../ANN/Layers/Conv.cl", options);

    for (int i = convolution; i < none; i++)
        create_kernel(conv_program, &conv_kernels, (Function)i);
}

void Conv::connect(Layer* prev){
    Layer::connect(prev);
    // Filters share channel dimension with previous Layer
    init_weights(features * filterh * filterw * prevc);
}

// Perform convolution using opencl kernel 
void Conv::calc_pre_act_values(){
    // outw and outh are the amount of masks that can be applied to the
    // input image forming the output image's dimensions
    int work_size[3] = { bsize * outh, outw, features };

    call_kernel(
        cl->command_queue, conv_kernels, convolution,
        3, NULL, work_size, NULL, 0, NULL, NULL,
        // Args
        weights_clmem,
        prev->get_values_clmem(),
        pre_act_values_clmem);

    clFinish(cl->command_queue);
}

// Calculate input gradients by applying reversed filters to padded and 
// dilated output gradients
// https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
void Conv::calc_prev_output_grad(){
    int pad_work_size[3]    = { bsize * outh, outw, features };
    int rev_work_size[3]    = { filterh, filterw, features };
    int deconv_work_size[3] = { bsize * prevh, prevw, prevc};
    cl_event padded;

    call_kernel(
        cl->command_queue, conv_kernels, pad_and_dilate,
        3, NULL, pad_work_size, NULL, 0, NULL, &padded,
        // Args
        input_grad_clmem,
        padded_values_grad_clmem);

    call_kernel(
        cl->command_queue, conv_kernels, deconvolution,
        3, NULL, deconv_work_size, NULL, 1, &padded, NULL,
        // Args
        weights_clmem,
        padded_values_grad_clmem,
        prev->get_output_grad_clmem());

    clFinish(cl->command_queue);
}

// Calculate weight gradients by applying previous Layer's values to
// padded and dilated output gradients
// https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
void Conv::calc_weight_grad(Function reg, float lambda){
    int pad_work_size[3]  = { bsize * outh, outw, features };
    int grad_work_size[3] = { bsize * filterh, filterw, prevc * features };
    cl_event padded, batch;

    call_kernel(
        cl->command_queue, conv_kernels, dilate,
        3, NULL, pad_work_size, NULL, 0, NULL, &padded,
        // Args
        input_grad_clmem,
        dilated_values_grad_clmem);

    // call_kernel(
    //     cl->command_queue, conv_kernels, convolution_weight_grads,
    //     3, NULL, grad_work_size, NULL, 1, &padded, NULL,
    //     // Args
    //     prev->get_values_clmem(),
    //     dilated_values_grad_clmem,
    //     weights_grad_clmem);

    call_kernel(
        cl->command_queue, conv_kernels, convolution_weight_grads,
        3, NULL, grad_work_size, NULL, 1, &padded, &batch,
        // Args
        prev->get_values_clmem(),
        dilated_values_grad_clmem,
        batch_weight_grads_clmem);

    call_kernel(
        cl->command_queue, conv_kernels, average_weight_grads,
        1, NULL, &nweights, NULL, 1, &batch, NULL,
        // Args
        batch_weight_grads_clmem,
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
