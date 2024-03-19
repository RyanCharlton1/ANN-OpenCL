#include <ANN/Dense.h>
#include <ANN/CLData.h>

#include <random>
#include <cstring>
#include <string>

void Dense::init_cl_mem(cl_context context, int bsize){
    this->bsize = bsize;
    values_clmem = alloc_buffer(
        context, "values", bsize * nunits * sizeof(float));
    
    weights_clmem = alloc_buffer(
        context, "weights", nweights * sizeof(float), weights, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

    if (has_bias)
    bias_clmem = alloc_buffer(
        context, "bias", nunits * sizeof(float), bias,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    
    pre_act_values_clmem = alloc_buffer(
        context, "pre act values", bsize * nunits * sizeof(float));

    values_grad_clmem = alloc_buffer(
        context, "values_grad_clmem", bsize * nunits * sizeof(float));

    loss_grad_clmem = alloc_buffer(
        context, "loss_grad_clmem", bsize * nunits * sizeof(float));

    weights_grad_clmem = alloc_buffer(
        context, "weights_grad_clmem", nweights * sizeof(float));

    act_grad_clmem = alloc_buffer(
        context, "act_diff_clmem", bsize * nunits * sizeof(float));

    if (has_bias)
    bias_grad_clmem = alloc_buffer(
        context, "bias_grad_clmem", nunits * sizeof(float));

    if (act == softmax)
    softmax_sum_clmem = alloc_buffer(
        context, "softmax_sum_clmem", bsize * sizeof(float));

}

void Dense::free_cl_mem(){
    clReleaseMemObject(values_clmem);
    clReleaseMemObject(weights_clmem);
    clReleaseMemObject(pre_act_values_clmem);

    clReleaseMemObject(values_grad_clmem);
    clReleaseMemObject(loss_grad_clmem);
    clReleaseMemObject(weights_grad_clmem);
    clReleaseMemObject(act_grad_clmem);
    
    if (has_bias){
        clReleaseMemObject(bias_clmem);
        clReleaseMemObject(bias_grad_clmem);
    }

    if (act == softmax)
        clReleaseMemObject(softmax_sum_clmem);
}

void Dense::cl_to_host_values() {
    cl_int status = clEnqueueReadBuffer(
        cl->command_queue, values_clmem, CL_FALSE, 0, 
        nunits * sizeof(float), values, 0, NULL, NULL);

    cl_print_err("Dense cl_to_host_values", status);
    clFinish(cl->command_queue);
}

void Dense::cl_to_host_weights() {
    cl_int status = clEnqueueReadBuffer(
        cl->command_queue, weights_clmem, CL_FALSE, 0, 
        nweights * sizeof(float), weights, 0, NULL, NULL);
    cl_print_err("Dense cl_to_host_weights", status);
    clFinish(cl->command_queue);
}

void Dense::calc_pre_act_values(){
    size_t work_size       = bsize * nunits;
    size_t local_work_size = nunits;
    
    call_kernel(
        cl, mat_vec_mult, 
        1, NULL, &work_size, &local_work_size, 0, NULL, NULL,
        // Args
        prev_nunits,
        &weights_clmem,
        prev->get_values_clmem(),
        &pre_act_values_clmem);

    clFinish(cl->command_queue);
}

void Dense::add_bias(){
    size_t work_size       = bsize * nunits;
    size_t local_work_size = nunits;
    
    call_kernel(
        cl, vec_vec_add_inplace,
        1, NULL, &work_size, &local_work_size, 0, NULL, NULL,
        // Args
        &pre_act_values_clmem,
        &bias_clmem);
    
    clFinish(cl->command_queue);
}

void Dense::apply_act(){
    size_t work_size = bsize * nunits; 
    size_t bsize_s   = bsize;
    size_t nunits_s  = nunits;

    switch (act){
    case softmax:
        call_kernel(cl, softmax_sum,
            1, NULL, &bsize_s, NULL, 0, NULL, NULL,
            // Args
            nunits, 
            &softmax_sum_clmem,
            &pre_act_values_clmem);

        call_kernel(cl, softmax,
            1, NULL, &work_size, &nunits_s, 0, NULL, NULL,
            // Args
            &softmax_sum_clmem,
            &pre_act_values_clmem,
            &values_clmem);
        
        break;
    
    case ReLU:
    case leaky_ReLU:
        call_kernel(
            cl, act,
            1, NULL, &work_size, NULL, 0, NULL, NULL,
            // Args,
            &pre_act_values_clmem,
            &values_clmem);

        break;
    }

    clFinish(cl->command_queue);
}

void Dense::connect(Layer* prev){
    this->prev  = prev; 
    prev_nunits = prev->get_nunits();
    nweights    = prev_nunits * nunits;

    weights      = new float[nweights];

    // Init weights with glorot
    float half_range;
    float range;
    switch(act){
    case ReLU:
    case leaky_ReLU:
        // Xavier initlisation
        half_range = sqrtf(6.0 / (prev_nunits + nunits));
        range      = 2.0f * half_range;
        break;
    
    case softmax:
        // Glorot initialisation
        half_range = sqrt(3.0 / (prev_nunits + nunits));
        range      = 2.0f * half_range;
        break;
    }

    for (int i = 0; i < nweights; i++)
        weights[i] = -half_range + (range * (float)rand() / (float)RAND_MAX);

    if (has_bias)
        for (int i = 0; i < nunits; i++)
            bias[i] = -half_range + (range * (float)rand() / (float)RAND_MAX);
}

void Dense::optimise(Function optimiser, float learn_rate){
    size_t nweights_sizet = nweights;
    size_t nbias_sizet    = nunits;

    switch (optimiser){
        case GrdDsc:
            call_kernel(cl, GrdDsc,
                1, NULL, &nweights_sizet, NULL, 0, NULL, NULL,
                // Args
                learn_rate,
                &weights_clmem,
                &weights_grad_clmem);


            if (has_bias)
                call_kernel(cl, GrdDsc,
                    1, NULL, &nbias_sizet, NULL, 0, NULL, NULL,
                    // Args
                    learn_rate,
                    &bias_clmem,
                    &bias_grad_clmem);
            
            break;
    }
}

void Dense::calc_weight_grad(){
    cl_mem* prev_values_clmem = prev->get_values_clmem();

    size_t global_size = nweights;
    size_t local_size  = prev_nunits;
    call_kernel(cl, weight_grad,
        1, NULL, &global_size, &local_size, 0, NULL, NULL,
        // Args
        bsize,
        &values_grad_clmem,
        prev_values_clmem,
        &weights_grad_clmem);

    if (has_bias){
        global_size = nunits;
        call_kernel(cl, bias_grad,
            1, NULL, &global_size, NULL, 0, NULL, NULL,
            // Args
            bsize,
            nunits,
            &values_grad_clmem,
            &bias_grad_clmem);
    }
}

void Dense::calc_loss_grad(){
    size_t local_size  = prev->get_nunits();
    size_t global_size = bsize * local_size;

    call_kernel(cl, mat_vec_mult_trans,
        1, NULL, &global_size, &local_size, 0, NULL, NULL,
        // Args
        nunits,
        &weights_clmem,
        &values_grad_clmem,
        prev->get_loss_grad_clmem());    

    clFinish(cl->command_queue);
}

void Dense::calc_value_grad(){
    size_t global_size = bsize * nunits;
    call_kernel(cl, vec_vec_mult,
        1, NULL, &global_size, NULL, 0, NULL, NULL,
        // Args
        &loss_grad_clmem,
        &act_grad_clmem,
        &values_grad_clmem);

    clFinish(cl->command_queue);
}

void Dense::calc_act_grad(){
    // Softmax der is combined into cross entropy der
    if (act == softmax)
        return;

    size_t work_size = bsize * nunits; 

    call_kernel(
        cl, derivative(act),
        1, NULL, &work_size, NULL, 0, NULL, NULL,
        // Args, all activaitons take a single input so I can generalise
        &pre_act_values_clmem,
        &act_grad_clmem);

    clFinish(cl->command_queue);
}

// Return string of weight matrix
std::string Dense::to_string(){
    char buffer[16];
    std::string s;

    for (int i = 0; i < nunits; i++){
        s += "[";
        for (int j = 0; j < prev_nunits; j++){
            // Format to have leading space if + and 5 decimal places
            sprintf(buffer, "% .5f ", weights[i * prev_nunits + j]);
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

    s += "\n";
    return s;
}