#include <ANN/Layers/Layer.h>

Layer::Layer(int nunits, Function act, bool bias){
    this->nunits   = nunits;
    this->has_bias = bias;
    this->act      = act;

    values = new float[nunits];

    if (bias) this->bias = new float[nunits];
}

void Layer::init_cl_mem(Function opt, int bsize){
    this->bsize = bsize;
    values_clmem = alloc_buffer(
        cl->context, "values", bsize * nunits * sizeof(float));
    
    weights_clmem = alloc_buffer(
        cl->context, "weights", nweights * sizeof(float), weights, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

    if (has_bias)
    bias_clmem = alloc_buffer(
        cl->context, "bias", nunits * sizeof(float), bias,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    
    pre_act_values_clmem = alloc_buffer(
        cl->context, "pre act values", bsize * nunits * sizeof(float));

    values_grad_clmem = alloc_buffer(
        cl->context, "values_grad_clmem", bsize * nunits * sizeof(float));

    loss_grad_clmem = alloc_buffer(
        cl->context, "loss_grad_clmem", bsize * nunits * sizeof(float));

    weights_grad_clmem = alloc_buffer(
        cl->context, "weights_grad_clmem", nweights * sizeof(float));

    act_grad_clmem = alloc_buffer(
        cl->context, "act_diff_clmem", bsize * nunits * sizeof(float));

    if (has_bias)
    bias_grad_clmem = alloc_buffer(
        cl->context, "bias_grad_clmem", nunits * sizeof(float));

    if (act == softmax)
    softmax_sum_clmem = alloc_buffer(
        cl->context, "softmax_sum_clmem", bsize * sizeof(float));

    if (opt == adam){
    use_adam = true;

    adam_weight_avg_clmem = alloc_buffer(
        cl->context, "adam_avg_clmem", nweights * sizeof(float));

    adam_weight_square_avg_clmem = alloc_buffer(
        cl->context, "adam_square_clmem", nweights * sizeof(float)); 

    if (has_bias){
    adam_bias_avg_clmem = alloc_buffer(
        cl->context, "adam_bias_avg_clmem", nunits * sizeof(float));

    adam_bias_square_avg_clmem = alloc_buffer(
        cl->context, "adam_bias_square_avg_clmem", nunits * sizeof(float));
    }
    }
    
}

void Layer::free_cl_mem(){
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
    
    if (use_adam){
        clReleaseMemObject(adam_weight_avg_clmem);
        clReleaseMemObject(adam_weight_square_avg_clmem);
        clReleaseMemObject(adam_bias_avg_clmem);
        clReleaseMemObject(adam_bias_square_avg_clmem);
    }
}

Layer::~Layer(){
    if (values)  delete[] values;
    if (weights) delete[] weights;
    if (bias)    delete[] bias;
}

void Layer::update(){
    calc_pre_act_values();
    if (has_bias)
        add_bias();
    // Apply activation function and store in values_clmem
    apply_act();
}

void Layer::cl_to_host_values() {
    cl_int status = clEnqueueReadBuffer(
        cl->command_queue, values_clmem, CL_FALSE, 0, 
        nunits * sizeof(float), values, 0, NULL, NULL);

    cl_print_err("cl_to_host_values", status);
    clFinish(cl->command_queue);
}

void Layer::cl_to_host_weights() {
    cl_int status = clEnqueueReadBuffer(
        cl->command_queue, weights_clmem, CL_FALSE, 0, 
        nweights * sizeof(float), weights, 0, NULL, NULL);
    cl_print_err("cl_to_host_weights", status);
    clFinish(cl->command_queue);
}

void Layer::zero_adam_avgs(){
    float zero = 0.0f;

    clEnqueueFillBuffer(
        cl->command_queue, adam_weight_avg_clmem, &zero, sizeof(float), 
        0, nweights * sizeof(float), 0, NULL, NULL);   

    clEnqueueFillBuffer(
        cl->command_queue, adam_weight_square_avg_clmem, &zero, sizeof(float),
        0, nweights * sizeof(float), 0, NULL, NULL);   

    clEnqueueFillBuffer(
        cl->command_queue, adam_bias_avg_clmem, &zero, sizeof(float), 
        0, nunits * sizeof(float), 0, NULL, NULL);   

    clEnqueueFillBuffer(
        cl->command_queue, adam_bias_square_avg_clmem, &zero, sizeof(float),
        0, nunits * sizeof(float), 0, NULL, NULL);  
}

void Layer::add_bias(){
    size_t work_size       = bsize * nunits;
    size_t local_work_size = nunits;
    
    call_kernel(
        cl, vec_vec_add_inplace,
        1, NULL, &work_size, &local_work_size, 0, NULL, NULL,
        // Args
        pre_act_values_clmem,
        bias_clmem);
    
    clFinish(cl->command_queue);
}

void Layer::apply_act(){
    size_t work_size = bsize * nunits; 
    size_t bsize_s   = bsize;
    size_t nunits_s  = nunits;

    cl_event sum_done;

    switch (act){
    case softmax:
        call_kernel(cl, softmax_sum,
            1, NULL, &bsize_s, NULL, 0, NULL, &sum_done,
            // Args
            nunits, 
            softmax_sum_clmem,
            pre_act_values_clmem);

        call_kernel(cl, softmax,
            1, NULL, &work_size, &nunits_s, 1, &sum_done, NULL,
            // Args
            softmax_sum_clmem,
            pre_act_values_clmem,
            values_clmem);
        
        break;
    
    case ReLU:
    case leaky_ReLU:
        call_kernel(
            cl, act,
            1, NULL, &work_size, NULL, 0, NULL, NULL,
            // Args,
            pre_act_values_clmem,
            values_clmem);

        break;
    }

    clFinish(cl->command_queue);
}

void Layer::optimise(Function optimiser, float learn_rate, int instance){
    size_t nweights_sizet = nweights;
    size_t nbias_sizet    = nunits;

    switch (optimiser){
        case GrdDsc:
            call_kernel(cl, GrdDsc,
                1, NULL, &nweights_sizet, NULL, 0, NULL, NULL,
                // Args
                learn_rate,
                weights_clmem,
                weights_grad_clmem);


            if (has_bias)
                call_kernel(cl, GrdDsc,
                    1, NULL, &nbias_sizet, NULL, 0, NULL, NULL,
                    // Args
                    learn_rate,
                    bias_clmem,
                    bias_grad_clmem);
            
            break; 
        
        case adam:
            call_kernel(cl, adam, 
                1, NULL, &nweights_sizet, NULL, 0, NULL, NULL,
                // Args
                learn_rate,
                weights_clmem,
                weights_grad_clmem,
                adam_weight_avg_clmem,
                adam_weight_square_avg_clmem,
                instance);

            if (has_bias){
                call_kernel(cl, adam,
                    1, NULL, &nbias_sizet, NULL, 0, NULL, NULL,
                    // Args
                    learn_rate,
                    bias_clmem,
                    bias_grad_clmem,
                    adam_bias_avg_clmem,
                    adam_bias_square_avg_clmem,
                    instance);
            }
            break;
    }
}

void Layer::calc_act_grad(){
    // Softmax der is combined into cross entropy der
    if (act == softmax)
        return;

    size_t work_size = bsize * nunits; 

    call_kernel(
        cl, derivative(act),
        1, NULL, &work_size, NULL, 0, NULL, NULL,
        // Args, all activaitons take a single input so I can generalise
        pre_act_values_clmem,
        act_grad_clmem);

    clFinish(cl->command_queue);
}