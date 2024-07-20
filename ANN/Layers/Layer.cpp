#include <ANN/Layers/Layer.h>

#include <cmath>

Layer::Layer(int nunits, Function act, bool norm, bool bias){
    this->nunits   = nunits;
    this->act      = act;
    this->has_bias = bias;
    this->norm     = norm;

    values = new float[nunits];

    // Bias shifts the distribution, but normalising centers at 0 
    // making it redundant
    if (norm) has_bias   = false;
    if (bias) this->bias = new float[nunits];
}

// Initialise OpenCL memory objects, weights_clmem and bias_clmem
// copy vlaues from weights and bias arrays
void Layer::init_cl_mem(Function opt, int bsize){
    this->bsize = bsize;
    values_clmem = alloc_buffer(
        cl->context, "values_clmem", bsize * nunits * sizeof(float));
    
    weights_clmem = alloc_buffer(
        cl->context, "weights_clmem", nweights * sizeof(float), weights, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

    if (has_bias)
    bias_clmem = alloc_buffer(
        cl->context, "bias_clmem", nunits * sizeof(float), bias,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    
    pre_act_values_clmem = alloc_buffer(
        cl->context, "pre_act_values_clmem", bsize * nunits * sizeof(float));

    input_grad_clmem = alloc_buffer(
        cl->context, "input_grad_clmem", bsize * nunits * sizeof(float));

    output_grad_clmem = alloc_buffer(
        cl->context, "output_grad_clmem", bsize * nunits * sizeof(float));

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
    
    if (norm)
        init_norm_cl_mem(adam);
}

// Initialise OpenCL memory objects for normalisation, norm_beta_clmem and
// norm_gamma_clmem, copy values from beta_values and gamma_values arrays
void Layer::init_norm_cl_mem(Function opt){
    float zero = 0.0f;
    float one  = 1.0f;

    size_t size = features * sizeof(float);

    cl_int status;

    norm_avg_clmem = alloc_buffer(
        cl->context, "norm_avg_clmem", size);

    norm_var_clmem = alloc_buffer(
        cl->context, "norm_var_clmem", size);

    norm_beta_clmem = alloc_buffer(
        cl->context, "norm_beta_clmem", size, beta_values, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

    norm_gamma_clmem = alloc_buffer(
        cl->context, "norm_gamma_clmem", size, gamma_values,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    
    pre_norm_values_clmem = alloc_buffer(
        cl->context, "pre_norm_values_clmem", bsize * nunits * sizeof(float));

    pre_affine_values_clmem = alloc_buffer(
        cl->context, "pre_affine_values_clmem", bsize * nunits * sizeof(float));

    norm_beta_grad_clmem = alloc_buffer(
        cl->context, "norm_beta_grad_clmem", size);

    norm_gamma_grad_clmem = alloc_buffer(
        cl->context, "norm_gamma_grad_clmem", size);

    if (opt == adam){
        
    adam_beta_avg_clmem = alloc_buffer(
        cl->context, "adam_beta_avg_clmem", nunits * sizeof(float));

    adam_beta_square_clmem = alloc_buffer(
        cl->context, "adam_beta_square_clmem", nunits * sizeof(float));

    adam_gamma_avg_clmem = alloc_buffer(
        cl->context, "adam_gamma_avg_clmem", nunits * sizeof(float));

    adam_gamma_square_clmem = alloc_buffer(
        cl->context, "adam_gamma_square_clmem", nunits * sizeof(float));
    }
}

// Free OpenCL memory objects
void Layer::free_cl_mem(){
    clReleaseMemObject(values_clmem);
    clReleaseMemObject(weights_clmem);
    clReleaseMemObject(pre_act_values_clmem);

    clReleaseMemObject(input_grad_clmem);
    clReleaseMemObject(output_grad_clmem);
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

        if (has_bias){
            clReleaseMemObject(adam_bias_avg_clmem);
            clReleaseMemObject(adam_bias_square_avg_clmem);
        }
    }

    if (norm){
        clReleaseMemObject(norm_avg_clmem);
        clReleaseMemObject(norm_var_clmem);
        clReleaseMemObject(norm_beta_clmem);
        clReleaseMemObject(norm_gamma_clmem);
        clReleaseMemObject(pre_norm_values_clmem);
        clReleaseMemObject(norm_beta_grad_clmem);
        clReleaseMemObject(norm_gamma_grad_clmem);
    }
}

// Free memory objects
Layer::~Layer(){
    if (values)       delete[] values;
    if (weights)      delete[] weights;
    if (bias)         delete[] bias;
    if (beta_values)  delete[] beta_values;
    if (gamma_values) delete[] gamma_values;
}

// Update the values using weights and previous Layer's values. Add bias,
// normalise and apply activation function depending on Layer's properties.
void Layer::update(){
    // Calculate input values from previous Layer's output values
    calc_pre_act_values();

    if (has_bias)
        add_bias();

    if (norm)
        normalise();

    // Apply activation function and store in values_clmem
    apply_act();
}

// Retrieve values from gpu memory and store in values array
void Layer::cl_to_host_values() {
    cl_int status = clEnqueueReadBuffer(
        cl->command_queue, values_clmem, CL_FALSE, 0, 
        nunits * sizeof(float), values, 0, NULL, NULL);

    cl_print_err("cl_to_host_values", status);
    clFinish(cl->command_queue);
}

// Retrieve weights from gpu memory and store in weights array
void Layer::cl_to_host_weights() {
    cl_int status = clEnqueueReadBuffer(
        cl->command_queue, weights_clmem, CL_FALSE, 0, 
        nweights * sizeof(float), weights, 0, NULL, NULL);

    cl_print_err("cl_to_host_weights", status);
    clFinish(cl->command_queue);
}

// Retrieve normalisation parameters from gpu memory and store in 
// beta_values and gamma_values arrays
void Layer::cl_to_host_norm(){
    size_t size = features * sizeof(float);

    cl_int status = clEnqueueReadBuffer(
        cl->command_queue, norm_beta_clmem, CL_FALSE, 0, 
        size, beta_values, 0, NULL, NULL);

    cl_print_err("cl_to_host_beta", status);

    status = clEnqueueReadBuffer(
        cl->command_queue, norm_gamma_clmem, CL_FALSE, 0, 
        size, gamma_values, 0, NULL, NULL);

    cl_print_err("cl_to_host_gamma", status);

    clFinish(cl->command_queue);
}

// Adam averages are calculate iteritvely, so they need to be zeroed 
// before each epoch
void Layer::zero_adam_avgs(){
    float zero = 0.0f;

    cl_int status;

    status = clEnqueueFillBuffer(
        cl->command_queue, adam_weight_avg_clmem, &zero, sizeof(float), 
        0, nweights * sizeof(float), 0, NULL, NULL);   
    cl_print_err("adam_weight_avg_clmem", status);

    status = clEnqueueFillBuffer(
        cl->command_queue, adam_weight_square_avg_clmem, &zero, sizeof(float),
        0, nweights * sizeof(float), 0, NULL, NULL);   
    cl_print_err("adam_weight_square_avg_clmem", status);

    if (has_bias){
    status = clEnqueueFillBuffer(
        cl->command_queue, adam_bias_avg_clmem, &zero, sizeof(float), 
        0, nunits * sizeof(float), 0, NULL, NULL);   
    cl_print_err("adam_bias_avg_clmem", status);

    status = clEnqueueFillBuffer(
        cl->command_queue, adam_bias_square_avg_clmem, &zero, sizeof(float),
        0, nunits * sizeof(float), 0, NULL, NULL);  
    cl_print_err("adam_bias_square_avg_clmem", status);
    }

    if (norm)
        zero_adam_norm();
}

// Adam averages are calculate iteritvely, so they need to be zeroed 
// before each epoch
void Layer::zero_adam_norm(){
    float zero = 0.0f;

    size_t size = features * sizeof(float);

    cl_int status;
    status = clEnqueueFillBuffer(
        cl->command_queue, adam_beta_avg_clmem, &zero, sizeof(float), 
        0, size, 0, NULL, NULL);  
    cl_print_err("adam_beta_avg_clmem", status);
    
    status = clEnqueueFillBuffer(
        cl->command_queue, adam_beta_square_clmem, &zero, sizeof(float), 
        0, size, 0, NULL, NULL);  
    cl_print_err("adam_beta_square_clmem", status);

    status = clEnqueueFillBuffer(
        cl->command_queue, adam_gamma_avg_clmem, &zero, sizeof(float), 
        0, size, 0, NULL, NULL);  
    cl_print_err("adam_gamma_avg_clmem", status);
    
    status = clEnqueueFillBuffer(
        cl->command_queue, adam_gamma_square_clmem, &zero, sizeof(float), 
        0, size, 0, NULL, NULL);  
    cl_print_err("adam_gamma_square_clmem", status);
}

// Initalise the weights with random values using Xavier or Glorot 
// depending on the activation function. Init all gamma values to 1 and
// all beta values to 0
void Layer::init_weights(int nweights){
    float half_range;
    float range;
    
    this->nweights = nweights;
    
    weights = new float[nweights];

    switch(act){
    case ReLU:
    case leaky_ReLU:
        // Xavier initlisation
        half_range = sqrtf(6.0 / (prev_nunits + nunits));
        break;
    
    case softmax:
        // Glorot initialisation
        half_range = sqrt(3.0 / (prev_nunits + nunits));
        break;
    }

    range = 2.0f * half_range;

    for (int i = 0; i < nweights; i++)
        weights[i] = -half_range + (range * (float)rand() / (float)RAND_MAX);

    if (has_bias)
        for (int i = 0; i < nunits; i++)
            bias[i] = -half_range + (range * (float)rand() / (float)RAND_MAX);


    if (norm){
        beta_values  = new float[features];
        gamma_values = new float[features];

        std::fill(beta_values, beta_values + features, 0.0f);
        std::fill(gamma_values, gamma_values + features, 1.0f);
    }
}

// Add bias to each unit in the Layer
void Layer::add_bias(){
    size_t work_size = bsize * nunits;
    
    call_kernel(
        cl, vec_vec_add_inplace,
        1, NULL, &work_size, NULL, 0, NULL, NULL,
        // Args
        nunits,
        pre_act_values_clmem,
        bias_clmem);
    
    clFinish(cl->command_queue);
}

// Apply activation function to pre act values to get final output values
void Layer::apply_act(){
    size_t work_size = bsize * nunits; 
    size_t bsize_s   = bsize;

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
            1, NULL, &work_size, NULL, 1, &sum_done, NULL,
            // Args
            nunits,
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

// Link to previous Layer
void Layer::connect(Layer* prev){
    this->prev  = prev; 
    prev_nunits = prev->get_nunits();
}

// Update learnable params using optimiser
void Layer::optimise(Function optimiser, float learn_rate, int instance){
    size_t nweights_sizet = nweights;
    size_t nbias_sizet    = features;

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
        
        if (norm){
            call_kernel(cl, GrdDsc,
                1, NULL, &nbias_sizet, NULL, 0, NULL, NULL,
                // Args
                learn_rate,
                norm_gamma_clmem,
                norm_gamma_grad_clmem);

            call_kernel(cl, GrdDsc,
                1, NULL, &nbias_sizet, NULL, 0, NULL, NULL,
                // Args
                learn_rate,
                norm_beta_clmem,
                norm_beta_grad_clmem);
        }
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

        if (norm){
            call_kernel(cl, adam,
                1, NULL, &nbias_sizet, NULL, 0, NULL, NULL,
                // Args
                learn_rate,
                norm_gamma_clmem,
                norm_gamma_grad_clmem,
                adam_gamma_avg_clmem,
                adam_gamma_square_clmem,
                instance);
                
            call_kernel(cl, adam,
                1, NULL, &nbias_sizet, NULL, 0, NULL, NULL,
                // Args
                learn_rate,
                norm_beta_clmem,
                norm_beta_grad_clmem,
                adam_beta_avg_clmem,
                adam_beta_square_clmem,
                instance);
        }
        break;
    }
    clFinish(cl->command_queue);
}

// Calculate activation gradient by applying activation function's 
// derivative to the pre act values  
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