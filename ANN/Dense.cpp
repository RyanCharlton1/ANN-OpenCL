#include <ANN/Dense.h>
#include <ANN/CLData.h>

#include <random>
#include <cstring>
#include <string>

void Dense::init_cl_mem(cl_context context, int bsize){
    this->bsize = bsize;
    values_clmem = alloc_buffer(
        context, "values", bsize * nunits * sizeof(float), values,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    
    weights_clmem = alloc_buffer(
        context, "weights", nweights * sizeof(float), weights, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

    bias_clmem = alloc_buffer(
        context, "bias", nunits * sizeof(float), bias,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    
    pre_act_values_clmem = alloc_buffer(
        context, "pre act values", bsize * nunits * sizeof(float));
}

void Dense::free_cl_mem(){
    clReleaseMemObject(values_clmem);
    clReleaseMemObject(weights_clmem);
    clReleaseMemObject(bias_clmem);
    clReleaseMemObject(pre_act_values_clmem);
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
    switch (act)
    {
    case ReLU:
        call_kernel(
            cl, ReLU, 
            1, NULL, &work_size, NULL, 0, NULL, NULL,
            // Args
            &pre_act_values_clmem,
            &values_clmem);
        break;

    case leaky_ReLU:
        call_kernel(
            cl, leaky_ReLU, 
            1, NULL, &work_size, NULL, 0, NULL, NULL,
            // Args
            &pre_act_values_clmem,
            &values_clmem);
        break;
    
    default:
        std::cout << "Activation function not found" << std::endl;
        break;
    }

    clFinish(cl->command_queue);
}

//void Dense::update(){
//    //// Clear values
//    //for (int i = 0; i < nunits; i++) pre_act[i] = 0.0f;
////
//    //// Multiply weight matrix by prev Layer's values
//    //float* prev_values = prev->get_values();
////
//    //for (int i = 0; i < nunits; i++)
//    //    for (int j = 0; j < prev_nunits; j++)
//    //        pre_act[i] += prev_values[j] * weights[i * prev_nunits + j];
////
//    //// Add bias
//    //if (has_bias) 
//    //    for (int i = 0; i < nunits; i++) pre_act[i] += bias[i];
////
//    //// Apply leaky ReLU activation function
//    //for (int i = 0; i < nunits; i++) 
//    //    values[i] = pre_act[i] > 0 ? pre_act[i] : 0.1f * pre_act[i]; 
//}

void Dense::connect(Layer* prev){
    this->prev  = prev; 
    prev_nunits = prev->get_nunits();
    nweights    = prev_nunits * nunits;

    weights      = new float[nweights];
    weights_grad = new float[nweights];

    // Init weights with glorot
    float half_range = sqrtf(6.0 / (prev_nunits + nunits));
    float range      = 2.0f * half_range;

    for (int i = 0; i < nweights; i++)
        weights[i] = -half_range + (range * (float)rand() / (float)RAND_MAX);

    if (has_bias)
        for (int i = 0; i < nunits; i++)
            bias[i] = -half_range + (range * (float)rand() / (float)RAND_MAX);
}

void Dense::optimise(float learn_rate){
    // Update weights 
    for (int i = 0; i < nunits; i++)
        for (int j = 0; j < prev_nunits; j++){
            int index       = i * prev_nunits + j;
            weights[index] -= learn_rate * weights_grad[index];
            }

    // Update bias
    if (has_bias)
        for (int i = 0; i < nunits; i++)
            bias[i] -= learn_rate * bias_grad[i];
}

void Dense::calc_act_grad(){
    // Using leaky ReLU 
    for (int i = 0; i < nunits; i++)
        act_grad[i] = pre_act[i] > 0.0f ? 1.0 : 0.1f;
}

void Dense::accumulate_weight_grad(){
    float* prev_values = prev->get_values();

    for (int i = 0; i < nunits; i++)
        for (int j = 0; j < prev_nunits; j++)
            weights_grad[i * prev_nunits + j] += values_grad[i] * prev_values[j];
    
    for (int i = 0; i < nunits; i++)
        bias_grad[i] += values_grad[i];
}

void Dense::calc_loss_grad(){
    float* prev_loss_grad = prev->get_loss_grad();

    // Weights matrix is prev_nunits by nunits:
    // [0, 1, ... prev_nunits]
    // [1, ...               ]
    // [...                  ]
    // [nunits, ...          ]
    // Because we're calculating backwards going from y to z:
    // Read collumns instead of rows
    for (int i = 0; i < prev_nunits; i++){
        float n = 0.0f;

        for (int j = 0; j < nunits; j++){
            n += prev_loss_grad[j] * weights[j * prev_nunits + i];
        }

        prev_loss_grad[i] = n;
    }
}

void Dense::calc_value_grad(){
    for (int i = 0; i < nunits; i++)
        values_grad[i] = loss_grad[i] * act_grad[i];
}

void Dense::clear_accumulators(){
    for (int i = 0; i < nweights; i++)
        weights_grad[i] = 0.0f;

    for (int i = 0; i < nunits; i++)
        bias_grad[i] = 0.0f;
}

void Dense::average_accumulators(int n){
    for (int i = 0; i < nweights; i++)
        weights_grad[i] /= (float)n;

    for (int i = 0; i < nunits; i++)
        bias_grad[i] /= (float)n;
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