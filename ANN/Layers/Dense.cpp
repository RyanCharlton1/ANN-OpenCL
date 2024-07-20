#include <ANN/Layers/Dense.h>
#include <ANN/CLData.h>

#include <random>
#include <cstring>
#include <string>

// Output are the result of multiplying the weight matrix by the input vlaues
void Dense::calc_pre_act_values(){
    size_t work_size = bsize * nunits;
    
    call_kernel(cl, mat_vec_mult, 
        1, NULL, &work_size, NULL, 0, NULL, NULL,
        // Args
        prev_nunits,
        nunits,
        weights_clmem,
        prev->get_values_clmem(),
        pre_act_values_clmem);

    clFinish(cl->command_queue);
}

// Use 1D batch normalisation before outputing values. 1D normalises each
// feature across the batch
void Dense::normalise(){
    size_t work_size     = bsize * nunits;
    size_t features_size = features;

    float zero = 0.0f;

    cl_int status;
    // Hold pre norm values for computing norm gradient
    clEnqueueCopyBuffer(cl->command_queue, 
        pre_act_values_clmem, pre_norm_values_clmem, 0, 0, 
        bsize * nunits * sizeof(float), 0, NULL, NULL);

    // Set initial average and variance calcs to zero
    status = clEnqueueFillBuffer(cl->command_queue, norm_avg_clmem,
        &zero, sizeof(float), 0, features * sizeof(float), 0, NULL, NULL);
    cl_print_err("norm_avg_clmem", status);

    status = clEnqueueFillBuffer(cl->command_queue, norm_var_clmem,
        &zero, sizeof(float), 0, features * sizeof(float), 0, NULL, NULL);
    cl_print_err("norm_var_clmem", status);

    clFinish(cl->command_queue);

    cl_event avg_complete, var_complete, norm_complete;

    // Calculate each feature's average across batches
    call_kernel(cl, avg, 
        1, NULL, &features_size, NULL, 0, NULL, &avg_complete,
        // Args
        bsize,
        pre_act_values_clmem,
        norm_avg_clmem);
    
    // Calculate each feature's variance across batches
    call_kernel(cl, var,
        1, NULL, &features_size, NULL, 1, &avg_complete, &var_complete,
        // Args
        bsize,
        norm_avg_clmem,
        pre_act_values_clmem,
        norm_var_clmem);
    
    // Normalise the values using each features variance and average
    call_kernel(cl, norm1d,
        1, NULL, &work_size, &features_size, 1, &var_complete, &norm_complete,
        // Args
        pre_act_values_clmem,
        norm_avg_clmem,
        norm_var_clmem);
    
    // Hold pre affine values for affine parameter derivative calc
    clEnqueueCopyBuffer(cl->command_queue, 
        pre_act_values_clmem, pre_affine_values_clmem, 0, 0, 
        bsize * nunits * sizeof(float), 1, &norm_complete, NULL);

    // Apply affine transformation to normalised values
    call_kernel(cl, affine, 
        1, NULL, &work_size, &features_size, 1, &norm_complete, NULL,
        // Args
        pre_act_values_clmem,
        norm_gamma_clmem,
        norm_beta_clmem);

    clFinish(cl->command_queue);
}

// Connect to previous Layer and initialise weights
void Dense::connect(Layer* prev){
    Layer::connect(prev);

    init_weights(prev_nunits * nunits);
}
 
// Weight gradients calculated by multiplying the output gradients by the
// input values that each weight connects. These are average over the batch.
// Bias is the special case where the input value is always 1.
void Dense::calc_weight_grad(Function reg, float lambda){
    cl_mem prev_values_clmem = prev->get_values_clmem();

    size_t global_size = nweights;
    size_t local_size  = prev_nunits;

    cl_event weights_done;

    call_kernel(cl, weight_grad,
        1, NULL, &global_size, NULL, 0, NULL, &weights_done,
        // Args
        bsize,
        nunits,
        prev_nunits,
        input_grad_clmem,
        prev_values_clmem,
        weights_grad_clmem);

    // TODO: fix regularisation
    switch (reg){
    case l2_reg:
        call_kernel(cl, l2_reg, 
            1, NULL, &global_size, NULL, 1, &weights_done, NULL,
            // Args
            lambda,
            weights_grad_clmem,
            weights_clmem);

        break;
    }

    if (has_bias){
        global_size = nunits;

        call_kernel(cl, bias_grad,
            1, NULL, &global_size, NULL, 0, NULL, NULL,
            // Args
            bsize,
            input_grad_clmem,
            bias_grad_clmem);
    }
}

// The gradient at a pervious Layer's node is the sum of the gradients 
// of the nodes it connects to multiplied by the weights connecting them.
// Calculate this by multiplying the input gradients by the weights transposed.
void Dense::calc_prev_output_grad(){
    size_t global_size = bsize * prev_nunits;

    call_kernel(cl, mat_vec_mult_trans,
        1, NULL, &global_size, NULL, 0, NULL, NULL,
        // Args
        nunits,
        prev_nunits,
        weights_clmem,
        input_grad_clmem,
        prev->get_output_grad_clmem());    

    clFinish(cl->command_queue);
}

// Calculate input gradient by appying the activation function gradient 
// to the output gradient. If using normalisation, this is not the final
// result.
void Dense::calc_input_grad(){
    size_t global_size = bsize * nunits;

    call_kernel(cl, vec_vec_mult,
        1, NULL, &global_size, NULL, 0, NULL, NULL,
        // Args
        output_grad_clmem,
        act_grad_clmem,
        input_grad_clmem);

    clFinish(cl->command_queue);
}

// Calculate the affine transformation gradients and gradients for the 
// pre normalisation values.
void Dense::calc_norm_grad(){
    size_t work_size  = bsize * nunits;
    size_t local_size = features;

    // Gamma gradient dAf/dg = N => dL/dg = dL/dAf * dAf/dg
    call_kernel(cl, gamma_grad,
        1, NULL, &local_size, NULL, 0, NULL, NULL,
        // Args
        bsize,
        pre_affine_values_clmem,
        input_grad_clmem,
        norm_gamma_grad_clmem);

    // Beta gradient dA/db = 1 => dL/db = dL/dAf * dAf/db
    call_kernel(cl, bias_grad,
        1, NULL, &local_size, NULL, 0, NULL, NULL,
        // Args
        bsize,
        input_grad_clmem,
        norm_beta_grad_clmem);

    // Derivation for normalisation derivative is in README
    call_kernel(cl, norm1d_der,
        1, NULL, &work_size, &local_size, 0, NULL, NULL,
        // Args
        pre_norm_values_clmem,
        norm_avg_clmem,
        norm_var_clmem,
        norm_gamma_clmem,
        input_grad_clmem);

    clFinish(cl->command_queue);
}

// Return string of weight matrix
std::string Dense::to_string(){
    char buffer[16];
    std::string s;

    // Print weights as matrices 
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
    
    // Print bias a vector length ways to not take up screen
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