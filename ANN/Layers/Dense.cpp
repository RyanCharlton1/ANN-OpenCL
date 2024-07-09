#include <ANN/Layers/Dense.h>
#include <ANN/CLData.h>

#include <random>
#include <cstring>
#include <string>

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

void Dense::normalise(){
    size_t work_size  = bsize * nunits;
    size_t features = get_features();

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
        1, NULL, &features, NULL, 0, NULL, &avg_complete,
        // Args
        (norm == norm1d ? bsize : nunits * bsize),
        pre_act_values_clmem,
        norm_avg_clmem);
    
    // Calculate each feature's variance across batches
    call_kernel(cl, var,
        1, NULL, &features, NULL, 1, &avg_complete, &var_complete,
        // Args
        (norm == norm1d ? bsize : nunits * bsize),
        norm_avg_clmem,
        pre_act_values_clmem,
        norm_var_clmem);
    
    // Normalise the values using each features variance and average
    call_kernel(cl, norm1d,
        1, NULL, &work_size, &features, 1, &var_complete, &norm_complete,
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
        1, NULL, &work_size, &features, 1, &norm_complete, NULL,
        // Args
        pre_act_values_clmem,
        norm_gamma_clmem,
        norm_beta_clmem);

    clFinish(cl->command_queue);
}

void Dense::connect(Layer* prev){
    this->prev  = prev; 
    prev_nunits = prev->get_nunits();
    nweights    = prev_nunits * nunits;
    weights     = new float[nweights];

    init_weights();
}
 
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
        values_grad_clmem,
        prev_values_clmem,
        weights_grad_clmem);

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
            values_grad_clmem,
            bias_grad_clmem);
    }
}

void Dense::calc_loss_grad(){
    size_t global_size = bsize * prev_nunits;

    call_kernel(cl, mat_vec_mult_trans,
        1, NULL, &global_size, NULL, 0, NULL, NULL,
        // Args
        nunits,
        prev_nunits,
        weights_clmem,
        values_grad_clmem,
        prev->get_loss_grad_clmem());    

    clFinish(cl->command_queue);
}

void Dense::calc_value_grad(){
    size_t global_size = bsize * nunits;

    call_kernel(cl, vec_vec_mult,
        1, NULL, &global_size, NULL, 0, NULL, NULL,
        // Args
        loss_grad_clmem,
        act_grad_clmem,
        values_grad_clmem);

    clFinish(cl->command_queue);
}

void Dense::calc_norm_grad(){
    size_t work_size  = bsize * nunits;
    size_t local_size = get_features();

    // Gamma gradient dAf/dg = N => dL/dg = dL/dAf * dAf/dg
    call_kernel(cl, gamma_grad,
        1, NULL, &local_size, NULL, 0, NULL, NULL,
        // Args
        (norm == norm1d ? bsize : nunits * bsize),
        pre_affine_values_clmem,
        values_grad_clmem,
        norm_gamma_grad_clmem);

    // Beta gradient dA/db = 1 => dL/db = dL/dAf * dAf/db
    call_kernel(cl, bias_grad,
        1, NULL, &local_size, NULL, 0, NULL, NULL,
        // Args
        bsize,
        values_grad_clmem,
        norm_beta_grad_clmem);

    call_kernel(cl, norm1d_der,
        1, NULL, &work_size, &local_size, 0, NULL, NULL,
        // Args
        pre_norm_values_clmem,
        norm_avg_clmem,
        norm_var_clmem,
        norm_gamma_clmem,
        values_grad_clmem);

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
    if (norm != none){
        s += "gamma: [";

        for (int i = 0; i < get_features(); i++){
            sprintf(buffer, "% .5f ", gamma_values[i]);
            s += buffer;
        }

        s.pop_back();
        s += "]\n";

        s += "beta:  [";

        for (int i = 0; i < get_features(); i++){
            sprintf(buffer, "% .5f ", beta_values[i]);
            s += buffer;
        }

        s.pop_back();
        s += "]\n";
    }

    s += "\n";
    return s;
}