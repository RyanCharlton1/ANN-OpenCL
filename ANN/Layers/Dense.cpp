#include <ANN/Layers/Dense.h>
#include <ANN/CLData.h>

#include <random>
#include <cstring>
#include <string>

// Output are the result of multiplying the weight matrix by the input vlaues
void Dense::calc_pre_act_values(){
    int work_size[1] = { bsize * nunits };
    
    call_kernel(cl->command_queue, cl->kernels, mat_vec_mult, 
        1, NULL, work_size, NULL, 0, NULL, NULL,
        // Args
        prev_nunits,
        nunits,
        weights_clmem,
        prev->get_values_clmem(),
        pre_act_values_clmem);

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

    int global_size[1] = { nweights };
    int local_size[1]  = { prev_nunits };

    cl_event weights_done;

    call_kernel(cl->command_queue, cl->kernels, weight_grad,
        1, NULL, global_size, NULL, 0, NULL, &weights_done,
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
        call_kernel(cl->command_queue, cl->kernels, l2_reg, 
            1, NULL, global_size, NULL, 1, &weights_done, NULL,
            // Args
            lambda,
            weights_grad_clmem,
            weights_clmem);

        break;
    }
}

// The gradient at a pervious Layer's node is the sum of the gradients 
// of the nodes it connects to multiplied by the weights connecting them.
// Calculate this by multiplying the input gradients by the weights transposed.
void Dense::calc_prev_output_grad(){
    int global_size[1] = { bsize * prev_nunits };

    call_kernel(cl->command_queue, cl->kernels, mat_vec_mult_trans,
        1, NULL, global_size, NULL, 0, NULL, NULL,
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
    int global_size[1] = { bsize * nunits };

    call_kernel(cl->command_queue, cl->kernels, vec_vec_mult,
        1, NULL, global_size, NULL, 0, NULL, NULL,
        // Args
        output_grad_clmem,
        act_grad_clmem,
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