#include <ANN/Layers/Dense.h>
#include <ANN/CLData.h>

#include <random>
#include <cstring>
#include <string>

void Dense::calc_pre_act_values(){
    size_t work_size       = bsize * nunits;
    size_t local_work_size = nunits;
    
    call_kernel(
        cl, mat_vec_mult, 
        1, NULL, &work_size, &local_work_size, 0, NULL, NULL,
        // Args
        prev_nunits,
        weights_clmem,
        prev->get_values_clmem(),
        pre_act_values_clmem);

    clFinish(cl->command_queue);
}

void Dense::connect(Layer* prev){
    this->prev  = prev; 
    prev_nunits = prev->get_nunits();
    nweights    = prev_nunits * nunits;
    weights     = new float[nweights];

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

void Dense::calc_weight_grad(Function reg, float lambda){
    cl_mem prev_values_clmem = prev->get_values_clmem();

    size_t global_size = nweights;
    size_t local_size  = prev_nunits;

    cl_event weights_done;

    call_kernel(cl, weight_grad,
        1, NULL, &global_size, &local_size, 0, NULL, &weights_done,
        // Args
        bsize,
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
            nunits,
            values_grad_clmem,
            bias_grad_clmem);
    }
}

void Dense::calc_loss_grad(){
    size_t local_size  = prev->get_nunits();
    size_t global_size = bsize * local_size;

    call_kernel(cl, mat_vec_mult_trans,
        1, NULL, &global_size, &local_size, 0, NULL, NULL,
        // Args
        nunits,
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