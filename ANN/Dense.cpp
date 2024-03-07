#include <ANN/Dense.h>

#include <random>
#include <cstring>
#include <string>

void Dense::update(){
    // Clear values
    for (int i = 0; i < nunits; i++) pre_act[i] = 0.0f;

    // Multiply weight matrix by prev Layer's values
    float* prev_values = prev->get_values();

    for (int i = 0; i < nunits; i++)
        for (int j = 0; j < prev_nunits; j++)
            pre_act[i] += prev_values[j] * weights[i * prev_nunits + j];

    // Add bias
    if (has_bias) 
        for (int i = 0; i < nunits; i++) pre_act[i] += bias[i];

    // Apply leaky ReLU activation function
    for (int i = 0; i < nunits; i++) 
        values[i] = pre_act[i] > 0 ? pre_act[i] : 0.1f * pre_act[i]; 
}

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