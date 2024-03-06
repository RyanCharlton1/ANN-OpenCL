#include <ANN/Dense.h>

#include <random>
#include <cstring>
#include <string>

void Dense::update(){
    // Clear values
    for (int i = 0; i < nunits; i++) values[i] = 0.0f;

    // Multiply weight matrix by prev Layer's values
    float* prev_values = prev->get_values();

    for (int i = 0; i < nunits; i++)
        for (int j = 0; j < prev_nunits; j++)
            pre_act[i] += prev_values[j] * weights[i * prev_nunits + j];

    // Add bias
    if (has_bias) 
        for (int i = 0; i < nunits; i++) pre_act[i] += bias[i] * bias_c;

    // Apply ReLU activation function
    for (int i = 0; i < nunits; i++) 
        values[i] = pre_act[i] > 0 ? pre_act[i] : 0.0f; 
}

void Dense::connect(Layer* prev){
    this->prev  = prev; 
    prev_nunits = prev->get_nunits();
    nweights    = prev_nunits * nunits;

    weights = new float[nweights];

    if (has_bias) bias = new float[nunits];

    // Init weights with glorot
    float half_range = sqrtf(6.0 / (prev_nunits + nunits));
    float range      = 2.0f * half_range;

    for (int i = 0; i < nweights; i++)
        weights[i] = -half_range + (range * (float)rand() / (float)RAND_MAX);

    if (has_bias)
        for (int i = 0; i < nunits; i++)
            bias[i] = -half_range + (range * (float)rand() / (float)RAND_MAX);
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
        sprintf(buffer, "] x % .5f\n", bias_c);
        s += buffer;
    }

    s += "\n";
    return s;
}