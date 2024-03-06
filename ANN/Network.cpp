#include <ANN/Network.h>

#include <iostream>
#include <cstring>
#include <stdlib.h>

Network::Network(int ninput){
    this->ninput = ninput;
    input = new Layer(ninput, false);
}

Network::~Network(){
    delete input;
}

void Network::compile(float learn_rate){
    this->learn_rate = learn_rate;

    // Temporary for debugging purposes
    srand(1212321);

    // Connect each Layer to the last, inits memory and sets vars 
    // refering to prev layer 
    Layer* prev = input;
    for (Layer* layer : layers) {
        layer->connect(prev);
        prev = layer;
    }
}

void Network::set_input(float* data, int dsize){
    std::memcpy(input->get_values(), data, dsize * sizeof(float));
}

void Network::calc(float* data, int dsize){
    if (dsize != ninput){
        std::cout << "Invalid input data size" << std::endl;
        return;
    }

    set_input(data, dsize);

    for (Layer* layer : layers)
        layer->update();
}

// Train the network on provided data using expected results
void Network::fit(float* data, int dsize, float* exp, int esize){
    if (esize != get_output_layer()->get_nunits()){
        std::cout << "Expected size doesn't match network output" << std::endl;
        return;
    }

    // Calculate the networks output for given data
    calc(data, dsize);
    // Calculate loss using MSE
    float* output   = get_output();
    float  loss     = 0.0f;
    float* loss_der = new float[esize]; 

    float error;
    for (int i = 0; i < esize; i++){
        error       = exp[i] - output[i];
        loss       += error * error;
        loss_der[i] = 2.0 * error / (float)esize;
    }

    std::cout << "Loss "

}

std::string Network::to_string(){
    std::string s;
    
    for (Layer* layer : layers)
        s += layer->to_string();
    
    return s;
}

std::string Network::trace(){
    char buffer[16];
    std::string s;

    float* values = input->get_values();

    s += "[";
    for (int i = 0; i < input->get_nunits(); i++){
        // Format to have leading space if + and 5 decimal places
        sprintf(buffer, "% .5f ", values[i]);   
        s += buffer;
    }
    s += "]\n";

    for (Layer* layer : layers){
        values = layer->get_values();
        
        s += "[";
        for (int i = 0; i < layer->get_nunits(); i++){
            // Format to have leading space if + and 5 decimal places
            sprintf(buffer, "% .5f ", values[i]);   
            s += buffer;
        }
        s += "]\n";
    }

    return s;
}