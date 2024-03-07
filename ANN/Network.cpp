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
    srand(392840238490);

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

void Network::clear_accumulators(){
    for (Layer* layer : layers)
        layer->clear_accumulators();
}

void Network::fit_batch(float* data, int dsize, float* exp, int esize, 
                        int bsize){
    
    if (esize != get_output_layer()->get_nunits()){
        std::cout << "Expected size doesn't match network output" << std::endl;
        return;
    }

    // Zero out weight and bias grads from last batch
    clear_accumulators();

    float* output    = get_output();
    float  loss      = 0.0f;
    float* loss_grad = new float[esize];

    // For each instance in batch
    for (int b = 0; b < bsize; b++){
        // Calculate the networks output for given data
        calc(&data[b * dsize], dsize);

        float error;
        for (int i = 0; i < esize; i++){
            error        = exp[b * esize + i] - output[i];
            loss        += error * error / (2.0f * esize);
            loss_grad[i] = -error / (float)esize;
        }

        // Calcualte activation gradient
        Layer* out_layer = get_output_layer();
        out_layer->calc_act_grad();

        // Calculate the gradient of loss at the output layer dL/dy by
        // multiplying loss_grad and act_grad, dL/da * da/dy = dL/dy
        float* out_values_grad = out_layer->get_values_grad();
        float* out_act_grad    = out_layer->get_act_grad();

        for (int i = 0; i < out_layer->get_nunits(); i++)
            out_values_grad[i] = loss_grad[i] * out_act_grad[i];
    
        // Back progpagate
        for (int i = layers.size() - 1; i >= 0; i--){
            Layer* layer = layers[i];
            Layer* prev  = i != 0 ? layers[i-1] : input;
            // Calculate weight gradient dL/dw
            layer->accumulate_weight_grad();

            if (prev == input) break;
            // Calculate prev Layer's act_grad at the pre_act_values dA/dz
            prev->calc_act_grad();
            // Calculate prev Layer's loss_grad dL/dA by multiplying 
            // dL/dy and dy/dA(w)
            layer->calc_loss_grad();
            // Calculate prev Layer's value_grad by multiplying 
            // dL/dA(act_grad) and dA/dz(loss_grad)
            prev->calc_value_grad();
        }
    }

    // Average accumulated values
    loss /= (float)bsize;

    // Adjust weights with optimiser
    for (Layer* layer : layers){
        layer->average_accumulators(bsize);
        layer->optimise(learn_rate);
    }

    
    std::cout << "Loss: " << loss << std::endl;
}

// Train the network on provided data using expected results
void Network::fit(float* data, int dsize, float* exp, int esize,
                  int batches, int bsize, int epochs){

    for (int e = 0; e < epochs; e++){
        for (int b = 0; b < batches; b++){
            fit_batch(&data[dsize * bsize * b], dsize,
                      &exp[esize * bsize * b], esize, bsize);
        }
    }
    /*
    if (esize != get_output_layer()->get_nunits()){
        std::cout << "Expected size doesn't match network output" << std::endl;
        return;
    }

    calc(data, dsize);
    // Calculate loss and loss grad using MSE
    float* output    = get_output();
    float  loss      = 0.0f;
    float* loss_grad = new float[esize]; 

    float error;
    for (int i = 0; i < esize; i++){
        error        = exp[i] - output[i];
        loss        += error * error / 2.0f;
        loss_grad[i] = -error / (float)esize;
    }
    loss /= (float)esize;

    std::cout << "Error: " << error << std::endl;
    std::cout << "Loss: " << loss << std::endl;
    
    // Calcualte activation gradient
    Layer* out_layer = get_output_layer();
    out_layer->calc_act_grad();

    // Calculate the gradient of loss at the output layer dL/dy by
    // multiplying loss_grad and act_grad, dL/da * da/dy = dL/dy
    float* out_values_grad = out_layer->get_values_grad();
    float* out_act_grad    = out_layer->get_act_grad();

    for (int i = 0; i < out_layer->get_nunits(); i++)
        out_values_grad[i] = loss_grad[i] * out_act_grad[i];
    
    // Back progpagate
    for (int i = layers.size() - 1; i >= 0; i--){
        Layer* layer = layers[i];
        Layer* prev  = i != 0 ? layers[i-1] : input;
        // Calculate weight gradient dL/dw
        layer->accumulate_weight_grad();

        if (prev == input) break;
        // Calculate prev Layer's act_grad at the pre_act_values dA/dz
        prev->calc_act_grad();
        // Calculate prev Layer's loss_grad dL/dA by multiplying 
        // dL/dy and dy/dA(w)
        layer->calc_loss_grad();
        // Calculate prev Layer's value_grad by multiplying 
        // dL/dA(act_grad) and dA/dz(loss_grad)
        prev->calc_value_grad();
    }

    // Adjust weights with optimiser
    for (Layer* layer : layers)
        layer->optimise(learn_rate);
    */
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