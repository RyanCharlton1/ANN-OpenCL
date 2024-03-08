#include <ANN/Network.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <stdlib.h>

void Network::create_kernel(Function func){
    cl_int status;

    cl.kernels.insert({func, clCreateKernel(
        cl.program, function_to_string(func), &status)});
    char buffer[128];

    sprintf(buffer, "%s kernel creation", function_to_string(func));
    cl_print_err(buffer, status);
}

void Network::create_kernels(){
    for (int i = 0; i < FUNCTION_COUNT; i++)
        create_kernel((Function)i);
}

Network::Network(int ninput){
    this->ninput = ninput;
    // Activation doesn't matter as it will never update()
    input = new Dense(ninput, ReLU);

    // Set up OpenCL to be called later
    // Get platform and device info 
    cl_platform_id* platforms = NULL;
    cl_uint         num_platforms;

    // Set up the platform
    cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_print_err("Get platform count:\t", status);

    platforms = new cl_platform_id[num_platforms]();
        
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    cl_print_err("Get platform ids:\t", status);

    // Get the devices list and choose the device you want
    cl_uint num_devices;

    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL,
        &num_devices);
    cl_print_err("Get device count:\t", status);

    cl.device_list = new cl_device_id[num_devices]();

    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices,
        cl.device_list, NULL);
    cl_print_err("Get device ids:\t\t", status);

    // Create a OpenCL contxt for each device
    cl.context = clCreateContext(NULL, num_devices, cl.device_list, NULL, 
        NULL, &status);
    cl_print_err("Context creation:\t", status);

    // Enables out of order execution 
    cl_queue_properties prop[] = 
        {CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
    // Create a command buffer
    cl.command_queue = clCreateCommandQueueWithProperties(
        cl.context, cl.device_list[0], prop, &status);
    cl_print_err("Command queue creation:\t", status);

    // Load kernel code from file
    std::ifstream f("../ANN/kernel.cl");
    std::stringstream buffer;
    buffer << f.rdbuf();

    std::string str         = buffer.str();
    const char* kernel_code = str.data();
        
    // Build program from source code at start
    cl.program = clCreateProgramWithSource(
        cl.context, 1, &kernel_code, NULL, &status);
    cl_print_err("Progam creation:\t", status);

    status = clBuildProgram(
        cl.program, 1, cl.device_list, NULL, NULL, NULL);
        
    cl_print_err("Program build:\t\t", status);

    if (status != CL_SUCCESS){
        size_t logsize;
        clGetProgramBuildInfo(cl.program, cl.device_list[0], 
        CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsize);
        char* plog = new char[logsize];
        clGetProgramBuildInfo(cl.program, cl.device_list[0], 
        CL_PROGRAM_BUILD_LOG, logsize, plog, NULL);
        
        std::cout << plog;
        delete[] plog;
    }

    create_kernels();
        
    delete[] platforms;
}

Network::~Network(){
    delete input;
}

void Network::compile(float learn_rate){
    this->learn_rate = learn_rate;

    // Temporary for debugging purposes
    srand(392840238490);
    input->set_cl(&cl);
    // Connect each Layer to the last, inits memory and sets vars 
    // refering to prev layer 
    Layer* prev = input;
    for (Layer* layer : layers) {
        layer->set_cl(&cl);
        layer->connect(prev);
        prev = layer;
    }
}

void Network::cl_to_host_weights(){
    for (Layer* layer : layers)
        layer->cl_to_host_weights();
}

void Network::cl_to_host_values(){
    //input->cl_to_host_values();
    for (Layer* layer : layers)
        layer->cl_to_host_values();
}

// Inits a read only cl buffer with the input data in
void Network::set_input(float* data, int dsize){
    //std::memcpy(input->get_values(), data, dsize * sizeof(float));
    cl_event finished;
    clEnqueueWriteBuffer(
        cl.command_queue, *input->get_values_clmem(), CL_TRUE, 0, 
        dsize * sizeof(float), data, 0, NULL, &finished);
    clFinish(cl.command_queue);
}

void Network::calc(float* data, int dsize){
    set_input(data, dsize);

    for (Layer* layer : layers)
        layer->update();
}

void Network::clear_accumulators(){
    for (Layer* layer : layers)
        layer->clear_accumulators();
}

void Network::fit_batch_cl(float* data, int dsize, float* exp, int esize,
                           int bsize){

    // Feed forward whole batch 
    calc(data, dsize * bsize);

}
    

void Network::fit_batch(float* data, int dsize, float* exp, int esize, 
                        int bsize){
    

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

    if (dsize != ninput){
        std::cout << "Invalid input data size" << std::endl;
        return;
    }
    
    if (esize != get_output_layer()->get_nunits()){
        std::cout << "Expected size doesn't match network output" << std::endl;
        return;
    }
    // OpenCL init mem here
    input->init_cl_mem(cl.context, bsize);
    for (Layer* layer : layers)
        layer->init_cl_mem(cl.context, bsize);

    

    for (int e = 0; e < epochs; e++){
        for (int b = 0; b < batches; b++){
            fit_batch_cl(&data[dsize * bsize * b], dsize,
                      &exp[esize * bsize * b], esize, bsize);
        }
    }

    cl_to_host_values();
    for (Layer* layer: layers)
        layer->free_cl_mem();
}

std::string Network::to_string(){
    std::string s;
    
    for (Layer* layer : layers)
        s += layer->to_string();
    
    return s;
}

// Don't trace before a calc, the value memory might be filled with
// huge numbers potentially causing buffer overflow 
std::string Network::trace(){
    char buffer[128];
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