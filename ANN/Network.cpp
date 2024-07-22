#include <ANN/Network.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdlib.h>

// Creare an OpenCL kernel object for function func
void Network::create_kernel(Function func){
    cl_int status;

    cl.kernels.insert({func, clCreateKernel(
        cl.program, function_to_string(func), &status)});
    char buffer[128];

    sprintf(buffer, "%s kernel creation", function_to_string(func));
    cl_print_err(buffer, status);
}

// Create a kernel object for each function 
void Network::create_kernels(){
    for (int i = 0; i < FUNCTION_COUNT; i++)
        create_kernel((Function)i);
}

// Create all OpenCL objects for queueing and executing kernels
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

// Free all memory associated with the Network and it's Layers
Network::~Network(){
    for (Layer* l : layers)
        delete l;
    
    cl.free();
    delete input;
}

// Connect Layers, initing their memory, generating weights/bias, giving
// opencl objects for calling kernels. Also seed random number generator
// for weights and set params for training
void Network::compile(
    float learn_rate, Function loss, Function opt, Function reg, float lambda){
    
    this->learn_rate = learn_rate;
    this->loss       = loss;
    this->opt        = opt;
    this->reg        = reg;
    this->lambda     = lambda;

    // For debugging purposes
    // srand(392840238490);
    srand(time(NULL));
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

// Allocate gpu buffers for expected results and loss calculations
void Network::init_clmem(int bsize){
    int output_size = get_output_layer()->get_nunits();    
    
    expected_clmem = alloc_buffer(
        cl.context, "expected_clmem", bsize * output_size * sizeof(float));

    // Computing eache element and summing in host mem is much easier 
    // and more efficient than implementing a mutex for the single value
    loss_clmem = alloc_buffer(
        cl.context, "loss_clmem", bsize * output_size * sizeof(float));

    loss_grad_clmem = alloc_buffer(
        cl.context, "loss_grad_clmem", bsize * output_size * sizeof(float));
}

// Free gpu buffers
void Network::free_clmem(){ 
    clReleaseMemObject(expected_clmem);
    clReleaseMemObject(loss_clmem);
    clReleaseMemObject(loss_grad_clmem);
}

// Copy expected results from host to gpu buffer
void Network::host_to_cl_expected(float* exp, int esize){
    cl_int status = clEnqueueWriteBuffer(
        cl.command_queue, expected_clmem, CL_TRUE, 0, esize * sizeof(float),
        exp, 0, NULL, NULL);
    
    cl_print_err("host to cl expected", status);
    clFinish(cl.command_queue);
}

// Copy weights from gpu to host for each Layer
void Network::cl_to_host_weights(){
    for (Layer* layer : layers)
        layer->cl_to_host_weights();
}

// Copy values from gpu to host for each Layer
void Network::cl_to_host_values(){
    input->cl_to_host_values();
    for (Layer* layer : layers)
        layer->cl_to_host_values();
}

// Copy normalisation values from gpu to host for each Layer
void Network::cl_to_host_norm(){
    for (Layer* layer : layers)
        if (layer->get_norm())
            layer->cl_to_host_norm();
}

// Copy all values from gpu to host for each Layer
void Network::cl_to_host(){
    cl_to_host_values();
    cl_to_host_weights();
    cl_to_host_norm();
}

// Inits a read only cl buffer with the input data in
void Network::set_input(float* data, int dsize){
    //std::memcpy(input->get_values(), data, dsize * sizeof(float));
    clEnqueueWriteBuffer(
        cl.command_queue, input->get_values_clmem(), CL_TRUE, 0, 
        dsize * sizeof(float), data, 0, NULL, NULL);

    clFinish(cl.command_queue);
}

// Complete forward pass on gpu
void Network::calc_cl(float* data, int dsize){
    set_input(data, dsize);

    for (Layer* layer : layers)
        layer->update();
}

// Init gpu buffers for each Layer, calculate forward pass and read
// result from gpu
float* Network::calc(float* data, int dsize){

    if (dsize != ninput){
        std::cout << "Invalid input data size" << std::endl;
        return nullptr;
    }
    
    // OpenCL init mem 
    init_clmem(1);
    input->init_cl_mem(opt, 1);
    for (Layer* layer : layers)
        layer->init_cl_mem(opt, 1);

    calc_cl(data, dsize);

    cl_to_host();

    // OpenCL free mem
    free_clmem();
    for (Layer* layer: layers)
        layer->free_cl_mem();

    return get_output();
}

// Calculate loss and loss gradient for the output Layer. Return loss as
// a metric to monitor during training
float Network::calc_loss(int bsize){
    Layer* out_layer   = get_output_layer();
    int global_size[1] = { bsize * out_layer->get_nunits() } ;
    int out_nunits = out_layer->get_nunits();

    switch (loss)
    {
    case MSE:
        call_kernel(&cl, MSE, 
            1, NULL, global_size, NULL, 0, NULL, NULL,
            // Args
            expected_clmem,
            out_layer->get_values_clmem(),
            loss_clmem);

        call_kernel(&cl, MSE_der,
            1, NULL, global_size, NULL, 0, NULL, NULL,
            // Args
            out_nunits,
            expected_clmem,
            out_layer->get_values_clmem(),
            loss_grad_clmem);

        break;

    case cross_entropy:
        call_kernel(&cl, cross_entropy,
            1, NULL, global_size, NULL, 0, NULL, NULL,
            // Args
            expected_clmem,
            out_layer->get_values_clmem(),
            loss_clmem);

        call_kernel(&cl, cross_entropy_der,
            1, NULL, global_size, NULL, 0, NULL, NULL,
            // Args
            expected_clmem,
            out_layer->get_values_clmem(),
            out_layer->get_input_grad_clmem());

        break;
    
    default:
        std::cout << "Loss function not found" << std::endl;
        break;
    }

    clFinish(cl.command_queue);
    
    float* loss_arr = new float[global_size[0]];
    cl_int status = clEnqueueReadBuffer(
        cl.command_queue, loss_clmem, CL_TRUE, 0, global_size[0] * sizeof(float),
        loss_arr, 0, NULL, NULL);
    cl_print_err("Read loss buffer", status);

    float l = 0.0f;
    for (int i = 0; i < global_size[0]; i++)
        l += loss_arr[i];

    delete[] loss_arr;
    //std::cout << "Loss: " << l;
    return l / (float)bsize;
}

// Calculate the output Layer's input value gradient by multiplying the 
// loss gradient by the activation gradient and norm gradient if used
void Network::calc_output_value_grad(int bsize){
    // Cross entropy derivative is combined with softmax in the loss
    // calculation and stored in out_values_grad already.
    if (loss == cross_entropy)
        return;

    Layer* out_layer = get_output_layer();

    // Calculate dA/dz
    out_layer->calc_act_grad();

    // Multiply activation grad by loss grad to get dL/dz
    int out_nunits[1] = { out_layer->get_nunits() };
    int work_size[1]  = { bsize * out_nunits[0] };

    call_kernel(&cl, vec_vec_mult,
        1, NULL, work_size, NULL, 0, NULL, NULL,
        // Args
        loss_grad_clmem,
        out_layer->get_act_grad_clmem(),
        out_layer->get_input_grad_clmem());

    // If using norm, previous step calculated dL/dAf because it hasn't
    // accounted for the affine and normalisation gradients
    if (out_layer->get_norm())
        out_layer->calc_norm_grad();

    clFinish(cl.command_queue);
}

// Complete a single forward and backwards pass on a batch of data. 
// Calculating loss, gradients and updating weights. Return loss as a
// metric to monitor during training
float Network::fit_batch_cl(
    float* data, int dsize, float* exp, int esize, int bsize, int instance){

    // Feed forward whole batch 
    calc_cl(data, dsize * bsize);
    // Send expected results to cl_mem
    host_to_cl_expected(exp, esize * bsize);
    // Calculate loss 
    float l = calc_loss(bsize);

    // Calculate value gradient at last layer by multiplying its
    // act gradient by loss gradient, dL/dy = dL/dA * dA/dy
    calc_output_value_grad(bsize);

    // Backpropagate
    for (int i = layers.size() - 1; i >= 0; i--){
        Layer* layer = layers[i];
        Layer* prev  = i != 0 ? layers[i-1] : input;
        // Calculate weight gradient dL/dw
        layer->calc_weight_grad(reg, lambda);

        if (prev == input) break;
        // Calculate prev Layer's act_grad dA/dz at the pre_act_values 
        prev->calc_act_grad();
        // Calculate prev Layer's loss_grad dL/dA by multiplying 
        // dL/dy and dy/dA(w)
        layer->calc_prev_output_grad();
        // Calculate prev Layer's value_grad by multiplying 
        // dL/dA(act_grad) and dA/dz(loss_grad)
        prev->calc_input_grad();
        // If using norm, previous step computes dL/Af
        if (prev->get_norm())
            prev->calc_norm_grad();
    }

    // Update learnable parameters
    for (Layer* layer : layers)
        layer->optimise(opt, learn_rate, instance);

    clFinish(cl.command_queue);

    return l;
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

    // OpenCL init mem 
    init_clmem(bsize);
    input->init_cl_mem(opt, bsize);
    for (Layer* layer : layers)
        layer->init_cl_mem(opt, bsize);
    
    // Set printing floats to 2 decimal places
    std::cout << std::fixed << std::setprecision(2);
    
    // Set width of space for printing batch progress to number of 
    // digits in batches
    int l = floor(log10((float)batches));
    std::cout << std::setw(l);

    for (int e = 0; e < epochs; e++){
        // Time stamp for start of epoch
        auto t_estart = std::chrono::high_resolution_clock::now();
        
        // If adam optimsier, clear moving averages 
        if (opt == adam){
            for (Layer* l : layers)
                l->zero_adam_avgs();
        }

        for (int b = 0; b < batches; b++){
            // Calculate loss and update weights
            float l = fit_batch_cl(&data[dsize * bsize * b], dsize,
                      &exp[esize * bsize * b], esize, bsize, b);

            // Print progress info
            std::cout << "\r" << b+1 << "/" << batches << "\t";
            
            // Progress bar
            float ratio = (float)(b+1) / (float)batches;
            int   bars  = ratio * 20;

            std::cout << '[';
            for (int i = 0; i < bars; i++)
                std::cout << '=';

            std::cout << '>';
            for (int i = 0; i < 20 - bars; i++)
                std::cout << ' ';

            std::cout << ']';
            std::cout << " loss: " << l;
            
            // Time stamp for end of batch
            auto t_now = std::chrono::high_resolution_clock::now();

            // Time 
            std::cout << " avg. batch time: ";
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_estart).count() / (b+1);
            
            std::cout << " ms epoch time: ";
            std::cout << std::chrono::duration<double>(t_now - t_estart).count();
            std::cout << "s";

            // Force console to print
            fflush(NULL);
        }
        std::cout << std::endl;
    }

    // Read back values from gpu
    cl_to_host();

    // OpenCL free mem
    free_clmem();
    input->free_cl_mem();
    for (Layer* layer: layers)
        layer->free_cl_mem();
}

// Evaluate network on test data and expected results and print accuracy
void Network::evaluate(float* test, int tsize, float* exp, int esize,
                       int count){

    // Track number of correct classifications
    int classify_correct = 0;
    // TODO: implement regression evaluation

    std::cout << std::fixed << std::setprecision(2);
    // TODO: optimise this if necersarry
    for (int i = 0; i < count; i++){
        
        // Print progress bar
        std::cout << "\r" << i+1 << "/" << count;
        
        float ratio = (float)(i+1) / (float)count;
        int   bars  = ratio * 20;

        std::cout << '[';
        for (int i = 0; i < bars; i++)
            std::cout << '=';

        std::cout << '>';
        for (int i = 0; i < 20 - bars; i++)
            std::cout << ' ';

        std::cout << ']';

        // Calculate instance's prediction
        float* prediction = calc(test + i * tsize, tsize);

        switch (loss){
        case cross_entropy: {
            auto max_it = std::max_element(prediction, &prediction[esize]);
            int  max_i  = std::distance(prediction, max_it);

            if (exp[esize * i + max_i] == 1.0f)
                classify_correct += 1;

            std::cout << " acc: " << (float)classify_correct / (float)i;
            } break;

        case MSE:
            std::cout << "tbc" << std::endl;
            break;
        }
    }
    std::cout << std::endl;
}

// Return a string representation of the network
std::string Network::to_string(){
    std::string s;
    
    for (Layer* layer : layers)
        s += layer->to_string();
    
    return s;
}

// Return values currently stored in the network. Used for debugging.
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