#include <ANN/CLData.h>

#include <stdarg.h>

// https://stackoverflow.com/a/24336429
const char *cl_errstr(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

// Print error message if error is not CL_SUCCESS
void cl_print_err(const char* entry, cl_int error){
    if (error == CL_SUCCESS)
        return;

    char buffer[128];
    sprintf(buffer, "%-50s%s", entry, cl_errstr(error));
    std::cout << buffer << std::endl;
}

// Free all OpenCL resources associated with CLdata
void CLdata::free(){
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    clReleaseProgram(program);

    delete[] device_list;
}

// Allocate a cl_mem object and handle errors
cl_mem alloc_buffer(cl_context context, const char* name, 
    int size, void* data, cl_mem_flags flag){

    if (size < 1){
        //std::cout << name << " invalid size" << std::endl;
        return NULL;
    }

    cl_int status;
    cl_mem buffer = clCreateBuffer(context, flag, size, data, &status);
    
    char str[128];
    sprintf(str, "Alloc buffer %s", name);
    cl_print_err(str, status);
    return buffer;
}

// Function enum is incremented by 1 to get the derivative function
Function derivative(Function f){
    return (Function)(f+1);
}

// Get Function enum as string, necerssary for loading kernels
const char* function_to_string(Function f){
    switch (f)
    {
    case mat_vec_mult:
        return "mat_vec_mult";
    case vec_vec_mult:
        return "vec_vec_mult";
    case vec_vec_add_inplace:
        return "vec_vec_add_inplace";
    case mat_vec_mult_trans:
        return "mat_vec_mult_trans";
    case weight_grad:
        return "weight_grad";
    case bias_grad:
        return "bias_grad";
    case ReLU:
        return "ReLU";
    case ReLU_der:
        return "ReLU_der";
    case leaky_ReLU:
        return "leaky_ReLU";
    case leaky_ReLU_der:
        return "leaky_ReLU_der";
    case softmax:
        return "softmax";
    case softmax_sum:
        return "softmax_sum";
    case MSE:
        return "MSE";
    case MSE_der:
        return "MSE_der";
    case cross_entropy:
        return "cross_entropy";
    case cross_entropy_der:
        return "cross_entropy_der";
    case GrdDsc:
        return "GrdDsc";
    case adam:
        return "adam";
    case l2_reg:
        return "l2_reg";
    case avg:
        return "avg";
    case var:
        return "var";
    case affine:
        return "affine";
    case norm1d:
        return "norm1d";
    case norm1d_der:
        return "norm1d_der";
    case gamma_grad:
        return "gamma_grad";
    case convolution:
        return "convolution";
    case deconvolution:
        return "deconvolution";
    case pad_and_dilate:
        return "pad_and_dilate";
    case convolution_weight_grads:
        return "convolution_weight_grads";
    }
    return "error";
}

// String representing function arg types for call_kernel variadic 
// function to interpret input using, similar to %d in a printf call.
// char : type, i : int, f : float, c : clmem
const char* function_arg_string(Function f){
    switch (f){
    case mat_vec_mult:
        return "iiccc";
    case vec_vec_mult:
        return "ccc";
    case vec_vec_add_inplace:
        return "icc";
    case mat_vec_mult_trans:
        return "iiccc";
    case weight_grad:
        return "iiiccc";
    case bias_grad:
        return "icc";
    case ReLU:
        return "cc";
    case ReLU_der:
        return "cc";
    case leaky_ReLU:
        return "cc";
    case leaky_ReLU_der:
        return "cc";
    case softmax:
        return "iccc";
    case softmax_sum:
        return "icc";
    case MSE:
        return "ccc";
    case MSE_der:
        return "iccc";
    case cross_entropy:
        return "ccc";
    case cross_entropy_der:
        return "ccc";
    case GrdDsc:
        return "fcc";
    case adam:
        return "fcccci";
    case l2_reg:
        return "fcc";
    case avg:
        return "icc";
    case var:
        return "iccc";
    case affine:
        return "ccc";
    case norm1d:
        return "ccc";
    case norm1d_der:
        return "ccccc";
    case gamma_grad:
        return "iccc";
    case convolution:
        return "iiiiiiiiccc";
    case deconvolution:
        return "iiiiiccc";
    case pad_and_dilate:
        return "iiiiiicc";
    case convolution_weight_grads:
        return "iiiiiiiccc";
    }
    return "error";
}

// Enqueue kernel jobs with variadic arguments
void call_kernel(CLdata* cl, Function fun, cl_uint work_dim, 
    const int* global_work_offset, const int* global_work_size,
    const int* local_work_size, cl_uint num_events_in_wait_list, 
    const cl_event* event_wait_list, cl_event* event...){

    va_list     args;
    cl_int      status;
    cl_kernel   kernel = cl->kernels[fun];
    const char* types  = function_arg_string(fun);
    
    int    n;
    float  f;
    cl_mem c;

    std::string error = "call_kernel ";
    error += function_to_string(fun);
    error += " ";

    va_start(args, event);

    // Interpret variadic inputs using function_arg_string
    int i = 0;
    while (*types){
        switch (*types){
        case 'i':
            n      = va_arg(args, int);
            status = clSetKernelArg(kernel, i, sizeof(int), &n);

            cl_print_err((error + "int arg").c_str(), status);
            break;

        case 'f':
            // ‘float’ is promoted to ‘double’ when passed through ‘...’gcc
            // va_arg(args, float) leads to illegal instruction 
            f       = va_arg(args, double);
            status  = clSetKernelArg(kernel, i, sizeof(float), &f);

            cl_print_err((error + "float arg").c_str(), status);
            break;

        case 'c':
            c      = va_arg(args, cl_mem);
            status = clSetKernelArg(kernel, i, sizeof(cl_mem), &c);

            cl_print_err((error + "clmem arg").c_str(), status);
            break;

        default:
            std::cout << (error + "error").c_str() << std::endl;
            return;
        }

        types++;
        i++;
    }

    // Cast int to size_t as size_t is a larger type than int
    size_t *global_work_size_ = new size_t[work_dim];

    size_t *global_offset_size_   = 
        global_work_offset ? new size_t[work_dim] : NULL;
    size_t *local_work_size_    = 
        local_work_size ? new size_t[work_dim] : NULL;

    for (int i = 0; i < work_dim; i++){
        global_work_size_[i] = global_work_size[i];

        if (global_offset_size_)
            global_offset_size_[i] = global_work_offset[i];

        if (local_work_size_)
            local_work_size_[i] = local_work_size[i];
    }

    // Queue kernel jobs
    status = clEnqueueNDRangeKernel(
        cl->command_queue, kernel, work_dim, global_offset_size_, 
        global_work_size_, local_work_size_, num_events_in_wait_list, 
        event_wait_list, event);
    
    cl_print_err((error + "enqueue").c_str(), status);

    delete global_work_size_;
    if (global_offset_size_)
        delete global_offset_size_;
    if (local_work_size)
        delete local_work_size_;

    va_end(args);
}