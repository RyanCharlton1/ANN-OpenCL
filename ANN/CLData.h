#pragma once 

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <iostream>
#include <map>

// norm2d and none don't have distinct kernels to load
#define FUNCTION_COUNT norm2d

enum Function {
    mat_vec_mult,
    vec_vec_mult,
    vec_vec_add_inplace,
    mat_vec_mult_trans,
    weight_grad,
    bias_grad,
    ReLU, ReLU_der,
    leaky_ReLU, leaky_ReLU_der,
    softmax, softmax_sum,
    MSE, MSE_der,
    cross_entropy, cross_entropy_der,
    GrdDsc,
    adam,
    l2_reg,
    avg,
    var,
    affine,
    norm1d,
    norm1d_der,
    gamma_grad,
    norm2d,
    none
};

struct CLdata{
    cl_int           device_count;
    cl_device_id*    device_list = nullptr;
    cl_context       context;
    cl_command_queue command_queue;
    cl_program       program;

    std::map<Function, cl_kernel> kernels;

    void free();
};

const char* cl_errstr(cl_int error);
void cl_print_err(const char* entry, cl_int error);
// Allocate a cl buffer with data in it, NULL for 0 init
cl_mem alloc_buffer(cl_context context, const char* name, 
    size_t size, void* data=NULL, cl_mem_flags flag=CL_MEM_READ_WRITE);

Function derivative(Function f);
const char* function_to_string(Function f);

// Return a string to represent types to be used in a Functions kernel
// args for use in a printf syle variadic function 
// type : char, int : i, float : f, cl_mem* : c
const char* function_arg_string(Function f);
// Variadic function to set args and call kernel for given Function f.
// All cl_mem should be given as pointers.
void call_kernel(CLdata* cl, Function fun, 
                 cl_uint         work_dim,
                 const size_t*   global_work_offset,
                 const size_t*   global_work_size,
                 const size_t*   local_work_size,
                 cl_uint         num_events_in_wait_list,
                 const cl_event* event_wait_list,
                 cl_event*       event ...);
 