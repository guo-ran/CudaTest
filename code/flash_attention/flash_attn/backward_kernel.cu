#include<iostream>
#include "fmha.h"
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

void dump_data(std::string filename, void* ptr, size_t size) {
    void* ptr_host;
    cudaMallocHost(&ptr_host, size);
    cudaMemcpy(ptr_host, ptr, size, cudaMemcpyDefault);
    cudaDeviceSynchronize();
    std::ofstream out_os;
    out_os.open("test/" + filename);
    out_os.write(reinterpret_cast<char *>(ptr_host), size);
}

void print_param(FMHA_dgrad_params params){
    std::cout<<"o_row_stride_in_elts "<<params.o_row_stride_in_elts<<std::endl;
    std::cout<<"o_head_stride_in_elts "<<params.o_head_stride_in_elts<<std::endl;
    std::cout<<"s_stride_in_bytes "<<params.s_stride_in_bytes<<std::endl;
    std::cout<<"b, seqlen_q, seqlen_k, d "<<params.b<<" "<< params.seqlen_q<<" "<< params.seqlen_k<< " "<< params.d<<std::endl;
    std::cout<<"scale_bmm1f "<<params.scale_bmm1f<<std::endl;
    std::cout<<"scale_bmm1 "<<params.scale_bmm1<<std::endl;
    std::cout<<"p_dropout "<<params.p_dropout<<std::endl;
    std::cout<<"p_dropout_in_uint "<<params.p_dropout_in_uint<<std::endl;
    std::cout<<"p_dropout_in_uint16_t "<<params.p_dropout_in_uint16_t<<std::endl;
    std::cout<<"rp_dropout "<<params.rp_dropout<<std::endl;
    std::cout<<"scale_bmm1_rp_dropout "<<params.scale_bmm1_rp_dropout<<std::endl;
    std::cout<<"scale_dropout "<<params.scale_dropout<<std::endl;
    std::cout<<"is_bf16 "<<params.is_bf16<<std::endl;
    std::cout<<"is_causal "<<params.is_causal<<std::endl;
    std::cout<<"dq_row_stride_in_elts "<<params.dq_row_stride_in_elts<<std::endl;
    std::cout<<"dk_row_stride_in_elts "<<params.dk_row_stride_in_elts<<std::endl;
    std::cout<<"dv_row_stride_in_elts "<<params.dv_row_stride_in_elts<<std::endl;
    std::cout<<"dq_head_stride_in_elts "<<params.dq_head_stride_in_elts<<std::endl;
    std::cout<<"dk_head_stride_in_elts "<<params.dk_head_stride_in_elts<<std::endl;
    std::cout<<"dv_head_stride_in_elts "<<params.dv_head_stride_in_elts<<std::endl;
    int batch_size = 64;
    int num_head = 16;
    int max_seqlen_q = 1024;
    int max_seqlen_k = 1024;
    int head_size = 1024 / num_head;
    size_t softmax_lse_size = batch_size*num_head*max_seqlen_q*sizeof(float);
    size_t data_size_q = batch_size*num_head*max_seqlen_q*head_size*sizeof(half);
    size_t out_size = batch_size*num_head*max_seqlen_q*head_size*sizeof(half);
    dump_data("q", params.q_ptr, data_size_q);
    dump_data("k", params.k_ptr, data_size_q);
    dump_data("v", params.v_ptr, data_size_q);
    dump_data("out", params.o_ptr, out_size);
    dump_data("softmax_lse_ptr", params.softmax_lse_ptr, softmax_lse_size);
    dump_data("dsoftmax_sum", params.dsoftmax_sum, softmax_lse_size);
    dump_data("dq", params.dq_ptr, data_size_q);
    dump_data("dk", params.dk_ptr, data_size_q);
    dump_data("dv", params.dv_ptr, data_size_q);
}

void set_params_fprop(FMHA_fprop_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t num_head,
                      const size_t head_size,
                      const size_t q_row_stride,
                      const size_t k_row_stride,
                      const size_t v_row_stride,
                      const size_t q_head_stride,
                      const size_t k_head_stride,
                      const size_t v_head_stride,
                      // device pointers
                      void* q_ptr,
                      void* k_ptr,
                      void* v_ptr,
                      int *cu_seqlens_q_d,
                      int *cu_seqlens_k_d,
                      void *o_packed_d,
                      void *o_tmp_d,
                      void *s_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal) {
    Data_type data_type = DATA_TYPE_FP16;
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = false;

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.q_row_stride_in_elts = q_row_stride;//q.stride(0);
    params.k_row_stride_in_elts = k_row_stride;//k.stride(0);
    params.v_row_stride_in_elts = v_row_stride;//v.stride(0);
    params.q_head_stride_in_elts = q_head_stride;//q.stride(1);
    params.k_head_stride_in_elts = k_head_stride;//k.stride(1);
    params.v_head_stride_in_elts = v_head_stride;//v.stride(1);
    params.o_ptr = o_packed_d;
    params.o_row_stride_in_elts = num_head * head_size;
    params.o_head_stride_in_elts = head_size;
    params.o_tmp_ptr = o_tmp_d;

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * num_head * seqlen_k, data_type);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = num_head;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = head_size;

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    const float scale_bmm1 = softmax_scale;

    params.scale_bmm1f = scale_bmm1;
    set_alpha(params.scale_bmm1, scale_bmm1, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_bmm1_rp_dropout = params.rp_dropout * params.scale_bmm1f;
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);

    params.is_causal = is_causal;
}


void set_params_dgrad(FMHA_dgrad_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t num_head,
                      const size_t head_size,
                      const size_t q_row_stride,
                      const size_t k_row_stride,
                      const size_t v_row_stride,
                      const size_t q_head_stride,
                      const size_t k_head_stride,
                      const size_t v_head_stride,
                      // device pointers
                      void* q_ptr,
                      void* k_ptr,
                      void* v_ptr,
                      void* dq_ptr,
                      void* dk_ptr,
                      void* dv_ptr,
                      int *cu_seqlens_q_d,
                      int *cu_seqlens_k_d,
                      void *o_packed_d,
                      void *dq_tmp_d,
                      void *do_packed_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, num_head, head_size,
                     q_row_stride, k_row_stride, v_row_stride,
                     q_head_stride, k_head_stride, v_head_stride,
                     q_ptr, k_ptr, v_ptr,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     o_packed_d,
                     dq_tmp_d,  // Reusing the o_tmp_ptr variable to store dq_tmp
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     is_causal);

    // Set the pointers and strides.
    params.dq_ptr = dq_ptr;
    params.dk_ptr = dk_ptr;
    params.dv_ptr = dv_ptr;
    params.dq_row_stride_in_elts = q_row_stride;
    params.dk_row_stride_in_elts = k_row_stride;
    params.dv_row_stride_in_elts = v_row_stride;
    params.dq_head_stride_in_elts = q_head_stride;
    params.dk_head_stride_in_elts = k_head_stride;
    params.dv_head_stride_in_elts = v_head_stride;
    params.do_ptr = do_packed_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;
}

int main(int argc, char **argv) {
    float param_dropout = atof(argv[1]);
    int param_is_causal = atoi(argv[2]);
    int batch_size = 64;
    int num_head = 16;
    int max_seqlen_q = 1024;
    int max_seqlen_k = 1024;
    int head_size = 1024 / num_head;
    void* q;
    void* k;
    void* v;
    void* dq;
    void* dk;
    void* dv;
    void* dq_tmp;
    size_t data_size_q = batch_size*num_head*max_seqlen_q*head_size*sizeof(half);
    size_t data_size_k = batch_size*num_head*max_seqlen_k*head_size*sizeof(half);
    size_t data_size_v = batch_size*num_head*max_seqlen_k*head_size*sizeof(half);
    Check(cudaMalloc(&q, data_size_q));
    Check(cudaMalloc(&k, data_size_k));
    Check(cudaMalloc(&v, data_size_v));
    Check(cudaMalloc(&dq, data_size_q));
    Check(cudaMalloc(&dk, data_size_k));
    Check(cudaMalloc(&dv, data_size_v));
    Check(cudaMalloc(&dq_tmp, data_size_q*2));//dq_tmp float
    Check(cudaMemset(dq, 0, data_size_q));
    Check(cudaMemset(dk, 0, data_size_k));
    Check(cudaMemset(dv, 0, data_size_v));
    Check(cudaMemset(dq_tmp, 0, data_size_q*2));
    int32_t* cu_seqlens_q;
    int32_t* cu_seqlens_k;
    Check(cudaMalloc(&cu_seqlens_q, (batch_size + 1) * sizeof(int32_t)));
    Check(cudaMalloc(&cu_seqlens_k, (batch_size + 1) * sizeof(int32_t)));

    void* q_host;
    void* k_host;
    void* v_host;
    void* dq_host;
    void* dk_host;
    void* dv_host;
    Check(cudaMallocHost(&q_host, data_size_q));
    Check(cudaMallocHost(&k_host, data_size_k));
    Check(cudaMallocHost(&v_host, data_size_v));
    Check(cudaMallocHost(&dq_host, data_size_q));
    Check(cudaMallocHost(&dk_host, data_size_k));
    Check(cudaMallocHost(&dv_host, data_size_v));
    int32_t* cu_seqlens_q_host;
    int32_t* cu_seqlens_k_host;
    Check(cudaMallocHost(&cu_seqlens_q_host, (batch_size + 1) * sizeof(int32_t)));
    Check(cudaMallocHost(&cu_seqlens_k_host, (batch_size + 1) * sizeof(int32_t)));
    std::ifstream data_is;
    data_is.open("q.bin");
    data_is.read(reinterpret_cast<char *>(q_host), data_size_q);
    data_is.close();
    Check(cudaMemcpy(q, q_host, data_size_q, cudaMemcpyDefault));
    data_is.open("k.bin");
    data_is.read(reinterpret_cast<char *>(k_host), data_size_k);
    data_is.close();
    Check(cudaMemcpy(k, k_host, data_size_k, cudaMemcpyDefault));
    data_is.open("v.bin");
    data_is.read(reinterpret_cast<char *>(v_host), data_size_v);
    data_is.close();
    Check(cudaMemcpy(v, v_host, data_size_v, cudaMemcpyDefault));
    //ref to https://github.com/HazyResearch/flash-attention/blob/0c01568dafb316d3673e9dc0fef6dbbd7deabc2d/tests/test_flash_attn.py#L52
    for(int i=0;i<batch_size+1;++i){
      cu_seqlens_q_host[i] = i * max_seqlen_q;
      cu_seqlens_k_host[i] = i * max_seqlen_k;
    }
    Check(cudaMemcpy(cu_seqlens_q, cu_seqlens_q_host, (batch_size + 1) * sizeof(int32_t), cudaMemcpyDefault));
    Check(cudaMemcpy(cu_seqlens_k, cu_seqlens_k_host, (batch_size + 1) * sizeof(int32_t), cudaMemcpyDefault));

    void* out;
    size_t out_size = batch_size*num_head*max_seqlen_q*head_size*sizeof(half);
    Check(cudaMalloc(&out, out_size));
    void* dout;
    Check(cudaMalloc(&dout, out_size));
    void* softmax_lse;//float* softmax_lse;
    size_t softmax_lse_size = batch_size*num_head*max_seqlen_q*sizeof(float);
    Check(cudaMalloc(&softmax_lse, softmax_lse_size));
    void* softmax_d;
    Check(cudaMalloc(&softmax_d, softmax_lse_size));
    Check(cudaMemset(softmax_d, 0, softmax_lse_size));
    
    void* softmax_lse_host;//float* softmax_lse;
    Check(cudaMallocHost(&softmax_lse_host, softmax_lse_size));
    void* out_host;
    Check(cudaMallocHost(&out_host, out_size));
    void* dout_host;
    Check(cudaMallocHost(&dout_host, out_size));

    data_is.open("out.bin");
    data_is.read(reinterpret_cast<char *>(out_host), out_size);
    data_is.close();
    Check(cudaMemcpy(out, out_host, out_size, cudaMemcpyDefault));
    data_is.open("dout.bin");
    data_is.read(reinterpret_cast<char *>(dout_host), out_size);
    data_is.close();
    Check(cudaMemcpy(dout, dout_host, out_size, cudaMemcpyDefault));
    data_is.open("softmax_lse.bin");
    data_is.read(reinterpret_cast<char *>(softmax_lse_host), softmax_lse_size);
    data_is.close();
    Check(cudaMemcpy(softmax_lse, softmax_lse_host, softmax_lse_size, cudaMemcpyDefault));
    
    cudaDeviceProp dprops;
    Check(cudaGetDeviceProperties(&dprops, 0));
    float p_dropout = param_dropout;//0.2;
    bool is_dropout = p_dropout > 0.0;
    bool is_sm75 = dprops.major == 7 && dprops.minor == 5;
    bool is_sm80 = dprops.major == 8 && dprops.minor == 0;
    bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
    int blocksize_c = ((head_size == 128 && (is_dropout || !is_sm80)) || (is_sm75 && head_size == 64 && is_dropout)) ? 128 : 256;
    bool loop = max_seqlen_k > blocksize_c;
    
    cudaStream_t stream;
    Check(cudaStreamCreate(&stream));
    bool return_softmax = false;
    const float softmax_scale =  1.f / sqrtf(head_size);
    const bool is_causal = param_is_causal>0;
    Launch_params<FMHA_dgrad_params> launch_params(&dprops, stream, is_dropout, return_softmax);

    const size_t q_row_stride = 1024;//num_head * max_seqlen_q * head_size;
    const size_t k_row_stride = 1024;//num_head * max_seqlen_k * head_size;
    const size_t v_row_stride = 1024;//num_head * max_seqlen_k * head_size;
    const size_t q_head_stride = 64;//max_seqlen_q * head_size;
    const size_t k_head_stride = 64;//max_seqlen_k * head_size;
    const size_t v_head_stride = 64;//max_seqlen_k * head_size;

    set_params_dgrad(launch_params.params,
                     batch_size,
                     max_seqlen_q,
                     max_seqlen_k,
                     num_head,
                     head_size,
                     q_row_stride, k_row_stride, v_row_stride,
                     q_head_stride, k_head_stride, v_head_stride,
                     q, k, v,
                     dq, dk, dv,
                     cu_seqlens_q,
                     cu_seqlens_k,
                     out,
                     loop ? dq_tmp : nullptr,
                     dout,
                     softmax_lse,
                     softmax_d,
                     p_dropout,
                     softmax_scale,
                     is_causal);
    dump_data("softmax_lse1",softmax_lse,softmax_lse_size);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    int64_t counter_offset = launch_params.elts_per_thread;
    at::PhiloxCudaState rng_engine_inputs;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        //std::lock_guard<std::mutex> lock(gen->mutex_);
        //launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
        //TODO:
        uint64_t seed = 0;
        uint64_t offset = 0;
        launch_params.params.philox_args = at::PhiloxCudaState(seed, offset);
    }
    Check(cudaDeviceSynchronize());
    Check(cudaGetLastError());
    std::cout<<"before run kernel"<<std::endl;
    print_param(launch_params.params);
    run_fmha_dgrad_fp16_sm80(launch_params, stream);

    Check(cudaMemcpy(dq_host, dq, data_size_q, cudaMemcpyDefault));
    Check(cudaMemcpy(dk_host, dk, data_size_k, cudaMemcpyDefault));
    Check(cudaMemcpy(dv_host, dv, data_size_v, cudaMemcpyDefault));
    Check(cudaDeviceSynchronize());
    Check(cudaGetLastError());
    std::ofstream out_os;
    out_os.open("dq.bin");
    out_os.write(reinterpret_cast<char *>(dq_host), data_size_q);
    out_os.close();
    out_os.open("dk.bin");
    out_os.write(reinterpret_cast<char *>(dk_host), data_size_k);
    out_os.close();
    out_os.open("dv.bin");
    out_os.write(reinterpret_cast<char *>(dv_host), data_size_v);
    out_os.close();
    return 0;
}
