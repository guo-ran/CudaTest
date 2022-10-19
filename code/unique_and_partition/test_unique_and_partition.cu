#include "hash_functions.cuh"
#include<iostream>
#include<fstream>
#include<vector>

void CudaCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}
const int32_t kCudaThreadsNumPerBlock = 512;
const int32_t kCudaMaxBlocksNum = 8192;

inline int32_t BlocksNum4ThreadsNum(const int32_t n) {
  return std::min((n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock, kCudaMaxBlocksNum);
}

#define CUDA_1D_KERNEL_LOOP_T(type, i, n)                                                      \
  for (type i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

template<typename K>
struct TableEntry {
  K key;
  uint32_t value;
};

template<typename K, typename V, typename IDX, typename HASH>
__global__ void HashTableUniqueAndPartitionPairs(const uint32_t table_capacity,
                                                 const uint32_t num_keys, int32_t num_partition,
                                                 IDX* unique_counts, TableEntry<K>* table,
                                                 const K* keys, const V* values,
                                                 K* partitioned_unique_keys,
                                                 V* partitioned_unique_values, IDX* reverse_index,
                                                 bool need_process_values) {
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_keys) {
    IDX r_index_plus_one = 0;
    const K key = keys[i];
    size_t key_hash = HASH()(key);
    uint32_t partition_id = key_hash % num_partition;
    IDX* unique_count = unique_counts + partition_id;
    K* unique_keys = partitioned_unique_keys + partition_id * num_keys;
    uint32_t pos = key_hash % table_capacity;
    const K key_hi = (key | 0x1);
    const K key_lo = (key & 0x1);
    uint32_t counter = 0;
    while (r_index_plus_one == 0) {
      bool prob_next = false;
      K* key_ptr = &table[pos].key;
      volatile uint32_t* table_value_ptr = &table[pos].value;
      const K old_key = atomicCAS(key_ptr, 0, key_hi);
      if (old_key == 0) {
        IDX unique_pos = atomicAdd(unique_count, 1);
        r_index_plus_one = unique_pos + 1;
        unique_keys[unique_pos] = key;
        if (need_process_values) {
          partitioned_unique_values[partition_id * num_keys + unique_pos] = values[i];
        }
        *table_value_ptr = ((r_index_plus_one << 1U) | key_lo);
      } else if (old_key == key_hi) {
        const uint32_t value = *table_value_ptr;
        if (value == 0) {
          // do nothing
        } else if ((value & 0x1) == key_lo) {
          r_index_plus_one = (value >> 1U);
        } else {
          prob_next = true;
        }
      } else {
        prob_next = true;
      }
      if (prob_next) {
        pos += 1;
        counter += 1;
        if (pos >= table_capacity) { pos -= table_capacity; }
        if (counter >= table_capacity) { __trap(); }
      }
    }
    reverse_index[i] = partition_id * num_keys + r_index_plus_one - 1;
  }
}

template<typename K, typename V, typename IDX, typename HASH>
void UniqueAndPartition(cudaStream_t cuda_stream, int64_t num_ids, size_t capacity,
                        int64_t num_partition, const K* ids, const V* table_ids,
                        IDX* num_partitioned_unique_ids_ptr, K* partitioned_unique_ids,
                        V* partitioned_unique_table_ids, IDX* inverse_unique_partition_indices,
                        void* workspace_ptr, size_t workspace_bytes, bool need_process_table_ids) {
  size_t table_capacity_bytes = capacity * sizeof(TableEntry<K>);
  CudaCheck(cudaMemsetAsync(workspace_ptr, 0, table_capacity_bytes, cuda_stream));
  CudaCheck(
      cudaMemsetAsync(num_partitioned_unique_ids_ptr, 0, num_partition * sizeof(IDX), cuda_stream));
  HashTableUniqueAndPartitionPairs<K, V, IDX, HASH>
      <<<BlocksNum4ThreadsNum(num_ids), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
          capacity, num_ids, num_partition, num_partitioned_unique_ids_ptr,
          reinterpret_cast<TableEntry<K>*>(workspace_ptr), ids, table_ids, partitioned_unique_ids,
          partitioned_unique_table_ids, inverse_unique_partition_indices, need_process_table_ids);
}

int main() {
    using K = int32_t;
    using U = uint8_t;
    using IDX = uint32_t;
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    int num_ids = 6912*26;
    int parallel_num = 8;
    size_t hash_table_capacity = num_ids * parallel_num;
    K* ids_ptr;
    K* host_ids_ptr;
    cudaMalloc(&ids_ptr, num_ids * sizeof(K));
    cudaMallocHost(&host_ids_ptr, num_ids * sizeof(K));
    std::ifstream ids_is;
    ids_is.open("ids");
    ids_is.read(reinterpret_cast<char *>(host_ids_ptr), num_ids * sizeof(K));
    CudaCheck(cudaMemcpy(ids_ptr, host_ids_ptr, num_ids * sizeof(K), cudaMemcpyDefault));

    U* table_ids_ptr;
    U* host_table_ids_ptr;
    cudaMalloc(&table_ids_ptr, num_ids * sizeof(U));
    cudaMallocHost(&host_table_ids_ptr, num_ids * sizeof(U));
    std::ifstream table_ids_is;
    table_ids_is.open("table_ids");
    table_ids_is.read(reinterpret_cast<char *>(table_ids_ptr), num_ids * sizeof(U));
    CudaCheck(cudaMemcpy(table_ids_ptr, host_table_ids_ptr, num_ids * sizeof(U), cudaMemcpyDefault));

    IDX* num_partitioned_unique;
    IDX* host_num_partitioned_unique;
    cudaMalloc(&num_partitioned_unique, parallel_num * sizeof(IDX));
    cudaMallocHost(&host_num_partitioned_unique, parallel_num * sizeof(IDX));
    K* partitioned_unique_ids;
    cudaMalloc(&partitioned_unique_ids, parallel_num * num_ids * sizeof(K));
    U* partitioned_unique_table_ids;
    cudaMalloc(&partitioned_unique_table_ids, parallel_num * num_ids * sizeof(U));
    IDX* inverse_unique_partition_indices;
    cudaMalloc(&inverse_unique_partition_indices, num_ids * sizeof(IDX));
    size_t workspace_size = hash_table_capacity * sizeof(TableEntry<K>);
    void* workspace_ptr;
    cudaMalloc(&workspace_ptr, workspace_size);

    UniqueAndPartition<K, U, IDX, ShardingHash>(
        stream, num_ids, hash_table_capacity, parallel_num,
        ids_ptr, table_ids_ptr, num_partitioned_unique,
        partitioned_unique_ids, partitioned_unique_table_ids,
        inverse_unique_partition_indices, workspace_ptr,
        workspace_size, true);
    CudaCheck(cudaMemcpy(host_num_partitioned_unique, num_partitioned_unique, parallel_num * sizeof(IDX), cudaMemcpyDefault));
    cudaDeviceSynchronize();
    cudaFree(ids_ptr);
    return 0;
    
}
