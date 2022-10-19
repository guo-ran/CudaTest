import xxhash
import numpy as np 
ids = np.load("sparse_fields.npy")
print("ids",ids)
local_batch_size = ids.shape[0] // 8
fields = ids.shape[1] #26
parallel_num = 8
hash_ids = []
for id in ids.flatten():
    hash_id = xxhash.xxh64(str(id), seed=1).intdigest()
    hash_ids.append(hash_id)
hash_ids=np.array(hash_ids, dtype=np.uint64)


counter_list = []
for i in range(parallel_num):
    local_hash_ids = hash_ids[i * local_batch_size * fields : (i + 1) * local_batch_size * fields]
    local_ids = ids.flatten()[i * local_batch_size * fields : (i + 1) * local_batch_size * fields]
    unique_local_ids,index = np.unique(local_ids, return_index=True)
    unique_local_hash_ids = local_hash_ids[index]
    print(unique_local_hash_ids % 8)
    unique_counter=[0,0,0,0,0,0,0,0]
    for hash_id in unique_local_hash_ids:
        partition_id = int(hash_id % np.uint64(8))
        unique_counter[partition_id] = unique_counter[partition_id] + 1
    counter_list.append(unique_counter)
print("counter_list",counter_list)
#counter_list [[4384, 4474, 4505, 4443, 4402, 4570, 4377, 4413], [4456, 4532, 4474, 4479, 4487, 4455, 4425, 4460], [4499, 4556, 4595, 4568, 4520, 4533, 4569, 4535], [4475, 4482, 4507, 4401, 4395, 4436, 4419, 4435], [4509, 4420, 4596, 4444, 4548, 4528, 4502, 4475], [4362, 4384, 4493, 4375, 4483, 4344, 4447, 4451], [4524, 4478, 4445, 4383, 4479, 4468, 4424, 4373], [4441, 4377, 4541, 4481, 4520, 4478, 4451, 4442]]
            
