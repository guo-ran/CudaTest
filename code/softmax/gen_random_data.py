import numpy as np 

data=np.random.rand(49152, 32768).astype(np.float16)
data.tofile("data.bin")
