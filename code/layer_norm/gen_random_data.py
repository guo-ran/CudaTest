import numpy as np 

data=np.random.rand(49152, 1022).astype(np.float16)
data.tofile("data.bin")
