import numpy as np
data = np.load('datasets/atari_data_ALE_Pong_v5.npz')
print(data['obs'].shape) # Kết quả in ra: (2581, 84, 84)
