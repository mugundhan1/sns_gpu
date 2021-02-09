import numpy as np
import matplotlib.pyplot as plt

file = "../data/test_data.bin.out"

data_fft = np.fromfile(file,dtype=np.complex64)
plt.plot(abs(data_fft[:16384]))
plt.show()
