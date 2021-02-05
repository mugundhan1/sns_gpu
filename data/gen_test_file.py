import numpy as np
import matplotlib.pyplot as plt
fs = 1.25e6
ts = 1/fs
Npts = 4096*4096
tmax = Npts*ts
tax = np.linspace(0,tmax-ts,Npts)
sig = np.float32(np.sin(2*np.pi*100e3*tax))

data_file = open('test_data.bin','wb')
sig.tofile(data_file)
data_file.close()

# now read and verify if data has been written correctly!
data_file = open('test_data.bin','rb')
sig_rd = np.fromfile(data_file,dtype=np.float32)

np.testing.assert_equal(sig,sig_rd)

