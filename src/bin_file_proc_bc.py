import os
import numpy as np
import bifrost.pipeline as bfp
from bifrost.blocks import BinaryFileReadBlock, BinaryFileWriteBlock, FftBlock, CopyBlock
from bifrost.blocks import detect, accumulate
import glob
import time

if __name__ == "__main__":
	filenames = sorted(glob.glob('../data/*.bin'))
	Nsamp = 4096*4096
	b_read = BinaryFileReadBlock(filenames,Nsamp,1,'f32')
	b_copy = CopyBlock(b_read,space='cuda',core=0,gpu=0)
	with bfp.block_scope(core=1, gpu=0,fuse = True):
		b_fft = FftBlock(b_copy,axes=1,core=1,gpu=0)
		b_fft = detect(b_fft,'scalar')
	#b_acc = accumulate(b_fft,nframe=4096)
	b_out = CopyBlock(b_fft,space='system',core=2)
	b_write = BinaryFileWriteBlock(b_out,core=3)
	#t0 = time.time()
	pipeline = bfp.get_default_pipeline()
	print(pipeline.dot_graph())
	t0 = time.time()
	pipeline.run()	 
	t1 = time.time()
	print "Time: ", t1-t0
