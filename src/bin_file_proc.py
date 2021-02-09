import os
import numpy as np
import bifrost.pipeline as bfp
from bifrost.blocks import BinaryFileReadBlock, BinaryFileWriteBlock, FftBlock, CopyBlock, print_header
from bifrost.blocks import detect, reduce
from bifrost.views import split_axis

import glob
import time

if __name__ == "__main__":
	filenames = sorted(glob.glob('../data/*.bin'))
	Nsamp = 1024*1024
	b_read = BinaryFileReadBlock(filenames,1024,1024,'f32')
	b_read_rs = split_axis(b_read, axis=1,n=1024,label='time')
	b_copy = CopyBlock(b_read_rs,space='cuda',core=0,gpu=0)
	with bfp.block_scope(core=1, gpu=0,fuse = True):
		b_fft = FftBlock(b_copy,axes=1,core=1,gpu=0)
		b_fft = detect(b_fft,'scalar')
	b_acc = reduce(b_fft,axis=2,op='mean')
	b_out = CopyBlock(b_acc,space='system',core=2)
	b_write = BinaryFileWriteBlock(b_out,core=3)
	print_header(b_read)
	#print_header(b_read_rs)
	print_header(b_copy)
	print_header(b_fft)
	print_header(b_acc)
	print_header(b_out)
	#print_header(b_write)
	#t0 = time.time()
	pipeline = bfp.get_default_pipeline()
	print(pipeline.dot_graph())
	t0 = time.time()
	pipeline.run()	 
	t1 = time.time()
	print "Time: ", t1-t0
