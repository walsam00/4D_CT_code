#Loads PSI raw data (projections, darks, blanks) from a single hdf5 file and normalizes the projections.
#Normalized projections are saved as a new hdf5 file. Normalization is done batch-wise due to memory restrictions.
#The input data format is uint16, the output is saved as float32.

import h5py
import tomopy
import tomopy.util.dtype as dtype
import numpy as np
import getopt, sys

fdir_in = ''
fdir_out = ''
f_name = ''

batch_stop = 2500 #arbitrary batch size -> adjust according to RAM availability
width = 2016 
height = 1800
depth = 27500
batch_max = 11

#get program parameters from command line options
#this code is run from a SLURM script, so this is a handy way of populating the variables
full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]

short_options = "i:o:n:s:m:w:e:d:"
long_options = ["fdir_in=", "fdir_out", "f_name=", "batch_size=", "batch_max=", "width=", "height=", "depth="]

try:
	arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
	print(str(err))
	sys.exit(2)

for current_argument, current_value in arguments:
	if current_argument in ("-i", "--fdir_in"):
		fdir_in = str(current_value)    #for example "/base_filepath/Raw_Data/N10_w_01/"
		print(fdir_in)
	if current_argument in ("-o", "--fdir_out"):
		fdir_out = str(current_value)   #for example "/base_filepath/Reconstructions/N10_w_01/"
		print(fdir_out)
	if current_argument in ("-n", "--f_name"):
		f_name = str(current_value)     #for example "N10_w_01"
		print(f_name)
	if current_argument in ("-s", "--batch_size"):
		batch_stop = int(current_value) #for example "500"
	if current_argument in ("-m", "--batch_max"):
		batch_max = int(current_value)  #for exampe "55"
	if current_argument in ("-w", "--width"):
		width = int(current_value)      #for example "2016"
	if current_argument in ("-e", "--height"):
		height = int(current_value)     #for example "1800"
	if current_argument in ("-d", "--depth"):
		depth = int(current_value)      #for example "27500"

batch_increment = batch_stop

#read acquisition log file
energy = 'Not found'
delta = 'Not found'
beta = 'Not found'
distance = 'Not found'
pixel_size = 'Not found'
projections = 'Not found'

log = open(str(fdir_in) + f_name + '.log', 'r')
log_line = log.readlines()

for line in log_line:
	if line.startswith('Actual pixel size'):
		line_split = line.split()
		pixel_size = str(line_split[-2])
	if line.startswith('Number of projections'):
		line_split = line.split()
		no_projections = int(line_split[-1])
		no_projections = str(no_projections)
	if line.startswith('Number of darks'):
		line_split = line.split()
		darks = int(line_split[-1])
		no_darks = darks
	if line.startswith('Number of flats'):
		line_split = line.split()
		flats = int(line_split[-1])
		no_flats = flats
	if line.startswith('Rot Y max position'):
		line_split = line.split()
		#nangles = int(line_split[-1])
log.close()

#normalize
print('Loading files for normalization: ', fdir_in, f_name)

batch = 1	#current batch
batch_start = 0

#create output file
with h5py.File(fdir_out + 'output.h5', 'a') as hdf:
	dset = hdf.create_dataset('normalized', (depth, height, width), dtype = 'float32', chunks=True)

#read raw data
with h5py.File(fdir_in + f_name+'.h5', 'r') as hdf:
	f = hdf['exchange/data_white']
	flat = np.array(f)
	d = hdf['exchange/data_dark']
	dark = np.array(d)

#normalize & save output
while batch <= batch_max:
	print('working on batch ', str(batch), ' out of ', str(batch_max))
	with h5py.File(fdir_in + f_name+'.h5', 'r') as hdf:
		p = hdf['exchange/data']
		proj = np.array(p[batch_start:batch_stop, :, :])

	print('Normalizing')
	proj = tomopy.normalize(proj, flat, dark, ncore=10)
	print(proj.dtype)

	print('Saving normalized projections')

	with h5py.File(fdir_out+'output.h5', 'a') as hdf:
		hdf['normalized'][batch_start:batch_stop, :, :] = proj

	batch_start = batch_stop
	batch_stop += batch_increment
	if batch_stop > depth:
		batch_stop = depth
	batch += 1
