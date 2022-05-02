#Loads PSI raw data (projections, darks, blanks) from a single hdf5 file and normalizes the projections.
#Normalized projections are saved as a new hdf5 file. Normalization is done batch-wise due to memory restrictions.
#The input data format is uint16, the output is saved as float32.

import h5py
import tomopy
import tomopy.util.dtype as dtype

import matplotlib.pyplot as plt
import numpy as np
import getopt, sys

fdir_in = ''
fdir_out = ''
f_name = ''

batch_stop = 2500 #5000 #arbitrary batch size, 1/10 of images -> adjust according to RAM availability
width = 2016 #1536 #2016
height = 1800 #760 #1800
depth = 27500
batch_max = 11

full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]

short_options = "i:o:n:s:m:w:e:d:h"
long_options = ["fdir_in=", "fdir_out", "f_name=", "batch_size=", "batch_max=", "width=", "height=", "depth=", "help"]

try:
	arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
	print(str(err))
	sys.exit(2)

for current_argument, current_value in arguments:
	if current_argument in ("-i", "--fdir_in"):
		fdir_in = str(current_value)
		print(fdir_in)
	if current_argument in ("-o", "--fdir_out"):
		fdir_out = str(current_value)
		print(fdir_out)
	if current_argument in ("-n", "--f_name"):
		f_name = str(current_value)
		print(f_name)
	if current_argument in ("-s", "--batch_size"):
		batch_stop = int(current_value)
	if current_argument in ("-m", "--batch_max"):
		batch_max = int(current_value)
	if current_argument in ("-w", "--width"):
		width = int(current_value)
	if current_argument in ("-e", "--height"):
		height = int(current_value)
	if current_argument in ("-d", "--depth"):
		depth = int(current_value)

	if current_argument in ("-h", "--help"):
		print("Normalizes a .h5 file. -fi --fdir_in specifies input directory, -fo --fdir_out specifies output directory, -fn --f_name specifies file name (w/o file extention)")

batch_increment = batch_stop

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

###normalize###
print('Loading files for normalization: ', fdir_in, f_name)

method = 'FBP_CUDA'
start = 0
end = int(no_projections)
theta = None
proj = None
flat = None
dark = None
nangles = 180

#Image dimensions of the time resolved scans: 2016x1800, 17'500 projections (N19_w_02)
#To Do: Batch the loading and normalizing (to prevent out of memory crashes) -> done in batches

batch = 1	#current batch
batch_start = 0

### for testing: only normalize & save 1 timepoint ###
# batch_start = 0
# batch_stop = 500
# depth = 500
# batch_max = 1
######################################################

#try:
with h5py.File(fdir_out + 'output.h5', 'a') as hdf:
	dset = hdf.create_dataset('normalized', (depth, height, width), dtype = 'float32', chunks=True)
#except:
#	print('failed to create output file template')

with h5py.File(fdir_in + f_name+'.h5', 'r') as hdf:
	f = hdf['exchange/data_white']
	flat = np.array(f)
	d = hdf['exchange/data_dark']
	dark = np.array(d)

while batch <= batch_max:
	print('working on batch ', str(batch), ' out of ', str(batch_max))
	with h5py.File(fdir_in + f_name+'.h5', 'r') as hdf:
		p = hdf['exchange/data']
		proj = np.array(p[batch_start:batch_stop, :, :])
	#theta = tomopy.angles(proj.shape[0], 0, nangles)

	###plt.imshow(proj[0, :, :], cmap='Greys_r')
	###plt.show()

	#if (theta is None):
	#	theta = tomopy.angles(proj.shape[0])
	#else:
	#	pass

	#normalize

	print('Normalizing')
	proj = tomopy.normalize(proj, flat, dark, ncore=10)
	###plt.imshow(proj[0, :, :], cmap='Greys_r')
	###plt.show()
	print(proj.dtype)

	#plt.imshow(proj[0, :, :])
	#plt.show()

	#print(proj.dtype)
	#proj
	# if proj_height > 2100:
		# proj_out_height = 2100
	# else:
	# if proj_height > 525:
		# proj_out_height = 525
	# if proj_width*2 > 5000:.
		# proj_out_width = 5000
	# else:
		# if proj_width*2 > 1250:
		# proj_out_width = 1250

	### !!!!!!!!!!!! ###
	# batch_start = 0
	# batch_stop = 500
	####################

	print('Saving normalized projections')
	###plt.imshow(proj[0, :, :], cmap='Greys_r')
	###plt.show()
	with h5py.File(fdir_out+'output.h5', 'a') as hdf:
		hdf['normalized'][batch_start:batch_stop, :, :] = proj

	batch_start = batch_stop
	batch_stop += batch_increment
	if batch_stop > depth:
		batch_stop = depth
	batch += 1
