#Loads normalized and paganin filtered PSI micro CT data from a single hdf5 file and reconstructs it.
#Utilizes tomopy and astra toolbox to run a GPU-enabled FBP algorithm for quick recon times

import tomopy
import numpy as np
import h5py
import getopt, sys

#populate default variable values
sname = 'output.h5'
data_f = '/base_file_path/Reconstructions/N13_w_01/'

method = 'FBP_CUDA'
start = 0
end = 500
theta = None
proj = None
flat = None
dark = None
nangles = 180
niter = 5
rot_center = 1008.0
do_circle_corr = False

no_projections = 27500
width = 2016
height = 1800
projections_per_timepoint = 500

#populate variables via command line arguments -> useful because this code is run from different SLURM scripts with different parameters

full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]

short_options = "i:p:w:e:r:c"
long_options = ["fdir_in=", "rot_center=" "no_projections=", "width=", "height=", "circle"]

try:
	arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
	print(str(err))
	sys.exit(2)

for current_argument, current_value in arguments:
	if current_argument in ("-i", "--fdir_in"):
		data_f = str(current_value)
        #for example '/base_file_path/Reconstructions/N13_w_01/'
	if current_argument in ("-p", "--no_projections"):
		no_projections = int(current_value)
        #for example 27500
	if current_argument in ("-w", "--width"):
		width = int(current_value)
        #for example 2016
		rot_center = float(width / 2)
        #by default, the rotation center is half the width, unless overwritten later
	if current_argument in ("-e", "--height"):
		height = int(current_value)
        #for example 1800
	if current_argument in ("-c", "--circle"):
		do_circle_corr = True
        #do ring artifact removal
		print("performing circle correction")
	if current_argument in ("-r", "--rot_center"):
		rot_center = float(current_value)
        #overwrite the rotation center 

#batch by timepoint
batch_total = int((no_projections/projections_per_timepoint))
batch_current = 0

if batch_total > 1:
	print('Image reconstruction separated into ', str(batch_total), ' batches.')
else:
	print('Image reconstruction in 1 batch')

#set up iterators
proj_start = 0
proj_end = projections_per_timepoint

rec_start = 0
rec_end = height

file_creator = batch_total

#create one output file per time point
while file_creator > 0:
	file_name = 'output_reconstructed_' + str(file_creator) + '.h5'
	try:
		with h5py.File(data_f + file_name, 'a') as hdf:
			dset = hdf.create_dataset('reconstructed', (height, width, width), dtype='uint16', chunks=True)
	except:
		print('Failed to create ''reconstructed'' dataset')
	file_creator -= 1

#reconstruct batch (time point) wise
while batch_current < batch_total:
	print('Reconstructing batch ', str(batch_current + 1), ' out of ', str(batch_total))
    
    #read projections
	with h5py.File(data_f + sname, 'r') as hdf:
		d = hdf['normalized']
		proj = np.array(d[proj_start:proj_end, :, :])

	print('Loading complete, image dimensions: ', proj.shape)
    
    #generate angles -> for these data always the same, evenly spaced
	print('Generating angles')
	theta = tomopy.angles(proj.shape[0], 0, nangles)
    
	if (theta is None):
		theta = tomopy.angles(proj.shape[0])
	else:
		pass
    
    #Apply fourier wavelet stripe removal
	print("removing stripes")
	proj = tomopy.prep.stripe.remove_stripe_fw(proj)
    
    #perform actual reconstruction on GPU
	print("Reconstructing with " + str(method))
	extra_options = {}#{'MinConstraint':0}
	options = {'proj_type':'cuda', 'method':method, 'gpu_List': [1], 'extra_options':extra_options}
	recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.astra, ncore=1, options=options)

	print('recon shape: ', recon.shape)
    
    #do the ring artifact removal
	if do_circle_corr == True:
		print("performing circle correction")
		proj = tomopy.remove_ring(proj,out = proj)

	print('converting to uint16')
	#scale all values by the largest one in the whole stack and multiply it by the max value of uint16 which is 65535
	recon_max = np.amax(recon)
	recon = recon/recon_max
	recon = recon * 65535
    
    #applies a circular mask around the reconstructed slices -> covers up some of the extremely bright values FBP creates at the periphery of the reconstructed image
	recon = tomopy.circ_mask(recon, axis = 0, ratio=0.90)

	print(recon.shape)
	print('writing images to disk')

    #save reconstructed images
	write_name = 'output_reconstructed_' + str(batch_current + 1) + '.h5'

	with h5py.File(data_f + write_name, 'a') as hdf:
		hdf['reconstructed'][0:height, :, :] = recon #[rec_start:rec_end, :, :]

	batch_current += 1
    
    #increase iterators
	proj_start = proj_end
	proj_end += projections_per_timepoint
	rec_start = rec_end
	rec_end += height
