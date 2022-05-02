#from __future__ import print_function
#import six
#import os.path
#import os
#import tempfile
import tomopy
#import dxchange
#import dxchange.reader as dxreader
##import matplotlib.pyplot as plt
#from skimage import io
import numpy as np
import h5py
import getopt, sys


#from skimage.util import img_as_uint

#print("temp dir", tempfile.gettempdir())

#os.environ["TEMP"] = "~/tmp/"
#os.environ["TMP"] = "~/tmp/"
#os.environ["TMPDIR"] = "~/tmp/"

sname = 'output.h5'
data_f = '/scicore/projects/pharmatech-scicore/time_resolved_uCT_data_PSI/Reconstructions/N13_w_01/'

method = 'FBP_CUDA'
start = 0
end = 500
theta = None
proj = None
flat = None
dark = None
nangles = 180
niter = 5
rot_center = 1008.0 #768 #1008 -> half of x coordinate
do_circle_corr = False

no_projections = 27500 #50000 #17500
width = 2016 #1536 #2016
height = 1800 #760 #1800
projections_per_timepoint = 500

full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]

short_options = "i:p:w:e:r:ch"
long_options = ["fdir_in=", "rot_center=" "no_projections=", "width=", "height=", "circle", "help"]

try:
	arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
	print(str(err))
	sys.exit(2)

for current_argument, current_value in arguments:
	if current_argument in ("-i", "--fdir_in"):
		data_f = str(current_value)
	if current_argument in ("-p", "--no_projections"):
		#batch_max = int(current_value)
		no_projections = int(current_value)
	if current_argument in ("-w", "--width"):
		width = int(current_value)
		rot_center = float(width / 2)
	if current_argument in ("-e", "--height"):
		height = int(current_value)
	if current_argument in ("-c", "--circle"):
		do_circle_corr = True
		print("performing circle correction")
	if current_argument in ("-r", "--rot_center"):
		rot_center = float(current_value)
	if current_argument in ("-h", "--help"):
		print("Reconstructs a output.h5 file. -i --fdir_in specifies input directory, -p --no_projections specifies the total number of projections, -r --rot_center specifies rotation center, -w --width specifies image width and -e --height specifies image height")


batch_total = int((no_projections/projections_per_timepoint))
batch_current = 0

### for testing: only reconstruct x timepoint(s) ###
#batch_total = 1
####################################################

if batch_total > 1:
	print('Image reconstruction separated into ', str(batch_total), ' batches.')
else:
	print('Image reconstruction in 1 batch')

proj_start = 0
proj_end = projections_per_timepoint #500

rec_start = 0
rec_end = height #1800

# current_start = 0
# if start > 0:
	# current_start = start

# if batches > 1:
	# current_end = 500
# else:
	# current_end = end

### for testing: adjust start and end of reconstruction ###
# proj_start = 0
# proj_end = 500


#rec_start = 3600
#rec_end = 5400
###########################################################

file_creator = batch_total

while file_creator > 0:
	file_name = 'output_reconstructed_' + str(file_creator) + '.h5'
	try:
		with h5py.File(data_f + file_name, 'a') as hdf:
			dset = hdf.create_dataset('reconstructed', (height, width, width), dtype='uint16', chunks=True)
	except:
		print('Failed to create ''reconstructed'' dataset')
	file_creator -= 1


while batch_current < batch_total:
	print('Reconstructing batch ', str(batch_current + 1), ' out of ', str(batch_total))

	#proj = np.zeros((no_projections, (current_end - current_start), width))

	with h5py.File(data_f + sname, 'r') as hdf:
		d = hdf['normalized']
		proj = np.array(d[proj_start:proj_end, :, :])

	print('Loading complete, image dimensions: ', proj.shape)

	#plt.imshow(proj[:,:,0], cmap='Greys_r')
	#plt.show()
	###plt.imshow(proj[:,0,:], cmap='Greys_r')
	###plt.show()

	#proj_max = np.amax(proj)

	#proj = proj/proj_max


	#plt.imshow(proj[0, :, :], cmap='Greys_r')
	#plt.show()

	print('Generating angles')

	theta = tomopy.angles(proj.shape[0], 0, nangles)

	if (theta is None):
		theta = tomopy.angles(proj.shape[0])
	else:
		pass

##stripe removal generates errors! maybe the data should be in a different form?

	# print('Removing stripes')
	# proj = tomopy.remove_stripe_ti(proj, ncore=10)
	# plt.imshow(proj[:, 0, :], cmap='Greys_r')
	# plt.show()
	print("removing stripes")
	proj = tomopy.prep.stripe.remove_stripe_fw(proj)

	print("Reconstructing with " + str(method))
	extra_options = {}#{'MinConstraint':0}
	options = {'proj_type':'cuda', 'method':method, 'gpu_List': [1], 'extra_options':extra_options}
	recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.astra, ncore=1, options=options)

	print('recon shape: ', recon.shape)
	###plt.imshow(recon[0, :,:], cmap='Greys_r')
	###plt.show()

	if do_circle_corr == True:
		print("performing circle correction")
		proj = tomopy.remove_ring(proj,out = proj)

	print('masking reconstructed images')


	print('converting to uint16')
	#scale all values by the largest one in the whole stack and multiply it by the max value of uint16 which is 65535
	recon_max = np.amax(recon)
	recon = recon/recon_max
	recon = recon * 65535

	recon = tomopy.circ_mask(recon, axis = 0, ratio=0.90)

	print(recon.shape)

	print('writing images to disk')

	write_name = 'output_reconstructed_' + str(batch_current + 1) + '.h5'

	with h5py.File(data_f + write_name, 'a') as hdf:
		hdf['reconstructed'][0:height, :, :] = recon #[rec_start:rec_end, :, :]

	batch_current += 1

	proj_start = proj_end
	proj_end += projections_per_timepoint #500
	rec_start = rec_end
	rec_end += height #1800
