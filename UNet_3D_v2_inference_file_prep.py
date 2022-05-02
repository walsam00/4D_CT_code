import numpy as np
import h5py
import getopt, sys
import skimage.transform
#import os
import numba
from numba import jit
#get command line arguments for which tablet should be processed

full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]

short_options = "s:o:x:f:m:h"
long_options = ["input_dir=", "output_dir=", "max_index=", "folder_name=", "model_path=", "help"]

try:
	arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
	print(str(err))
	sys.exit(2)

for current_argument, current_value in arguments:
	if current_argument in ("-s", "--ssh"):
		ssh_bool = int(current_value)
		if ssh_bool == 0:
			in_dir = '/mnt/nfs-scicore/time_resolved_uCT_data_PSI/'
		else:
			in_dir = '/scicore/projects/pharmatech-scicore/time_resolved_uCT_data_PSI/'
	if current_argument in ("-o", "--output_dir"):
		out_dir = str(current_value)
	if current_argument in ("-x", "--max_index"):
		max_index = int(current_value)
	if current_argument in ("-f", "--folder_name"):
		folder = str(current_value)
	if current_argument in ("-m", "--model_path"):
		model_path = str(current_value)
	if current_argument in ("-h", "--help"):
		print("Downsizes contents of a uCT recon folder to 243x243 and saves it to a separate directory. -f OR --folder= specifies the folder name (e.g. N2_w_01) and -m OR -max_index specifies the max index of that folder")

input_dir = in_dir + 'Reconstructions/' + folder
output_dir = in_dir + 'TF/' + folder

# if not os.path.exists(output_dir):
#     try:
#         os.makedirs(output_dir)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise

print('Input directory: ' + input_dir)
print('Output directory: ' + output_dir)
print(max_index)

#load images from /Reconstructions folder, downsize them and save them in /TF folder

for time_index in range(max_index):
	print('downsizing file ' + str(time_index+1) + ' of ' + str(max_index))
	input_file = input_dir + '/output_reconstructed_' + str(time_index + 1) + '.h5'
	with h5py.File(input_file, 'r') as hdf:
		z,y,x = hdf['reconstructed'][...].shape
		print(z)
		print(y)
		print(x)
		data = np.array(hdf['reconstructed'])

	#determine slice size and downscale factor
	ds_factor_base = x // 243
	ds_factor_remainder = x % 243
	if ds_factor_remainder > 0:
		ds_factor_base += 1
		padded_x = ds_factor_base * 243
		ds_factor_remainder = padded_x - x
		padding_l = ds_factor_remainder // 2
		padding_r =  ds_factor_remainder - padding_l
		#prep padded output array
	else:
		padded_x = x
		padded_y = y
		padding_l = 0
		padding_r = 0

	ds_z_base = z // ds_factor_base
	ds_z_remainder = z % ds_factor_base
	if ds_z_remainder > 0:
		padded_z = (z//ds_factor_base + 1) * ds_factor_base
		padding_z_d = (padded_z - z) // 2
		padding_z_u = (padded_z - z) - padding_z_d
	else:
		padded_z = z
		padding_z_d = 0
		padding_z_u = 0
	z_out = padded_z // ds_factor_base
	data_padded = np.zeros([padded_z,padded_x,padded_x])
	print(data_padded.shape)
	print(padding_l)
	print(padding_r)
	print(padding_z_d)
	print(padding_z_u)
	if padding_z_d > 0 and padding_z_u > 0 and padding_l > 0 and padding_r > 0:
		data_padded[padding_z_d:-padding_z_u,padding_l:-padding_r,padding_l:-padding_r] = data
	elif padding_z_d == 0 and padding_z_u == 0 and padding_l > 0 and padding_r > 0:
		data_padded[:,padding_l:-padding_r,padding_l:-padding_r] = data
	elif padding_z_d > 0 and padding_z_u > 0 and padding_l == 0 and padding_r == 0:
		data_padded[padding_z_d:-padding_z_u,:,:] = data
	elif padding_z_d == 0 and padding_z_u > 0 and padding_l == 0 and padding_r > 0:
		data_padded[padding_z_d:-padding_z_u,padding_l:-padding_r,padding_l:-padding_r] = data
	elif paddin_z_d == 0 and padding_z_u == 0 and padding_l == 0 and padding_r > 0:
		data_padded[:,padding_l:-padding_r,padding_l:-padding_r] = data
	elif padding_z_d == 0 and padding_z_u > 0 and padding_l == 0 and padding_r == 0:
		data_padded[padding_z_d:-padding_z_u,:,:] = data
	z_in,y_in,x_in = data_padded.shape
	data_padded = data_padded.reshape(-1)
	factor = ds_factor_base
	z_out = z_in // factor
	y_out = y_in // factor
	x_out = x_in // factor
	data_out = np.empty([z_out,y_out,x_out])
	@numba.jit(nopython=True)
	def gofast(data, data_out, factor, z_out, y_out, x_out):
		for z in range(z_out):
		    for y in range(y_out):
		        for x in range(x_out):

		            sum = 0
		            for z_inner in range(factor):
		                for y_inner in range(factor):
		                    for x_inner in range(factor):
		                        x_kernel = x*factor+x_inner
		                        y_kernel = y*factor+y_inner
		                        z_kernel = z*factor+z_inner
		                        pos = ((z_kernel*y_in)*x_in) + (y_kernel*x_in) + x_kernel
		                        sum += data_padded[pos]
		            data_out[z,y,x] = sum/(factor*factor*factor)
		return(data_out)
	data_out = gofast(data, data_out, factor, z_out, y_out, x_out)
    #add 'color' axis
	data_out = data_out[:,:,:,np.newaxis]

    #save downsized file
	print('saving file ' + str(time_index+1) + ' of ' + str(max_index))
	filename_out = output_dir + '/' + folder + '_reconstructed_' + str(time_index + 1) + '.h5'
	with h5py.File(filename_out, 'a') as hdf2:
		dset = hdf2.create_dataset('data', data_out.shape, dtype=np.float32)
		hdf2['data'][...] = data_out[...]
