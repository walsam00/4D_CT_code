# This code takes the reconstructed CT image stacks from a specified directory, 
# bins them to size 243x243 (x,y) and saves them in a new path for further processing

import numpy as np
import h5py
import getopt, sys
import numba
from numba import jit

#get command line arguments for which tablet should be processed
#this code is called by a SLURM script so this is a useful way of populating the variables

full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]

short_options = "x:f:"
long_options = ["max_index=", "folder_name="]

try:
	arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
	print(str(err))
	sys.exit(2)

for current_argument, current_value in arguments:
	if current_argument in ("-x", "--max_index"):
		max_index = int(current_value)
		#for example "55"
	if current_argument in ("-f", "--folder_name"):
		folder = str(current_value)
		#for example "N2_w_01"
		#this string is used for both defining the folder name, as well as iterating through the files contained in that folder

#generate file paths

in_dir = '/base_directory/'
input_dir = in_dir + 'Reconstructions/' + folder
output_dir = in_dir + 'TF/' + folder

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
    #determine padding, if needed
    
	ds_factor_base = x // 243
	ds_factor_remainder = x % 243
	if ds_factor_remainder > 0:
		ds_factor_base += 1
		padded_x = ds_factor_base * 243
		ds_factor_remainder = padded_x - x
		padding_l = ds_factor_remainder // 2
		padding_r =  ds_factor_remainder - padding_l
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
    
    #create padded array
    
	data_padded = np.zeros([padded_z,padded_x,padded_x])
	print(data_padded.shape)
	print(padding_l)
	print(padding_r)
	print(padding_z_d)
	print(padding_z_u)
    
    #populate padded array 
    
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
    
    #use numba to rapidly do the downscaling using a moving average algorithm
    
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
    
    #populate the output array
    
	data_out = gofast(data, data_out, factor, z_out, y_out, x_out)
    
    #add 'color' axis
    
	data_out = data_out[:,:,:,np.newaxis]

    #save downsized file
    
	print('saving file ' + str(time_index+1) + ' of ' + str(max_index))
	filename_out = output_dir + '/' + folder + '_reconstructed_' + str(time_index + 1) + '.h5'
	with h5py.File(filename_out, 'a') as hdf2:
		dset = hdf2.create_dataset('data', data_out.shape, dtype=np.float32)
		hdf2['data'][...] = data_out[...]
