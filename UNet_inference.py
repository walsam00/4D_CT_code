#This code takes binned CT image series from a specified directory, and segments the image data 
#by running prediction using a specified tensorflow model.
#This code is run in a singularity container from a SLURM-script

#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import h5py
import getopt, sys

#get command line arguments for which tablet should be processed
full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]

short_options = "x:f:m:"
long_options = ["max_index=", "folder_name=", "model_path="]

try:
	arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
	print(str(err))
	sys.exit(2)

for current_argument, current_value in arguments:
	if current_argument in ("-x", "--max_index"):
		max_index = int(current_value)
	if current_argument in ("-f", "--folder_name"):
		folder = str(current_value)
	if current_argument in ("-m", "--model_path"):
		model_path = str(current_value)

#generate file paths
in_dir = '/base_file_directory/'
input_dir = in_dir + 'TF/' + folder
output_dir = in_dir + 'TF/' + folder

print('Input directory: ' + input_dir)
print('Output directory: ' + output_dir)

#load tensorflow model, run inference on downsized images batched by timepoint, save masks in same /TF folder
model = tf.keras.models.load_model(model_path)

for time_index in range(max_index):
	print('Segmenting file ' + str(time_index + 1) + ' of ' + str(max_index))
	input_file = output_dir + '/' + folder + '_reconstructed_' + str(time_index + 1) + '.h5'

	with h5py.File(input_file, 'r') as hdf:
		z,x,y,c = hdf['data'][...].shape
		data = np.array(hdf['data'])
    
    #set up output array
    #initially, the U-Net required images of size 243x243 (x,y). The final model however used image stacks of dimensions 240x240, so the images are cropped
	recon_data_out = np.empty([z,240,240,1])
	recon_data_out = data[:,2:-1,2:-1,:]
    
	#batch across z axis
    #do the prediction using an overlap which eliminates the effects of padding at the image borders
	overlap = 2	#overlap on each side of the image stack along the z axis
	if z < 16:
		batches = 1
	elif z < 16+(16-(2*overlap)):
		batches = 2
	else:
		batches = (z-(2*(16-overlap))) // (16-(2*overlap))
		if ((z-16) % 12) > 0:
			batches += 3
		else:
			batches += 2

    #pad the data to make the sliding window fit
	data_padded = np.zeros([1,((batches-2)*(16-2*overlap)+(2*(16-overlap))),240,240,1])
	data_padded[0,0:z,:,:,:] = data[:,2:-1,2:-1,:]

    #set up mask and cropped image arrays
	mask_stack = np.empty([((batches-2)*(16-2*overlap)+(2*(16-overlap))),240,240,1])
	rec_stack = np.empty([1,16,240,240,1])
    
	#segment batch-wise
	for batch_iterator in range(batches):
		print('Subset ' + str(batch_iterator + 1) + ' of ' + str(batches))
		start = (16-(2*overlap)) * batch_iterator
		stop = start + 16
		rec_stack = data_padded[:,start:stop,:,:,:]
        
        #do the actual prediction
		rec_stack_tf = tf.convert_to_tensor(rec_stack, dtype=tf.float32)
		rec_stack_tf = tf.image.per_image_standardization(rec_stack_tf)
		mask_batch_pred = model.predict(rec_stack_tf)
		mask_batch_tf = mask_batch_pred[0]
		mask_batch_tf = tf.argmax(mask_batch_tf, axis=-1)
		mask_batch_tf = mask_batch_tf[...,tf.newaxis]
		mask_batch = np.array(mask_batch_tf)
        
        #deal with the overlap
		if start == 0 and stop < z:
			mask_start = start
			mask_stop = stop - overlap
			mask_stack[mask_start:mask_stop,:,:,:] = mask_batch[:-overlap,:,:,:]
		elif start == 0 and stop >= z:
			mask_start = start
			mask_stop = stop
			mask_stack[mask_start:mask_stop,:,:,:] = mask_batch[:,:,:,:]
		elif start > 0 and stop >= z:
			mask_start = start + overlap
			mask_stop = stop
			mask_stack[mask_start:mask_stop,:,:,:] = mask_batch[overlap:,:,:,:]
		elif start > 0 and stop < z:
			mask_start = start + overlap
			mask_stop = stop - overlap
			mask_stack[mask_start:mask_stop,:,:,:] = mask_batch[overlap:-overlap,:,:,:]

	#save the prediction
    data_out = mask_stack[0:z,:,:,:]
	filename_out = output_dir + '/' + folder + '_segmented_slim_unet_' + str(time_index + 1) + '.h5'
	with h5py.File(filename_out, 'a') as hdf:
		dset = hdf.create_dataset('data', data_out.shape, dtype=np.float32)
		hdf['data'][...] = data_out[...]
