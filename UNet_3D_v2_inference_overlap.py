#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import h5py
import getopt, sys
#import os

#get command line arguments for which tablet should be processed
model_path = '/scicore/projects/pharmatech-scicore/time_resolved_uCT_data_PSI/TF/UNet_3D_inference/saved_model_larger_kernel_100_epochs'
print('starting')
full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]

short_options = "s:o:x:f:m:h"
long_options = ["ssh=", "output_dir=", "max_index=", "folder_name=", "model_path=", "help"]

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

input_dir = in_dir + 'TF/' + folder
output_dir = in_dir + 'TF/' + folder

# if not os.path.exists(output_dir):
#     try:
#         os.makedirs(output_dir)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise

print('Input directory: ' + input_dir)
print('Output directory: ' + output_dir)

#load UNet3D_new_data model, run inference on downsized images batched by timepoint, save masks in same /TF folder

model = tf.keras.models.load_model(model_path)
#print(model.count_params())

for time_index in range(max_index):
    print('Segmenting file ' + str(time_index + 1) + ' of ' + str(max_index))
    input_file = output_dir + '/' + folder + '_reconstructed_' + str(time_index + 1) + '.h5'
    ###################input_file = output_dir + '/' + str(time_index + 1) + '.h5'

    with h5py.File(input_file, 'r') as hdf:
        z,x,y,c = hdf['data'][...].shape
        data = np.array(hdf['data'])

    #batch across z axis

    batches = ((z - 81) // 71) + 1

    if ((z-81) % 71) > 0:
        batches += 1

    #pad the data to make it divisible by 81
    data_padded = np.zeros([1,((batches-1)*71+81),243,243,1])
    data_padded[0,0:z,:,:,:] = data[:,:,:,:]


    #segment batch-wises
    mask_stack = np.empty([(batches*81),243,243,1])
    rec_stack = np.empty([1,81,243,243,1])

    for batch_iterator in range(batches):
        print('Subset ' + str(batch_iterator + 1) + ' of ' + str(batches))
        start = 71 * batch_iterator
        stop = start + 81
        rec_stack = data_padded[:,start:stop,:,:,:]
        #print(rec_stack.shape)
        rec_stack_tf = tf.convert_to_tensor(rec_stack, dtype=tf.float32)
        rec_stack_tf = tf.image.per_image_standardization(rec_stack_tf)
        #print(rec_stack_tf)
        #print(rec_stack_tf.shape)
        mask_batch_pred = model.predict(rec_stack_tf)
        mask_batch_tf = mask_batch_pred[0]      ###???
        mask_batch_tf = tf.argmax(mask_batch_tf, axis=-1)
        mask_batch_tf = mask_batch_tf[...,tf.newaxis]
        mask_batch = np.array(mask_batch_tf)
        #start = batch_iterator * 81
        #stop = start + 81
        if start == 0 and stop != 0:
            mask_start = start
            mask_stop = stop - 5
            mask_stack[mask_start:mask_stop,:,:,:] = mask_batch[:-5,:,:,:]
        elif start != 0 and stop >= z:
            mask_start = start + 5
            mask_stop = stop
            mask_stack[mask_start:mask_stop,:,:,:] = mask_batch[5:,:,:,:]
        elif start == 0 and stop >= z:
            mask_start = start
            mask_stop = z
            mask_stack[mask_start:mask_stop,:,:,:] = mask_batch[:,:,:,:]
        elif start != 0 and stop < z:
            mask_start = start + 5
            mask_stop = stop - 5
            mask_stack[mask_start:mask_stop,:,:,:] = mask_batch[5:-5,:,:,:]

        #mask_stack[mask_start:mask_stop,:,:,:] = mask_batch[:,:,:,:]
    data_out = mask_stack[0:z,:,:,:]
    filename_out = output_dir + '/' + folder + '_segmented_larger_kernel_100_overlap_' + str(time_index + 1) + '.h5'
    ####################filename_out = output_dir + '/' + 'segmented_' + str(time_index + 1) + '.h5'
    with h5py.File(filename_out, 'a') as hdf:
        dset = hdf.create_dataset('data', data_out.shape, dtype=np.float32)
        hdf['data'][...] = data_out[...]
