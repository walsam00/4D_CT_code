#This code generates real-time videos depticting the volumetric, time-resolved reconstructed microCT data as an assembly of cross sections through the data. The videos show just tablet voxels, the background is removed.
#This code can serve as a basis for looping through the data and dealing with the differing acquisition framerates

import numpy as np
import h5py
import getopt, sys
import imageio as iio

#dictionary with formulation name and number of timepoints
name_timepoint_dict = {
'N10_w_01':55,
'N1_1ms_500prj_WB50_w_01':120,
'N11_w_01':55,
'N12_w_01':55,
'N13_w_01':55,
'N14_w_01':55,
'N15_w_01':29,
'N16_w_01':15,
'N17_w_01':55,
'N18_w_01':55,
'N19_w_02':35,
'N20_w_01':35,
'N21_w_01':15,
'N22_w_01':10,
'N23_w_01':13,
'N24_w_01':11,
'N25_w_01':35,
'N26_w_01':35,
'N27_w_01':35,
'N28_w_02':55,
'N29_w_01':11,
'N2_w_01':55,
'N30_w_01':8,
'N31_w_01':55,
'N33_w_01':35,
'N34_w_01':35,
'N32_w_01':10,
'N35_w_01':35,
'N36_w_01':35,
'N37_w_01':52,
'N38_w_01':33,
'N39_w_01':16,
'N3_w_01':55,
'N40_w_01':24,
'N41_w_01':35,
'N42_w_01':35,
'N43_w_01':35,
'N44_w_01':35,
'N45_w_01':35,
'N46_w_01':22,
'N47_w_01':71,
'N49_w_01':55,
'N48_w_02':55,
'N4_w_01':55,
'N51_w_01':35,
'N52_w_01':35,
'N50_w_01':55,
'N53_w_02':13,
'N54_w_01':13,
'N55_w_01':25,
'N56_w_01':22,
'N57_w_01':35,
'N58_w_01':35,
'N59_w_01':35,
'N5_w_02':31,
'N60_w_01':35,
'N61_w_01':13,
'N62_w_01':11,
'N63_w_01':26,
'N64_w_01':16,
'N6_w_01':29,
'N7_w_03':9,
'N8_1ms_500prj_WB50_w_01':60,
'N9_w_01':55
}
#create lists with the very common acquisition frequency sequences 1;30(1Hz);24(0.2Hz) and 1;10(1Hz);24(0.2Hz) -> the rest will be called on individually
one_thirty_twentyfour = ['N2_w_01','N3_w_01','N4_w_01','N5_w_02','N6_w_01','N9_w_01','N10_w_01','N11_w_01','N12_w_01','N13_w_01', 'N14_w_01', 'N15_w_01','N16_w_01','N17_w_01','N18_w_01','N27_w_01','N28_w_02','N32_w_01','N49_w_01','N50_w_01', 'N21_w_01','N22_w_01','N23_w_01','N24_w_01','N29_w_01','N30_w_01','N31_w_01', 'N37_w_01','N38_w_01','N39_w_01','N40_w_01','N45_w_01','N46_w_01','N48_w_02','N53_w_02','N54_w_01','N55_w_01','N56_w_01','N61_w_01','N62_w_01','N63_w_01','N64_w_01']

one_ten_twentyfour = ['N19_w_02','N20_w_01','N25_w_01','N26_w_01','N33_w_01','N34_w_01','N35_w_01','N36_w_01','N41_w_01','N42_w_01','N43_w_01','N44_w_01','N51_w_01','N52_w_01','N57_w_01','N58_w_01','N59_w_01','N60_w_01',]

#goal:
#load image, load mask for that images
#adjust mask values: 0 stays 0, 1 becomes 0, 1 stays 1, 2 becomes 1
#multiply image matrix by adjusted masks
#duplicate resulting images in sequence in a way that matches their acquisition frequency
#save result as .mp4

input_dir = '/FILE_PATH_TO_DATA_FOLDERS/'
image_data = np.zeros([200,240,1])
image_mask = np.zeros([200,240,1])
image_data_masked = np.zeros([200,240,1])

for folder in one_thirty_twentyfour:
    output_filename = input_dir + 'visualization/fancy_mp4/' + folder + '.mp4'
    timepoint_iterator_max = name_timepoint_dict[folder]
    timepoint_iterator = 1
    if timepoint_iterator_max > 31:
        number_of_images_out = 31 + (timepoint_iterator_max - 31) * 5
        image_stack_out = np.zeros([number_of_images_out,440,440,1])
    else:
        image_stack_out = np.zeros([timepoint_iterator_max,440,440,1])

    while timepoint_iterator <= timepoint_iterator_max:
        image_filename = input_dir + folder + '/' + folder + '_reconstructed_binned_' + str(timepoint_iterator) + '.h5'
        mask_filename = input_dir + folder + '/' + folder + '_segmented_slim_unet_' + str(timepoint_iterator) + '.h5'
        print('working on ' + folder + ' timepoint ' + str(timepoint_iterator) + ' of ' + str(timepoint_iterator_max))
        #load just cross sections through the center point
        with h5py.File(image_filename, 'r') as hdf:
            image_data_1 = np.array(hdf['data'][:,119,:,:])
            image_data_2 = np.array(hdf['data'][:,:,119,:])
            image_data_3 = np.array(hdf['data'][99,:,:,:])
        #load the corresponding image mask (segmented images)
        # deal with the fact that some folders were missnamed (legacy)
        try:
            with h5py.File(mask_filename, 'r') as hdf:
                image_mask_1 = np.array(hdf['data'][:,119,:,:])
                image_mask_2 = np.array(hdf['data'][:,:,119,:])
                image_mask_3 = np.array(hdf['data'][99,:,:,:])
        except:
            mask_filename = input_dir + folder + '/' + folder + '_segmented_slim_unet' + str(timepoint_iterator) + '.h5'
            print(folder + ' needs to be renamed! (underscore missing in segmented file name)')
            with h5py.File(mask_filename, 'r') as hdf:
                image_mask_1 = np.array(hdf['data'][:,119,:,:])
                image_mask_2 = np.array(hdf['data'][:,:,119,:])
                image_mask_3 = np.array(hdf['data'][99,:,:,:])
        
        image_mask_1[image_mask_1 == 1] = 0
        image_mask_1[image_mask_1 == 2] = 1
        image_mask_1[image_mask_1 == 3] = 2
        image_mask_2[image_mask_2 == 1] = 0
        image_mask_2[image_mask_2 == 2] = 1
        image_mask_2[image_mask_2 == 3] = 2
        image_mask_3[image_mask_3 == 1] = 0
        image_mask_3[image_mask_3 == 2] = 1
        image_mask_3[image_mask_3 == 3] = 2
        image_data_masked_1 = np.multiply(image_data_1, image_mask_1)
        image_data_masked_1 = np.flipud(image_data_masked_1)
        image_data_masked_2 = np.multiply(image_data_2, image_mask_2)
        image_data_masked_2 = np.flipud(image_data_masked_2)
        image_data_masked_3 = np.multiply(image_data_3, image_mask_3)
        image_data_masked_3 = np.flipud(image_data_masked_3)
        if timepoint_iterator <= 31:
            image_stack_out[(timepoint_iterator-1),240:,0:240,:] = image_data_masked_1[:,:,:]
            image_stack_out[(timepoint_iterator-1),0:240,240:,:] = np.rot90(image_data_masked_2[:,:,:])
            image_stack_out[(timepoint_iterator-1),0:240,0:240,:] = image_data_masked_3[:,:,:]
        else:
            start = 31 + (timepoint_iterator - 32) * 5
            stop = start + 5
            image_stack_out[start:stop,240:,0:240,:] = image_data_masked_1[:,:,:]
            image_stack_out[start:stop,0:240,240:,:] = np.rot90(image_data_masked_2[:,:,:])
            image_stack_out[start:stop,0:240,0:240,:] = image_data_masked_3[:,:,:]
        timepoint_iterator += 1
    iio.mimsave(output_filename, image_stack_out, fps=1)

for folder in one_ten_twentyfour:
    output_filename = input_dir + 'visualization/fancy_mp4/' + folder + '.mp4'
    timepoint_iterator_max = name_timepoint_dict[folder]
    timepoint_iterator = 1
    if timepoint_iterator_max > 11:
        number_of_images_out = 11 + (timepoint_iterator_max - 11) * 5
        image_stack_out = np.zeros([number_of_images_out,440,440,1])
    else:
        image_stack_out = np.zeros([timepoint_iterator_max,440,440,1])

    while timepoint_iterator <= timepoint_iterator_max:
        image_filename = input_dir + folder + '/' + folder + '_reconstructed_binned_' + str(timepoint_iterator) + '.h5'
        mask_filename = input_dir + folder + '/' + folder + '_segmented_slim_unet_' + str(timepoint_iterator) + '.h5'
        print('working on ' + folder + ' timepoint ' + str(timepoint_iterator) + ' of ' + str(timepoint_iterator_max))
        with h5py.File(image_filename, 'r') as hdf:
            image_data_1 = np.array(hdf['data'][:,119,:,:])
            image_data_2 = np.array(hdf['data'][:,:,119,:])
            image_data_3 = np.array(hdf['data'][99,:,:,:])

        try:
            with h5py.File(mask_filename, 'r') as hdf:
                image_mask_1 = np.array(hdf['data'][:,119,:,:])
                image_mask_2 = np.array(hdf['data'][:,:,119,:])
                image_mask_3 = np.array(hdf['data'][99,:,:,:])
        except:
            mask_filename = input_dir + folder + '/' + folder + '_segmented_slim_unet' + str(timepoint_iterator) + '.h5'
            print(folder + ' needs to be renamed! (underscore missing in segmented file name)')
            with h5py.File(mask_filename, 'r') as hdf:
                image_mask_1 = np.array(hdf['data'][:,119,:,:])
                image_mask_2 = np.array(hdf['data'][:,:,119,:])
                image_mask_3 = np.array(hdf['data'][99,:,:,:])

        image_mask_1[image_mask_1 == 1] = 0
        image_mask_1[image_mask_1 == 2] = 1
        image_mask_1[image_mask_1 == 3] = 2
        image_mask_2[image_mask_2 == 1] = 0
        image_mask_2[image_mask_2 == 2] = 1
        image_mask_2[image_mask_2 == 3] = 2
        image_mask_3[image_mask_3 == 1] = 0
        image_mask_3[image_mask_3 == 2] = 1
        image_mask_3[image_mask_3 == 3] = 2
        image_data_masked_1 = np.multiply(image_data_1, image_mask_1)
        image_data_masked_1 = np.flipud(image_data_masked_1)
        image_data_masked_2 = np.multiply(image_data_2, image_mask_2)
        image_data_masked_2 = np.flipud(image_data_masked_2)
        image_data_masked_3 = np.multiply(image_data_3, image_mask_3)
        image_data_masked_3 = np.flipud(image_data_masked_3)
        if timepoint_iterator <= 11:
            image_stack_out[(timepoint_iterator-1),240:,0:240,:] = image_data_masked_1[:,:,:]
            image_stack_out[(timepoint_iterator-1),0:240,240:,:] = np.rot90(image_data_masked_2[:,:,:])
            image_stack_out[(timepoint_iterator-1),0:240,0:240,:] = image_data_masked_3[:,:,:]
            #for unmasked image:
            #image_stack_out[(timepoint_iterator-1),240:,0:240,:] = image_data_1[:,:,:]
            #image_stack_out[(timepoint_iterator-1),0:240,240:,:] = np.rot90(image_data_2[:,:,:])
            #image_stack_out[(timepoint_iterator-1),0:240,0:240,:] = image_data_3[:,:,:]
        else:
            start = 11 + (timepoint_iterator - 12) * 5
            stop = start + 5
            image_stack_out[start:stop,240:,0:240,:] = image_data_masked_1[:,:,:]
            image_stack_out[start:stop,0:240,240:,:] = np.rot90(image_data_masked_2[:,:,:])
            image_stack_out[start:stop,0:240,0:240,:] = image_data_masked_3[:,:,:]
            #for unmasked image:
            #image_stack_out[start:stop,240:,0:240,:] = image_data_1[:,:,:]
            #image_stack_out[start:stop,0:240,240:,:] = np.rot90(image_data_2[:,:,:])
            #image_stack_out[start:stop,0:240,0:240,:] = image_data_3[:,:,:]
        timepoint_iterator += 1
    iio.mimsave(output_filename, image_stack_out, fps=1)

#exception 1: N1 -> 0.2Hz for 120 timepoints
output_filename = input_dir + 'visualization/fancy_mp4/' + 'N1_1ms_500prj_WB50_w_01' + '.mp4'
timepoint_iterator_max = 120
timepoint_iterator = 1
image_stack_out = np.zeros([120,440,440,1])

while timepoint_iterator <= timepoint_iterator_max:
    image_filename = input_dir + 'N1_1ms_500prj_WB50_w_01' + '/' + 'N1_1ms_500prj_WB50_w_01' + '_reconstructed_binned_' + str(timepoint_iterator) + '.h5'
    mask_filename = input_dir + 'N1_1ms_500prj_WB50_w_01' + '/' + 'N1_1ms_500prj_WB50_w_01' + '_segmented_slim_unet_' + str(timepoint_iterator) + '.h5'
    with h5py.File(image_filename, 'r') as hdf:
        image_data_1 = np.array(hdf['data'][:,119,:,:])
        image_data_2 = np.array(hdf['data'][:,:,119,:])
        image_data_3 = np.array(hdf['data'][99,:,:,:])
    with h5py.File(mask_filename, 'r') as hdf:
        image_mask_1 = np.array(hdf['data'][:,119,:,:])
        image_mask_2 = np.array(hdf['data'][:,:,119,:])
        image_mask_3 = np.array(hdf['data'][99,:,:,:])
    image_mask_1[image_mask_1 == 1] = 0
    image_mask_1[image_mask_1 == 2] = 1
    image_mask_1[image_mask_1 == 3] = 2
    image_mask_2[image_mask_2 == 1] = 0
    image_mask_2[image_mask_2 == 2] = 1
    image_mask_2[image_mask_2 == 3] = 2
    image_mask_3[image_mask_3 == 1] = 0
    image_mask_3[image_mask_3 == 2] = 1
    image_mask_3[image_mask_3 == 3] = 2
    image_data_masked_1 = np.multiply(image_data_1, image_mask_1)
    image_data_masked_1 = np.flipud(image_data_masked_1)
    image_data_masked_2 = np.multiply(image_data_2, image_mask_2)
    image_data_masked_2 = np.flipud(image_data_masked_2)
    image_data_masked_3 = np.multiply(image_data_3, image_mask_3)
    image_data_masked_3 = np.flipud(image_data_masked_3)
    image_stack_out[(timepoint_iterator-1),240:,0:240,:] = image_data_masked_1[:,:,:]
    image_stack_out[(timepoint_iterator-1),0:240,240:,:] = np.rot90(image_data_masked_2[:,:,:])
    image_stack_out[(timepoint_iterator-1),0:240,0:240,:] = image_data_masked_3[:,:,:]
    timepoint_iterator += 1
iio.mimsave(output_filename, image_stack_out, fps=0.2)

#excepton 2: N7 -> 0.02 Hz for 8 time points
output_filename = input_dir + 'visualization/fancy_mp4/' + 'N7_w_03' + '.mp4'
timepoint_iterator_max = 9
timepoint_iterator = 1
image_stack_out = np.zeros([60,440,440,1])

while timepoint_iterator <= timepoint_iterator_max:
    image_filename = input_dir + 'N7_w_03' + '/' + 'N7_w_03' + '_reconstructed_binned_' + str(timepoint_iterator) + '.h5'
    mask_filename = input_dir + 'N7_w_03' + '/' + 'N7_w_03' + '_segmented_slim_unet_' + str(timepoint_iterator) + '.h5'
    with h5py.File(image_filename, 'r') as hdf:
        image_data_1 = np.array(hdf['data'][:,119,:,:])
        image_data_2 = np.array(hdf['data'][:,:,119,:])
        image_data_3 = np.array(hdf['data'][99,:,:,:])
    with h5py.File(mask_filename, 'r') as hdf:
        image_mask_1 = np.array(hdf['data'][:,119,:,:])
        image_mask_2 = np.array(hdf['data'][:,:,119,:])
        image_mask_3 = np.array(hdf['data'][99,:,:,:])
    image_mask_1[image_mask_1 == 1] = 0
    image_mask_1[image_mask_1 == 2] = 1
    image_mask_1[image_mask_1 == 3] = 2
    image_mask_2[image_mask_2 == 1] = 0
    image_mask_2[image_mask_2 == 2] = 1
    image_mask_2[image_mask_2 == 3] = 2
    image_mask_3[image_mask_3 == 1] = 0
    image_mask_3[image_mask_3 == 2] = 1
    image_mask_3[image_mask_3 == 3] = 2
    image_data_masked_1 = np.multiply(image_data_1, image_mask_1)
    image_data_masked_1 = np.flipud(image_data_masked_1)
    image_data_masked_2 = np.multiply(image_data_2, image_mask_2)
    image_data_masked_2 = np.flipud(image_data_masked_2)
    image_data_masked_3 = np.multiply(image_data_3, image_mask_3)
    image_data_masked_3 = np.flipud(image_data_masked_3)
    image_stack_out[(timepoint_iterator-1),240:,0:240,:] = image_data_masked_1[:,:,:]
    image_stack_out[(timepoint_iterator-1),0:240,240:,:] = np.rot90(image_data_masked_2[:,:,:])
    image_stack_out[(timepoint_iterator-1),0:240,0:240,:] = image_data_masked_3[:,:,:]
    timepoint_iterator += 1
iio.mimsave(output_filename, image_stack_out, fps=1) #slow down 5000% in adobe premiere

#exception 3: N8 -> 2 Hz for 60 time points continuous -> flipping has to be done
output_filename = input_dir + 'visualization/fancy_mp4/' + 'N8_1ms_500prj_WB50_w_01' + '.mp4'
timepoint_iterator_max = 60
timepoint_iterator = 1
image_stack_out = np.zeros([60,440,440,1])

while timepoint_iterator <= timepoint_iterator_max:
    image_filename = input_dir + 'N8_1ms_500prj_WB50_w_01' + '/' + 'N8_1ms_500prj_WB50_w_01' + '_reconstructed_binned_' + str(timepoint_iterator) + '.h5'
    mask_filename = input_dir + 'N8_1ms_500prj_WB50_w_01' + '/' + 'N8_1ms_500prj_WB50_w_01' + '_segmented_slim_unet_' + str(timepoint_iterator) + '.h5'
    with h5py.File(image_filename, 'r') as hdf:
        image_data_1 = np.array(hdf['data'][:,119,:,:])
        image_data_2 = np.array(hdf['data'][:,:,119,:])
        image_data_3 = np.array(hdf['data'][99,:,:,:])
    with h5py.File(mask_filename, 'r') as hdf:
        image_mask_1 = np.array(hdf['data'][:,119,:,:])
        image_mask_2 = np.array(hdf['data'][:,:,119,:])
        image_mask_3 = np.array(hdf['data'][99,:,:,:])
    image_mask_1[image_mask_1 == 1] = 0
    image_mask_1[image_mask_1 == 2] = 1
    image_mask_1[image_mask_1 == 3] = 2
    image_mask_2[image_mask_2 == 1] = 0
    image_mask_2[image_mask_2 == 2] = 1
    image_mask_2[image_mask_2 == 3] = 2
    image_mask_3[image_mask_3 == 1] = 0
    image_mask_3[image_mask_3 == 2] = 1
    image_mask_3[image_mask_3 == 3] = 2
    image_data_masked_1 = np.multiply(image_data_1, image_mask_1)
    image_data_masked_1 = np.flipud(image_data_masked_1)
    image_data_masked_2 = np.multiply(image_data_2, image_mask_2)
    image_data_masked_2 = np.flipud(image_data_masked_2)
    image_data_masked_3 = np.multiply(image_data_3, image_mask_3)
    image_data_masked_3 = np.flipud(image_data_masked_3)
    if timepoint_iterator % 2 == 0:
        image_data_masked_1 = np.fliplr(image_data_masked_1)
        image_data_masked_1[:,:-1,:] = image_data_masked_1[:,1:,:]
        image_data_masked_2 = np.fliplr(image_data_masked_2)
        image_data_masked_2[:,:-1,:] = image_data_masked_2[:,1:,:]
        image_data_masked_3 = np.fliplr(image_data_masked_3)
        image_data_masked_3 = np.flipud(image_data_masked_3)
        image_data_masked_3[1:,:-1,:] = image_data_masked_3[:-1,1:,:]
    image_stack_out[(timepoint_iterator-1),240:,0:240,:] = image_data_masked_1[:,:,:]
    image_stack_out[(timepoint_iterator-1),0:240,240:,:] = np.rot90(image_data_masked_2[:,:,:])
    image_stack_out[(timepoint_iterator-1),0:240,0:240,:] = image_data_masked_3[:,:,:]
    timepoint_iterator += 1
iio.mimsave(output_filename, image_stack_out, fps=2)

#exception 4: N47 -> 1,30(1Hz),40(0.2Hz)
output_filename = input_dir + 'visualization/fancy_mp4/' + 'N47_w_01' + '.mp4'
timepoint_iterator_max = 71
timepoint_iterator = 1
image_stack_out = np.zeros([231,440,440,1])

while timepoint_iterator <= timepoint_iterator_max:
    image_filename = input_dir + 'N47_w_01' + '/' + 'N47_w_01' + '_reconstructed_binned_' + str(timepoint_iterator) + '.h5'
    mask_filename = input_dir + 'N47_w_01' + '/' + 'N47_w_01' + '_segmented_slim_unet' + str(timepoint_iterator) + '.h5'
    with h5py.File(image_filename, 'r') as hdf:
        image_data_1 = np.array(hdf['data'][:,119,:,:])
        image_data_2 = np.array(hdf['data'][:,:,119,:])
        image_data_3 = np.array(hdf['data'][99,:,:,:])
    with h5py.File(mask_filename, 'r') as hdf:
        image_mask_1 = np.array(hdf['data'][:,119,:,:])
        image_mask_2 = np.array(hdf['data'][:,:,119,:])
        image_mask_3 = np.array(hdf['data'][99,:,:,:])
    image_mask_1[image_mask_1 == 1] = 0
    image_mask_1[image_mask_1 == 2] = 1
    image_mask_1[image_mask_1 == 3] = 2
    image_mask_2[image_mask_2 == 1] = 0
    image_mask_2[image_mask_2 == 2] = 1
    image_mask_2[image_mask_2 == 3] = 2
    image_mask_3[image_mask_3 == 1] = 0
    image_mask_3[image_mask_3 == 2] = 1
    image_mask_3[image_mask_3 == 3] = 2
    image_data_masked_1 = np.multiply(image_data_1, image_mask_1)
    image_data_masked_1 = np.flipud(image_data_masked_1)
    image_data_masked_2 = np.multiply(image_data_2, image_mask_2)
    image_data_masked_2 = np.flipud(image_data_masked_2)
    image_data_masked_3 = np.multiply(image_data_3, image_mask_3)
    image_data_masked_3 = np.flipud(image_data_masked_3)
    if timepoint_iterator <= 31:
        image_stack_out[(timepoint_iterator-1),240:,0:240,:] = image_data_masked_1[:,:,:]
        image_stack_out[(timepoint_iterator-1),0:240,240:,:] = np.rot90(image_data_masked_2[:,:,:])
        image_stack_out[(timepoint_iterator-1),0:240,0:240,:] = image_data_masked_3[:,:,:]
    else:
        start = 31 + (timepoint_iterator - 32) * 5
        stop = start + 5
        image_stack_out[start:stop,240:,0:240,:] = image_data_masked_1[:,:,:]
        image_stack_out[start:stop,0:240,240:,:] = np.rot90(image_data_masked_2[:,:,:])
        image_stack_out[start:stop,0:240,0:240,:] = image_data_masked_3[:,:,:]
    timepoint_iterator += 1
iio.mimsave(output_filename, image_stack_out, fps=1)
