#This code extracts a voxel count from each tablet at each timepoint, from which a disintegration rate constant can be approximated.
#This code can serve as a template for how to loop through the data

import numpy as np
import h5py
import getopt, sys
import csv

#Genereate dictionary with number of timepoints for each tablet dataset
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

#Generate list with all tablet dataset names
folder_list = ['N2_w_01','N3_w_01','N4_w_01','N5_w_02','N6_w_01','N9_w_01','N10_w_01','N11_w_01','N12_w_01','N13_w_01', 'N14_w_01', 'N15_w_01','N16_w_01','N17_w_01','N18_w_01','N27_w_01','N28_w_02','N32_w_01','N49_w_01','N50_w_01','N21_w_01','N22_w_01','N23_w_01','N24_w_01','N29_w_01','N30_w_01','N31_w_01', 'N37_w_01','N38_w_01','N39_w_01','N40_w_01','N45_w_01','N46_w_01','N48_w_02','N53_w_02','N54_w_01','N55_w_01','N56_w_01','N61_w_01','N62_w_01','N63_w_01','N64_w_01','N19_w_02','N20_w_01','N25_w_01','N26_w_01','N33_w_01','N34_w_01','N35_w_01','N36_w_01','N41_w_01','N42_w_01','N43_w_01','N44_w_01','N51_w_01','N52_w_01','N57_w_01','N58_w_01','N59_w_01','N60_w_01','N1_1ms_500prj_WB50_w_01', 'N7_w_03', 'N8_1ms_500prj_WB50_w_01', 'N47_w_01']

#iterate through the data, folder by folder
in_dir = 'FILEPATH_TO_DATASET_FOLDERS/' #needs to be changed to the actual file path
for folder in folder_list:
    value_counter_1 = []
    value_counter_2 = []
    current_value_1 = 0
    current_value_2 = 0
    timepoint_iterator_max = name_timepoint_dict[folder]
    timepoint_iterator = 2 
    #load the first timepoint's segmented mask
    mask_filename = in_dir + folder + '/' + folder + '_segmented_slim_unet_' + '1' + '.h5'    
    
    #deal with the fact that some folders were missnamed (legacy)
    try:
        with h5py.File(mask_filename, 'r') as hdf:
            image_mask_initial = np.array(hdf['data'][:,:,:,:])
    except:
        mask_filename = in_dir + folder + '/' + folder + '_segmented_slim_unet' + str(timepoint_iterator) + '.h5'
        print(folder + ' needs to be renamed! (underscore missing in segmented file name)')
        with h5py.File(mask_filename, 'r') as hdf:
            image_mask_initial = np.array(hdf['data'][:,:,:,:])
    #Populate counters for segmentation class 2 (organic tablet component) and class 3 (inorganic tablet component) -> class 3 only exists in half of the data, will be always zero for the other half
    current_value_1 = np.count_nonzero(image_mask_initial == 2)
    value_counter_1.append(str(current_value_1))
    current_value_2 = np.count_nonzero(image_mask_initial == 3)
    value_counter_2.append(str(current_value_2))
    #iterate through all timepoints, populate counters
    while timepoint_iterator <= timepoint_iterator_max:
        mask_filename = in_dir + folder + '/' + folder + '_segmented_slim_unet_' + str(timepoint_iterator) + '.h5'
        print('working on ' + folder + ' timepoint ' + str(timepoint_iterator) + ' of ' + str(timepoint_iterator_max))
        #deal with the fact that some folders were missnamed (legacy)
        try:
            with h5py.File(mask_filename, 'r') as hdf:
                image_mask = np.array(hdf['data'][:,:,:,:])
        except:
            mask_filename = in_dir + folder + '/' + folder + '_segmented_slim_unet' + str(timepoint_iterator) + '.h5'
            print(folder + ' needs to be renamed! (underscore missing in segmented file name)')
            with h5py.File(mask_filename, 'r') as hdf:
                image_mask = np.array(hdf['data'][:,:,:,:])
        #append counters
        current_value_1 = np.count_nonzero(image_mask == 2)
        value_counter_1.append(str(current_value_1))
        current_value_2 = np.count_nonzero(image_mask == 3)
        value_counter_2.append(str(current_value_2))

        timepoint_iterator += 1
        
    #print some progress information 
    print(folder)
    print(value_counter_1)
    print(folder)
    print(value_counter_2)
    
    #get counters ready to be printed to file
    list_to_file = []
    list_to_file.append(folder)
    list_to_file.append('tablet_organic')
    list_to_file += value_counter_1
    list_to_file.append('tablet_inorganic')
    list_to_file += value_counter_2
    
    #write counters to file
    with open('Voxel_counter_whole_output.txt', 'a') as f:
        f.writelines('\n'.join(list_to_file))
        f.write('\n')
