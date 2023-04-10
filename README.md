# 4D_CT_code
This repository exists in conjunction with a (future) publication (DOI, link) to make the code created for that work publicly available.
The code is used to reconstruct and segment a specific set of 4D micro-computed tomography raw data. For further details on the data refer to the publication.
Typical order of operations:
normalization -> paganin filter -> reconstruction -> binning -> segmentation using U-Net -> further analysis / visualization

Package versions:

python          3.6.11

astra toolbox   1.9.9.dev4

h5py            2.10.0

numba           0.53.1

numpy           1.19.1

tomopy          1.4.2

tensorflow      2.5.0

ArrayFire       3.6.4
