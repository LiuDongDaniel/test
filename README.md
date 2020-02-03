false positive nodules decrease
================================
# version: 1.0
# write time: 2019.11.22
# change time:
# change content:

introduction
------------
# false positive nodules is a main problem in the pulmonary nodules detection study.   
# this project is built for detecting the false positive nodules.
# the version of the package and operating system: ubuntu 16.04; python3.6; pytorch

the main steps
---------------
# 1. generate the cube from CT dataset: 
#    a. the code is  in the cut_cube folder which include four .np file. {dicom, mhd} is the type of the input CT. {x, y, z} is the cut direction.
#       I try different direction for generate the cube, finally, I use the "node_extraction_mhd_coord_z.py"  and "annotations.csv"to generate the true positive nodules,
#       and the size of the cube is 1.5 times(x, y diameter), 1 time z diameter. Totally, I cut 1186 true positive samples
#    b. because there are lots of false positive nodules in the candidate_V2.csv file, so I just randomly choose 6000 samples using in this programme. the code of choosing samples in the 
#       random_select_samples.py, and the function is random_select_candidate(), and then use the "node_extraction_mhd_coord_z.py" and "choosed samples.csv" to generate the false positive samples

# 2. generate the motion history image: using the mhi_algorithm.py to generate the mhi image for true positive samples and false positive samples

# 3. data augmentation:
#    a.using save_to_csv.py to save the path of true positive and false positive nodules respectively
#    b.using random_select_samples.split_samples_to_three_csv() to split the samples to test/val/train samples(8%,8%,84%)
#    c.augment the true positive train samples 5 times(from 996 to 4980)

# 4. train models:
#    a.using train.py to train models
#    b. there are five model can be choosed in the model.model file. the best one is Simple_CNN_30

# 5. test models:
#    a. using the test.py to test models
#    b. the result of the Simple_CNN_30: recall=90.5%; precision=95.5%
#    c. the model and roc can be found in the folder "batch32_lr0.0003_weight_decay0_time2019-11-21-16:22"


