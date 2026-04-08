This data is provided by the Human-Oriented Robotics and Controls (HORC) lab, ASU. http://horc.engineering.asu.edu/HORC/Home.html


When using this dataset, please cite the following paper:

C. Nguyen, G. Karavas, P. Artemiadis,"Inferring imagined speech using EEG signals: a new approach using Riemannian Manifold features", Journal of Neural Engineering, July 2017 

Dataset description:
- This folder contains the dataset for the vowels case after appying the following preprocessing steps:
	- Bandpass filtering at [8-70]Hz using a 5th order Butterworth filter.
	- Notch filter at 60Hz, using the Matlab function iirnotch.
	- EOG artifact removal.
 	- Downsampling from 1000Hz to 256Hz.
- Each file has two variables:
	- eeg_data_wrt_task_rep_no_eog_256Hz_end_trial : is a cell of dimension [3 x 100], of 3 classes (1=/a/,2=/i/,3=/u/) and 100 trials. Each cell element contains eeg signals from 64 channels during 2seconds at the end of the trial, (during resting condition).
	- eeg_data_wrt_task_rep_no_eog_256Hz_last_beep : is a cell of dimension [3 x 100], of 3 classes (1=/a/,2=/i/,3=/u/) and 100 trials. Each cell element contains  eeg signals from 64 channels during 5seconds after the last beep, ( during speech imagery).

- Channel indices [1,10,33,64] were used to record EOG artifacts; thus they should only be used for EOG removal.

- File sub_8_ch64_v_eog_removed_256Hz_time_correlation_effect.mat contains data for analyzing the time correlation artifact as described in Section VI.A (paragraphs 7 and 8). 

July 2017.


