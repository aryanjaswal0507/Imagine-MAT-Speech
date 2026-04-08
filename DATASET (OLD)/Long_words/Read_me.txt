This data is provided by the Human-Oriented Robotics and Controls (HORC) lab, ASU. http://horc.engineering.asu.edu/HORC/Home.html


When using this dataset, please cite the following paper:

C. Nguyen, G. Karavas, P. Artemiadis,"Inferring imagined speech using EEG signals: a new approach using Riemannian Manifold features", Journal of Neural Engineering, July 2017 

Dataset description:
- This folder contains the dataset for the long words case after appying the following preprocessing steps:
	- Bandpass filtering at [8-70]Hz using a 5th order Butterworth filter.
	- Notch filter at 60Hz, using the Matlab function iirnotch.
	- EOG artifact removal.
 	- Downsampling from 1000Hz to 256Hz.
- Each file has two variables:
	- eeg_data_wrt_task_rep_no_eog_256Hz_end_trial : is a cell of dimension [2 x 100], of 2 classes (1="cooperate",2="independent") and 100 trials. Each cell element contains eeg signals from 64 channels during 2seconds at the end of the trial, (during resting condition).		- eeg_data_wrt_task_rep_no_eog_256Hz_last_beep : is a cell of dimension [2 x 100], of 2 classes (1="cooperate",2="independent") and 100 trials. Each cell element contains eeg signals from 64 channels during 5seconds after the last beep, ( during speech imagery).

- Channel indices [1,10,33,64] were used to record EOG artifacts; thus they should only be used for EOG removal.

July 2017.


