# MIMII Dataset
$$\small{\textbf{Digital Signal Processing and Deep Learning/Machine learning }}$$

$$\large{\textbf{Unsupervised classification: }}$$ 

$$\large{\textbf{Malfunctioning Industrial Machine Investigation and Inspection}}$$ 

$$\small{\textbf{Dr. Stéphane DEDIEU, Spring - Summer 2024 }}$$



## General Introduction

<span style="color:#4169E1">  
  
Factory machinery is prone to failure or breakdown, resulting in significant expenses for companies. Hence, there is a rising in
terest in machine monitoring using different sensors including microphones. In the scientific community, the emergence of public datasets has led to advancements in acoustic detection and classification of scenes and events, but there are no public datasets that focus on the sound of industrial machines under normal and anomalous operating conditions in real factory environments. 
    
We develop an automatic unsupervised classification model or automatic diagnosis model for detecting failures or breakdowns of industrial machinery based on their acoustics characteristics, recorded with a 8-microphones circular array. 
       
The model is based on the MIMII dataset by Hitachi, Ltd described in the next section.    
Many unsupervised classification models based on this dataset are available in the literature or on Github. We will provide the links and references. 
    
 In this study we somewhat violate the rules of the initial challenge: classification in noisy environment. But since we have access to multiple channels, it makes much sense to denoise the signals before starting the classification process. 

 Therefore, here the challenge is more about turning the 8-microphones array into a <b> "sensor" for monitoring industrial machinery sounds in a noisy envionement.</b> And identifying anomalies, failures, breakdowns.    
    
Instead of classifying various machines or types of machines: pump, fan, valve, slider, ...  we will:

- focus on a specific machine type: valve
- denoise the recordings* using MVDR beamforming and a custom fixed Generalized Sidelobe Canceler (GSC)
- apply unsupervised classification: auto-encoder to two sets: single microphone recordings and denoised GSC output.   
    
<i>*Note that in all noisy recordings, the background noise was recorded separately with the 8-microphones array, and added to the devices sounds.  3 cases: SNR= -6 dB, 0 dB, 6 dB. </i>  
    
  
<b> Plan  </b> 
    
- I   Dataset MIMII
- II  Analysis of sounds/noises
- III Introduction to Denoising strategy
- IV  Valve Activity Detector (VAD)
- V   MVDR + GSC:  creation of a new dataset with single channel of denoised recordings: the GSC output.  
- VI Classification Methodology
- VII  Results
- VIII   Conclusions 
    
    
<b> Potential Applications </b>  

- <b> Rotating machinery </b> Failure Detection: bearings, motors,rotors.  
- <b> HVAC </b> Fault detection and diagnosis (FDD): pumps, compressors, valves.                  

<br> 
<b>Keywords:</b> Python, TensorFlow, Deep Learning, Complex Continuous Wavelets


## Dataset: Recording environment and Set-up 
    
<br>
<span style="color:#4169E1">  
    
We quote the reference article [1] in green: 
    
Regarding the dataset:  <br> 
 
<span style="color:#029942">  

<i> "In this paper, we present a new dataset of industrial machine sounds that we call a sound dataset for malfunctioning industrial machine investigation and inspection (MIMII dataset). Normal sounds were recorded for different types of industrial machines (i.e., valves, pumps, fans, and slide rails), and to resemble a real-life scenario, various anomalous sounds were recorded (e.g., contamination, leakage, rotating unbalance, and rail damage). The purpose of releasing the MIMII dataset is to assist the machine-learning and signal processing community with their development of automated facility maintenance." </i>
<br>
<span style="color:#4169E1">  
Regarding the 8-microphones recordings:
<span style="color:#029942">  
<i> "The dataset was collected using a TAMAGO-03 microphone manufactured by System In Frontier Inc. [21]. It is a circular micro-
phone array that consists of eight distinct microphones, the details of which are shown in Fig. 1. By using this microphone array, we can evaluate not only single-channel-based approaches but also multi-channel-based ones. The microphone array was kept at a distance of 50 cm from the machine (10 cm in the case of valves), and 10-second sound segments were recorded. The dataset contains eight separate channels for each segment. Figure 2 depicts the recording setup with the direction and distance for each kind of machine. Note that each machine sound was recorded in a separate session. Under the running condition, the sound of the machine was recorded as 16-bit audio signals sampled at 16 kHz in a reverberant environment. Apart from the target machine sound, background noise in multiple real factories was continuously recorded and later mixed with the target machine sound for simulating real environments. For recording the background noise, we used the same microphone array as for the target machine sound."</i>
<br>
<span style="color:#4169E1">  
All datasets for normal and "abnormal" machines: pumps, valves, sliders, fans can be downloaded here:
    
https://zenodo.org/records/3384388  
    
<span style="color:#4169E1">  
Part of this dataset: Single channel microphone only, plus Toy car, Toy conveyor, was used in the DCASE 2020 Challenges in 2020 and in the following years.  
    
https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds

https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring    
    

    
#### Microphone Array     
    
The MIMII dataset was recorded with the follwing 8-microphones array:  <br> 

    
| <p align="center"> <img src="MIMII_Microphone_array.png" width="300"  /> </p> |  <p align="center"> <img src="https://github.com/DrStef/MIMII/blob/main/Tamago_egg.png" width="200"  /> </p> |   
| ---       |   ---  |   
| <center> <b><i> The circular microphone array <br> from [1] </i></b> </center> |   <center> <b><i> The Tamago concept <br> from https://www.sifi.co.jp/en/product/microphone-array/ </i></b> </center> |     
    
The microphone array is embedded in a hard "egg shape" in a vertical position. For optimizing the beamformer, we should account for the diffraction of acoustic waves on the egg shape. This requires either:
- an approximation of the egg shape by a prolate spheroid providing an analytical solution of the ascoutic field 
- or a Boundary Element Model 
    
We may work on an analytical model, the prolate spheroid but it will take some time. <span style="color:#900C3F">  At the moment we will treat the 8-microphone array in free field. It is an approximation, and the MVDR beamformer will perform properly at low frequency when the acoustic wavelength is very large compared with the size of the egg, but it will poorly perform in the medium and high frequency range.  </span>
   
    
The configuration for recording the various machines is presented below.  
    
| <p align="center"> <img src="MIMII_Microphone_array_setup.png" width="400"  /> </p> |  
| ---       |   
| <center> <b><i> Recording configuration with the circular microphone array <br> from [1] </i></b> </center> |      
    
https://www.sifi.co.jp/en/product/microphone-array/
    
Index Terms  
Machine sound dataset, Acoustic scene classification, Anomaly detection, Unsupervised anomalous sound detection  


Challenges have been closed for a while 
Our approach is different.    
    

    
- SNR= 6 dB
- SNR= 0 dB
- SNR= -6 dB


##  Denoising ? 

<br>

<span style="color:#4169E1">  
    

In many results with the DCASE2020, DCASE2022 challenge datasets, that include single channels of the MIMII dataset, noise and reverberation are often reported as a contributing factor for poor classification accuracy.  
<br>
If we were to design a system for acquiring industrial sounds, a microphone array is an ideal tool to:

- attenuate reverberations.
- attenuate ambient noise.

with the ability to steer a beam in the direction of interest: the sound source to be monitored.   


Can a beamformer get rid of ambient noise artifically added to the sound of interest ? 
    
- Assuming mic 1 is in the direction of the source, and that some noise was recorded in the direction of microphone number 1, it will be difficult to denoise the recording.  
- if some isotropic ambient noise was recorded, in this case the beamformer will be efficient

##  Multi-Microphone diagnosis sensor.

<br>
<span style="color:#4169E1">  
    
    
If we were to design a sensor for monitoring industrial machinery sounds, in a noisy envionement, then a multi-microphone sesnor i.e. a microphone array, makes absolute sense. 8 microphones might be an overkill, but 6 microphones in a small form factor would do a good job. 
    Here we are going to turn the TAMAGO microphone array in a diagnosis sensor. 
    With proper beamforming filters and noise reduction strategy. 
<br>    
    
    
####  Beamforming
    
Beamforming is a noise reduction technique based on <b><i>spatial filtering</i></b>. Basically the multiple microphones capture acoustic waves  and thei output is combined to increase the gain in a specific direction. 
Beamforming can be combined with classic Noise Reduction techniques as we will see in the next section.       
    
The 68 mm diameter microphone array is small and the number of microphones: 8 is an overkill, and it will oversample acoustic waves at low frequency. When implementing MVDR beamforming we will introduce significant regularization which will limit the Directivity Index. <b> After multiple experimentation, strong regularization was needed... even minor microphone mismatch in magnitude and phase, can significantly degrade the performance of the beamformer. </b>

R=0.068/2; <br> % Radius of the circular array.
%Circular array geometry <br>
- RP(1,:)= [R                   0                        0.00];
- RP(2,:)= [R*cos(45*pi/180)    R*sin(45*pi/180)         0.00];
- RP(3,:)= [R*cos(90*pi/180)    R*sin(90*pi/180)         0.00];
- RP(4,:)= [R*cos(135*pi/180)   R*sin(135*pi/180)        0.00];
- RP(5,:)= [R*cos(pi)           0                        0.00];
- RP(6,:)= [R*cos(225*pi/180)   R*sin(225*pi/180)        0.00];
- RP(7,:)= [R*cos(270*pi/180)   R*sin(270*pi/180)        0.00];
- RP(8,:)= [R*cos(315*pi/180)   R*sin(315*pi/180)        0.00];

    
|<p align="center"> <img src="Wopt_00deg.png" width="450"  /> </p> |  <p align="center"> <img src="DI_90deg_sig5_1em4.png" width="400"  /> </p> |
|       ---       |         ---       | 
| <center> <b><i> Optimum filters 000 deg </i></b> </center> | <center> <b><i> Directivity Index </i></b> </center> |       
    
    
####  Beampatterns
    
| <p align="center"> <img src="Directivity_HorizontalPlan.png" width="350"  /> </p> |  
|       ---       |       
| <center> <b><i> Directivity v. Frequency - Horizontal plane. </i></b> </center> |     
    

The main beam is steered at 000 degrees. In the valve direction. <br>
Beamforming Filters in the frequency domain: real_part and imaginary part are stored in the following files: 

Filters:  512 points, Fs= 16000 Hz, double-sided ! 

Frequencies=[0 : Fs/NFFT : Fs-Fs/NFFT]


Computing optimal MVDR beamforming filters   
    
The 8-microphones array is embedded in a rigid egg shape. It cannot be treated as free field array, except at low frequency when the acoustic wavelength is very large compared with the size of the egg. We will assume that the TAMAGO egg is a hard prolate spheroid and we will use analytical or semi-analytical models for characterezing the acoustics field diffracted by the "egg".  This will be devlopped in PART II.  Beamforming.      
    
Main beam: 
  

Noise channel

The Noise channel is intended to implement a non-adaptive Generalized Sidelobe Canceller or Multi-channel Wiener.  

The code for generating the 2 sets of filters is confidential. Especially the "noise channel" filters, since it is a custom implementation that alleviates ill-conditonned noise coherence matrix at low frequency, when the acoustic wavelenth is very large compared with the size of the array. And it is the case here with a small 60mm diameter array.    

Theoretical aspects for computing the filters are presented in Ward [], chapter II: <i> "Superdirective Microphone Arrays" </i>. 
We compute:     
- optimal MVDR beamforming filters, for the main beam and main channel. Where we assume an isotropic noise field. Filters $W^H_f$  on the block diagram. 
- filters of a non adaptive generalized side lobe canceller (GSC) or multi-channel Wiener for the secondary, "orthogonal" channel. Filters $W^H_{v}B$ on the block diagram. 
   
The computation of the filters is left as an exercise. Some experimentation will be needed for regularizing the various ill-conditionned matrices. 

#### Generalized Side Lobe Canceller 

We will use a fixed beamforming approch. Where the GSC has a frozen filters and enhances 
The fixed GSC strategy is equivalent to a multi-channel Wiener gain. 

Denoising is performed in two stages:

- beamforming alone 
- Generalized Sidelobe Canceller with 2 channels 

https://www.researchgate.net/figure/General-structure-of-the-generalized-sidelobe-canceller-GSC-with-Y-k-b-being-the_fig2_224208512
    
    
| <p align="center"> <img src="GSC_blockdiagram.png" width="600"  /> </p> |  
| ---       |   
| <center> <b><i> GSC Block-Diagram from [6] </i></b> </center> |       
    

We propose a pseudo-real time implementation.     
A "valve activity detector" would be needed when performing the spectral subtraction: when the valve is active, the algorithm stops collecting noise frames!    

In a first approximation we will work without a "valve activity detection". Because the background noise is somewhat pesudo-stationary while the valve sound is brief, we will collect a long "noise history" that will     . This method will not work for other devices: pump, fan. 

The GSC introduces distortion in the valve sound. This may not be a problem if we train the model with distorted valve sounds.  But it would be a redhibitory issue for ASR applications for example.    

    
    
Pseudo code: 
For denoising the recordings:
- Nfft= 512, fs=16000Hz, t= 32 ms.  
- sliding a NFFT=512 points window frame on the 10 s recordings with a shift of NFFT/3  
- compute the FFT of each microphone channel.
- apply the beamforming filters to each microphone channel in the frequency domain
- sum
- compute the IFFT
- rebuild the denoised signal frame by frame. 

The procedure is in the attached Octave/Matlab script. It will be turned into a Python code.  


##  Valve Activity Detection (VAD)

<br> 
<span style="color:#4169E1">  
    
Just like in speech enhancements, for optimum performance, the second stage of the denoising algorithm, must stop collecting noise frames when the valve is active. 
Therefore we design a VAD. Not voice activity detection but "Valve Activity Detection". 
    
We created two datasets of 512 points mono sound frames, 32 ms, sampled at fs= 16kHz. 
    
- Valve sound frames: collected in the <b>6dB SNR</b> dataset, Normal, id00, id02, id04, id06. Ideally we would like to have access to valve sounds in silent background, but this data is not available. We had access to "noisy" valve sound frames only, the reason why we selected the best SNR data available.     
- Background noise frames: collected in the <b>-6dB SNR</b> dataset, Normal, id00, id02, id04, id06 when valves are not active. 
    
In this notebook we will investigate: 
    
- low complexity VAD models based on Machine Learning. These models will be used in our 2-stages Noise Reduction algorithm. 
- high complexity models based on advanced features and Deep Learning.  
    
The low complexity models are the priority for proceeding with Noise Reduction. <br> 
The Notebook will be updated on a regular basis with more advanced models. 
    
    
    
    
<span style="color:#4169E1">  

For designing the valve activity detector we will: <br>
- build labeled datasets of background noise and valve sounds. Ideally we should have access to valve sound without any noise.   
- 32 ms frames. Nfft=512 bins.  
- compute STFT, mel-spectrograms or wavelets transforms.
- build a Deep Learning model.  CNN.   

###   References 

<br>
<span style="color:#4169E1">  
    

https://github.com/MIMII-hitachi/mimii_baseline/
    

[1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, <i>“MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,”</i> arXiv preprint arXiv:1909.09347, 2019.

[2] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, <i>“MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” </i> in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.

[3] Y. Kawaguchi, R. Tanabe, T. Endo, K. Ichige, and K. Hamada, <i>“Anomaly detection based on an ensemble of dereverberation and anomalous sound extraction,”</i> in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 865–869


[6] Nilesh Madhu and Rainer Martin, <i>"A Versatile Framework for Speaker Separation Using a Model-Based Speaker Localization Approach" </i>, October 2011 IEEE Transactions on Audio Speech and Language Processing 19(7):1900 - 1912, DOI:10.1109/TASL.2010.2102754















