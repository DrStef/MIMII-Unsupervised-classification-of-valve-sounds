# MIMII Dataset
$$\small{\textbf{Digital Signal Processing and Deep Learning/Machine learning }}$$

$$\large{\textbf{Unsupervised classification: }}$$ 

$$\large{\textbf{Malfunctioning Industrial Machine Investigation and Inspection}}$$ 

$$\small{\textbf{Dr. Stéphane DEDIEU, Spring - Summer 2024 }}$$



## General Introduction

<b>This document is under construction.</b> (Sept 17th, 2024)

<span style="color:#4169E1">  
  
Industrial machinery often experiences failures or breakdowns, leading to considerable costs for businesses. Consequently, there's growing interest in monitoring these machines with various sensors, such as microphones. <br>
Within the scientific community, the availability of public datasets has enhanced the development of acoustic detection and classification techniques for various scenes and events. <br> 

Hitachi Ltd. has developed the MIMII dataset for classifying sounds from industrial machines operating under both normal and faulty conditions in actual factory environments. This dataset includes:

- Subsets of machines: pumps, valves, sliders, fans.
- Subsets of operating conditions: normal and abnormal.
- Background noise.


MIMII stands for Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection. <br> Many unsupervised classification models based on this dataset can be found in the literature or on GitHub. We will provide links and references accordingly.

We are developing an automatic, unsupervised classification model or an automatic diagnosis model for detecting failures or breakdowns in industrial machinery based on their acoustic characteristics, recorded with an 8-microphone circular array. Unlike most classification models found in literature, this study somewhat deviates from the initial challenge's rules: classification of noisy signals. However, since we have access to multiple channels, it makes practical sense to denoise the signals before initiating the classification process.

Thus, the challenge here is transforming the 8-microphone array into a <b> "sensor" for monitoring industrial machinery sounds in noisy environments</b>. Then, we apply the classification model to these denoised signals to automatically identify anomalies, failures, or breakdowns.

Rather than classifying various types of machines (pumps, fans, valves, sliders), our focus will be:

- Concentrating on a specific machine type: valves.
- Denoising the recordings using MVDR beamforming combined with a custom, fixed Generalized Sidelobe Canceler (GSC).
- Applying unsupervised classification techniques (auto-encoder, etc.) to two sets of signals: single microphone recordings and the denoised GSC output, to detect defective valves and demonstrate the benefits of MVDR beamforming combined with GSC.


<i>*Note: In all noisy recordings, the background noise was captured separately using the 8-microphone array and then added to the device sounds. This was done under three SNR conditions: -6 dB, 0 dB, and 6 dB. More details can be found in the acquisition setup section.</i>


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
    
https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds <br>
https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring    
    

    
#### Microphone Array     
    
The MIMII dataset was recorded with the follwing 8-microphones array:  <br> 

| <p align="center"> <img src="MIMII_Microphone_array.png" width="300"  /> </p> |  <p align="center"> <img src="https://github.com/DrStef/MIMII/blob/main/Tamago_egg.png" width="200"  /> </p> |   
| ---       |   ---  |   
| <center> <b><i> The circular microphone array <br> from [1] </i></b> </center> |   <center> <b><i> The Tamago concept <br> from https://www.sifi.co.jp/en/product/microphone-array/ </i></b> </center> |     
    
The microphone array is embedded in a hard "egg shape" in a vertical position. For optimizing the beamformer, we should account for the diffraction of acoustic waves on the egg shape. This requires either:
- an approximation of the egg shape by a prolate spheroid providing an analytical solution of the ascoutic field 
- or a Boundary Element Model 
    
We may work on an analytical model, the prolate spheroid, but it will take some time. <span style="color:#900C3F">  At the moment we will treat the 8-microphone array in free field. It is an approximation, and the MVDR beamformer will perform properly at low frequency when the acoustic wavelength is very large compared with the size of the egg, but it will poorly perform in the medium and high frequency range up to 8kHz.  </span>
   
    
The configuration for recording the various machines is presented below.  
    
| <p align="center"> <img src="MIMII_Microphone_array_setup.png" width="400"  /> </p> |  
| ---       |   
| <center> <b><i> Recording configuration with the circular microphone array <br> from [1] </i></b> </center> |     

We will work with the Valve dataset only, therefore with a beamformer steered at 000 deg.  
    
https://www.sifi.co.jp/en/product/microphone-array/
    

##  Denoising ? 

<br>

<span style="color:#4169E1">  
    
In various models and results developed during the DCASE2020 and DCASE2022 challenges, which included single channels from the MIMII dataset, noise and reverberation were often identified as significant factors contributing to poor classification accuracy.

When designing a system for capturing industrial sounds, a microphone array would be an optimal choice to:

- Attenuate reverberations.
- Attenuate ambient noise.

Additionally, it offers the capability to focus a beam towards the sound source of interest.

Can a beamformer eliminate ambient noise artificially added to the sound of interest?

Background noise that was added to the sound sources of interest in the MIMII dataset was captured using the same TAMAGO 8-microphone array, and subsequently, all channels were mixed under the following conditions:

- SNR = 6 dB
- SNR = 0 dB
- SNR = -6 dB (representing the worst-case scenario)

We plan to denoise and classify signals under the SNR = -6 dB condition, which is the most challenging scenario.

- Assuming microphone 1 is directed towards the sound source of interest (in this case, a valve), and if background noise was also recorded from that direction, it would be difficult to denoise the recordings effectively.
- However, if the ambient noise recorded by the array is isotropic, the beamformer would be more efficient in this scenario.

Fortunately, in most recordings we reviewed from the -6dB_Valve dataset, the ambient noise appears to be relatively isotropic, or at least, the primary noise source is not directly at 0 degrees. Consequently, the MVDR beamformer should effectively attenuate the ambient noise, particularly at frequencies below 1000-1500 Hz, assuming the array operates in a free field.

##  Multi-Microphone diagnosis sensor.

<br>
<span style="color:#4169E1">  
  
If we were to design a sensor for monitoring industrial machinery sounds in a noisy environment, then a multi-microphone sensor, i.e., a microphone array, would make absolute sense.

Here, we are going to transform the TAMAGO microphone array into a diagnostic sensor, employing proper beamforming filters and a noise reduction strategy.

#### Beamforming

Beamforming is a noise reduction technique based on spatial filtering. Essentially, multiple microphones capture acoustic waves, and their outputs are combined to increase the gain in a specific direction. Beamforming can be combined with classic noise reduction techniques, as we will see in the next section.

The 68 mm diameter microphone array, though small, with its eight microphones, represents an overkill, leading to oversampling of acoustic waves at low frequencies, which creates the following issues:

- With too many microphones, optimal MVDR beamforming filters can achieve very high gains for maximum directivity, typically +50 or +60 dB. This significantly degrades the White Noise Gain (WNG), making their practical implementation problematic.
- Minor mismatches in magnitude and phase among the microphones can further degrade the beamformer's performance significantly.


Therefore, when implementing the MVDR beamforming with the TAMAGO microphone array, we will introduce substantial regularization at low frequencies, which will compromise the Directivity Index at these frequencies.

#### Computing Optimal MVDR Beamforming Filters

The 8-microphone array is embedded within a rigid egg shape. It cannot be treated as a free-field array, except at low frequencies where the acoustic wavelength is very large compared to the size of the egg. We will assume that the TAMAGO egg is a hard prolate spheroid and will use analytical or semi-analytical models to characterize the acoustic field diffracted by the "egg". This will be explored in PART II. Once the simulation is complete, we will develop a new MIMII denoised valve dataset.

We compute two sets of filters:

- Main beam: optimal MVDR beamforming filters for the main beam and channel, assuming an isotropic noise field. These are filters $W^H_f$ in the block diagram in the next section.
- "Noise channel": filters for a non-adaptive generalized sidelobe canceller (GSC) or multi-channel Wiener for the secondary, "orthogonal" channel. These are filters  $W^H_{v}B$ in the block diagram.

The code for generating these two sets of filters is confidential. <br> Theoretical aspects for computing the filters are presented in Ward [citation], Chapter II: "Superdirective Microphone Arrays". <br> The computation of the filters is left as an exercise. Some experimentation will be necessary for regularizing the various ill-conditioned matrices.


<i>R=0.068/2</i>  % Radius of the circular array in meter (m) <br>
% Circular array geometry <br>
- <i> RP(1,:)= [R                   0                        0.00]
- RP(2,:)= [R*cos(45*pi/180)    R*sin(45*pi/180)         0.00]
- RP(3,:)= [R*cos(90*pi/180)    R*sin(90*pi/180)         0.00]
- RP(4,:)= [R*cos(135*pi/180)   R*sin(135*pi/180)        0.00]
- RP(5,:)= [R*cos(pi)           0                        0.00]
- RP(6,:)= [R*cos(225*pi/180)   R*sin(225*pi/180)        0.00]
- RP(7,:)= [R*cos(270*pi/180)   R*sin(270*pi/180)        0.00]
- RP(8,:)= [R*cos(315*pi/180)   R*sin(315*pi/180)        0.00]</i>

    
|<p align="center"> <img src="Wopt_00deg.png" width="450"  /> </p> |  <p align="center"> <img src="DI_90deg_sig5_1em4.png" width="400"  /> </p> |
|       ---       |         ---       | 
| <center> <b><i> Optimum filters 000 deg </i></b> </center> | <center> <b><i> Directivity Index </i></b> </center> |       

Beamforming Filters in the frequency domain: real_part and imaginary part are stored in the following files: 

Filters:  512 points, Fs= 16000 Hz, double-sided ! 
Frequencies=[0 : Fs/NFFT : Fs-Fs/NFFT]
    
####  Beampatterns

We can plot the beampatterns in the horizintal plane v. frequency : 
    
| <p align="center"> <img src="Directivity_HorizontalPlan.png" width="350"  /> </p> |  
|       ---       |       
| <center> <b><i> Directivity v. Frequency - Horizontal plane. </i></b> </center> |     
    

The main beam is steered at 000 degrees. In the valve direction. <br>

    
    
|<p align="center"> <img src="Beampattern_500Hz.png" width="250"  /> </p> |  <p align="center"> <img src="Beampattern_1000Hz.png" width="250"  /> </p> |   <p align="center"> <img src="Beampattern_4500Hz.png" width="250"  /> </p>                       |
|       ---       |         ---       |  ---  |
| <center> <b><i> Beampattern_500Hz  </i></b> </center> | <center> <b><i> Beampattern_1000Hz </i></b> </center> |   <center> <b><i> Beampattern_4500Hz </i></b> </center> |


#### Generalized Side Lobe Canceller 

We will use a fixed beamforming approach. The fixed GSC strategy is equivalent to a multi-channel Wiener gain. But instead of implementing a spectral difference, we can replace it with more advanced NR gains and evaluation of a priori_SNR. 

Denoising is performed in two stages:

- stage I: MVDR beamforming alone 
- stage II: Generalized Sidelobe Canceller with 2 channels 

The structure of a real GSC is presented in the following article: <br>
https://www.researchgate.net/figure/General-structure-of-the-generalized-sidelobe-canceller-GSC-with-Y-k-b-being-the_fig2_224208512

We extract the block diagram. 
    
| <p align="center"> <img src="GSC_blockdiagram.png" width="600"  /> </p> |  
| ---       |   
| <center> <b><i> GSC Block-Diagram from [6] </i></b> </center> |       
    

We propose a pseudo-real-time implementation. A "valve activity detector" would be necessary when performing spectral subtraction; the algorithm must stop collecting noise frames when the valve is active!

Initially, we will approximate without a "valve activity detector" because much of the background noise is somewhat pseudo-stationary, while the valve sound is brief. We will accumulate a long "noise history." This approach might not be effective for other devices like pumps or fans. Subsequently, we will develop a Valve Activity Detector and compare both methods: with and without VAD.

The GSC might introduce distortion into the valve sound. We will assess whether this affects the accuracy of the classification model. Such added distortion could be a significant issue for applications like Automatic Speech Recognition (ASR).

#### Pseudo-Real-Time Implementation:

Frame-by-Frame Processing: We implement this with overlap-add, sliding an NFFT-length window over all 10-second signals with a 66% overlap. We compute the FFT, apply beamforming, and noise reduction gain in the frequency domain, then reconstruct the denoised output signals frame by frame using an IFFT.

Parameters for Denoising the Recordings:

- Frames: NFFT = 512, sampling rate (fs) = 16000 Hz, frame duration (t) = 32 ms.
- Slide an NFFT=512 point window frame over the 10-second recordings with a shift of NFFT/3.
- Compute the FFT for each microphone channel.
- Apply the beamforming filters in the frequency domain to each microphone channel.
- Sum the results.
- Compute the IFFT.
- Reconstruct the denoised signal frame by frame.

This procedure is detailed in the Jupyter Notebook: Part I: Preliminary Activities.

##  Valve Activity Detection (VAD)

<br> 
<span style="color:#4169E1">  

Just like in speech enhancement algorithms, for optimal performance, the second stage of the denoising process must cease collecting noise frames when the valve is active. Therefore, we design a VAD, not for Voice Activity Detection, but for "Valve Activity Detection".

Ideally, we should have access to valve sounds without any noise. We built labeled datasets of background noise and valve sounds consisting of 512-point mono sound frames, each lasting 32 ms, sampled at 16 kHz. 
 
- Valve sound frames: collected from the 6 dB SNR dataset, specifically Normal, id00, id02, id04, id06. Ideally, we would have access to valve sounds against a silent background, but such data is not available. We had access only to "noisy" valve sound frames, which is why we selected the best SNR data available.
- Background noise frames: collected from the -6 dB SNR dataset, Normal, id00, id02, id04, id06, when valves are inactive.

In this study, we will explore:

- Low-complexity VAD models based on Machine Learning. These models will be utilized in our two-stage Noise Reduction algorithm.
- High-complexity models leveraging advanced features:  Short-Time Fourier Transforms (STFT), mel-spectrograms, or wavelet transforms and Deep Learning, specifically a Convolutional Neural Network (CNN).

The low-complexity models are the priority for proceeding with Noise Reduction. Notebooks will be updated regularly with more advanced models.


##  Early Results Beamforming







https://github.com/user-attachments/assets/c4e86685-4dbe-4a32-aa33-dfcb5771128d





|  https://github.com/user-attachments/assets/03c98bcb-2b2b-4ff4-864a-7c9310baee98   |  https://github.com/user-attachments/assets/d790a929-063d-4aa7-aadf-b157f4bfc6bf      |   https://github.com/user-attachments/assets/c4e86685-4dbe-4a32-aa33-dfcb5771128d      |    
|   ---      |   ---     |   ---      |    
|         |        |         |  


##   References 

<br>
<span style="color:#4169E1">  

https://github.com/MIMII-hitachi/mimii_baseline/
    
[1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, <i>“MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,”</i> arXiv preprint arXiv:1909.09347, 2019.

[2] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, <i>“MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” </i> in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.

[3] Y. Kawaguchi, R. Tanabe, T. Endo, K. Ichige, and K. Hamada, <i>“Anomaly detection based on an ensemble of dereverberation and anomalous sound extraction,”</i> in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 865–869


[6] Nilesh Madhu and Rainer Martin, <i>"A Versatile Framework for Speaker Separation Using a Model-Based Speaker Localization Approach" </i>, October 2011 IEEE Transactions on Audio Speech and Language Processing 19(7):1900 - 1912, DOI:10.1109/TASL.2010.2102754



##   Notebooks


<b><i>Part I: Preliminary Activities</i></b>    (includes the General Introduction)












