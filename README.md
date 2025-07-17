

$$\small{\textbf{Digital Signal Processing, Deep Learning, and Machine Learning for Unsupervised Anomaly Detection}} $$   
$$\small{\textbf{Malfunctioning Industrial Machine Investigation and Inspection (MIMII)}}$$
$$\small{\textbf{Dr. Stéphane DEDIEU,  Summer 2024 - rev. June 2025 }}$$



<h1 align="center">MIMII Dataset: Unsupervised Classification of Valve Sounds with CNN-Based Autoencoder</h1>

This repository hosts an unsupervised classification pipeline for detecting anomalies in industrial valve sounds using the MIMII dataset (valves, 6 dB, 0 dB, –6 dB SNR). Our solution leverages an 8-microphone array for noise reduction and a convolutional neural network (CNN)-based autoencoder for anomaly detection, targeting industrial applications like predictive maintenance for HVAC systems (e.g., TRANE compressors). The pipeline includes proprietary AC-STFT feature extraction, multi-channel denoising, and frame alignment to ensure robust performance in noisy environments. This work is developed by [Bloo Audio], with ongoing results showcased on ([LinkedIn](https://www.linkedin.com/in/sdedieu/))    

## Project Overview
We aim to automatically detect valve failures (e.g., contamination, leakage) in the MIMII dataset using unsupervised learning, focusing on acoustic signals recorded with an 8-microphone TAMAGO-03 array (16 kHz, 16-bit). Unlike traditional approaches, we denoise multi-channel signals before classification, transforming the array into a "smart sensor" for industrial monitoring. The pipeline is divided into three parts:

- **Part I: Autoencoder Classification** – Train a CNN-based autoencoder on normal valve sounds (6 dB SNR) to flag anomalies via reconstruction errors.
- **Part II: Noise Reduction** – Develop 8-mic beamforming and Ephraim-Malah filtering to isolate valve sounds in noisy factories.
- **Part III: Comparative Analysis** – Compare autoencoder performance on raw vs. denoised –6 dB valve data, using aligned 1.5s frames.

### Key Features

- **AC-STFT Features**: Proprietary complex transform (256x256x2 spectrograms magnitude+phase, hop_length=91, n_fft=512) detects anomalies, outperforming STFT by ~25% (AUC: 0.992–0.998 vs. 0.7416 for id_04).
- **CNN Autoencoder**: Unsupervised model (latent_dim=128, dropout=0.5, L2=0.002) trained on normal frames, achieving AUC > 0.8 (ongoing, June 3, 2025).
- **Frame Alignment**: Identical 1.5s frames extracted across 6 dB and –6 dB (raw/denoised) datasets using saved indices for consistency.
- **Scalability**: Pipeline adapts to other machines (compressor, sliders...) for predictive maintenance.
- **8-Mic Denoising**: Beamforming (e.g., MVDR) and spectral subtraction suppress factory noise, enhancing valve signals (–6 dB SNR).

### Dataset
- **MIMII Dataset**: Valve sounds (id_00, id_02, id_04, id_06) at 6 dB, 0 dB, –6 dB SNR, recorded with 8-mic TAMAGO-03 array. Normal (~5000–10000s) and anomalous (~1000s) sounds per valve. [Zenodo: 10.5281/zenodo.3384388][](https://zenodo.org/records/3384388)
- **6_dB_valve**:
10 seconds recordings (WAV files), fs= 16 kHz, SNR=6dB. We extract 1.5s frames from the 10s recordings and focus on the valves impulse sounds.    
  - Normal: 3691 1.5s frames (id_00: 991, id_02: 708, id_04: 1000, id_06: 992).
  - Abnormal: 479 frames (id_00: 119, others: 120).
  - Source: 10s WAVs split via Hilbert envelope filtering (threshold_factor=1.5).
- **–6_dB_valve**: Noisy data, to be denoised and aligned with 6_dB_valve frames (planned, in Part III, Summer 2025).
- **Preprocessing**: AC-STFT spectrograms (256x256x2, magnitude + phase, scaled [0, 1]) from 1.5s frames.

### Pipeline Details

#### Part I: CNN-Based Autoencoder
- **Objective**: Train an autoencoder on normal 6_dB_valve frames to detect anomalies (high reconstruction errors).
- **Model**:
  - Architecture: CNN (32→64→128 filters, Conv2D/Transpose, latent_dim=128).
  - Regularization: Dropout=0.5, spatial dropout=0.3, L2=0.002.
  - Optimizer: Adam with ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6).
  - Loss: MSE, batch_size=64, epochs=30, early stopping (patience=10).
- **Training**:
  - Data: `ids_X_train` (3691 normal, 256x256x2), `ids_X_test` (958: 479 normal + 479 abnormal, seed=25).
  - Initial Run (Stopped, Epoch 10): Overfitting (loss=0.0467, val_loss=0.0848, gap=0.0381, spike to 0.1111).
  - Retraining (June 3, 2025, Ongoing): Improved at Epoch 8 (loss=0.0500, val_loss=0.0529, gap=0.0029, lr=0.0010).
  - Target: val_loss ~0.0500, gap < 0.006, AUC > 0.8 (vs. id_04’s 0.992–0.998).
- **Results**: Awaiting Epoch 10+ metrics (AUC, confusion matrix, FN spectrograms for >4 kHz patterns).
- **Visualizations**: ROC curve, confusion matrix, training history, error histogram, FN spectrograms (results/plots/unified).


#### Part II: Noise Reduction
- **Objective**: Denoise –6 dB valve audio to create `denoised_id##` dataset, enhancing valve signals for anomaly detection.
- **Methods**:
  - **Beamforming**: MVDR or delay-and-sum using 8-mic TAMAGO-03 array, modeled as a prolate spheroid for diffraction (Part II simulation planned).
  - **Ephraim-Malah Filtering**: Spectral subtraction to remove residual noise.
  - **VAD**: Excludes artifacts (e.g., “coin coin” noise).
- **Status**: Denoising pipeline in development, to be applied to –6 dB 10s WAVs before 1.5s frame extraction (IA laptop, June 3, 2025).
- **Output**: Denoised WAVs in `-6_dB_valve/denoised`, aligned with 6_dB_valve frames.


#### Part III: Comparative Analysis
- **Objective**: Compare autoencoder performance on raw vs. denoised –6 dB valve data, using aligned 1.5s frames.
- **Frame Alignment Strategy**:
  - Extract 1.5s frames from 6_dB_valve 10s WAVs (Hilbert envelope, hop=100 ms), saving indices (valve_id, file_name, label, frame_number, start_time) to `results/6dB_frame_indices.csv`.
  - Use indices to extract identical frames from –6 dB_valve (raw: `-6_dB_valve/valve_1p5s`, denoised: `-6_dB_valve/denoised_1p5s`).
  - Status: Planned for IA laptop, June 3, 2025.
- **Analysis**: Train/test autoencoders on raw/denoised –6 dB data, report AUCs, and compare to 6_dB_valve.

### Results
- **id_04 (Single Valve, 1.5s, Seed=25)**: AUC=0.992 (vs. 0.998 seed=42), val_loss ~0.0501, gap ~0.0044.
- **Unified Model (All Valves, Retraining)**:
  - Epoch 8 (June 3, 2025): loss=0.0500, val_loss=0.0529, gap=0.0029.
  - Awaiting final AUC, confusion matrix (TN, FP, FN, TP), F1-score.
- **id_02 Note**: Monitoring 708 normal frames (vs. ~1000 others); augmentation (time-shifting, noise_factor=0.05) planned if AUC < 0.8.
- **Plots**: ROC, confusion matrix, FN spectrograms (results/plots/unified) for Part II and TRANE pitch.

### Future Work
- **–6 dB Denoising**: Implement 8-mic pipeline (beamforming, Ephraim-Malah) for Part I, extract aligned frames for Part III.
- **TRANE Application**: Adapt pipeline for compressor monitoring (8-mic arrays, vibration sensors), as pitched to TRANE Canada (PowerPoint in development).
- **ClearFormer Exploration**: Future project on Google’s ClearFormer for ASR noise reduction (time-frequency masks), separate from MIMII.
- **LinkedIn Showcase**: Post updated AUCs, plots, and PowerPoint schematic (inspired by ResearchGate, 2022) on [Your Company LinkedIn].

### Installation
```bash
pip install -r requirements.txt
```





## General Introduction (Preliminary MVDR Beamforming developments)

<i> <b>Note:</b> In the preliminary stage of the project, we developed and tested MVDR beamforming using an 8-microphone Tamago array, combined with an Ephraim-Malah denoising algorithm, to support future denoising of the -6dB valve dataset. This introduction and its results are kept as is and will not be included in the notebook’s Part II: Denoising Strategy. </i>
<b>This document is under construction.</b> (Sept 17th, 2024- Update June 2025)

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
    
- I    Dataset: Recording environment and Set-up 
- II   Denoising strategy
- III  Multi-Microphone diagnosis sensor.
- IV   Valve Activity Detection (VAD) 
- V   Early Results Beamforming
- VI  References
- VII   Conclusions 
    
    
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
    
The MIMII dataset was recorded with the following 8-microphones array:  <br> 

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
    

##  Denoising strategy

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

### Beamforming

Beamforming is a noise reduction technique based on spatial filtering. Essentially, multiple microphones capture acoustic waves, and their outputs are combined to increase the gain in a specific direction. Beamforming can be combined with classic noise reduction techniques, as we will see in the next section.

The 68 mm diameter microphone array, though small, with its eight microphones, represents an overkill, leading to oversampling of acoustic waves at low frequencies, which creates the following issues:

- With too many microphones, optimal MVDR beamforming filters can achieve very high gains for maximum directivity, typically +50 or +60 dB. This significantly degrades the White Noise Gain (WNG), making their practical implementation problematic.
- Minor mismatches in magnitude and phase among the microphones can further degrade the beamformer's performance significantly.


Therefore, when implementing the MVDR beamforming with the TAMAGO microphone array, we will introduce substantial regularization at low frequencies, which will compromise the Directivity Index at these frequencies.

### Computing Optimal MVDR Beamforming Filters

The 8-microphone array is embedded within a rigid egg shape. It cannot be treated as a free-field array, except at low frequencies where the acoustic wavelength is very large compared to the size of the egg. We will assume that the TAMAGO egg is a hard prolate spheroid and will use analytical or semi-analytical models to characterize the acoustic field diffracted by the "egg". This will be explored in PART II. Once the simulation is complete, we will develop a new MIMII denoised valve dataset.

We compute two sets of filters:

- Main beam: optimal MVDR beamforming filters for the main beam and channel, assuming an isotropic noise field. These are filters $W^H_f$ in the block diagram in the next section.
- "Noise channel": filters for a non-adaptive generalized sidelobe canceller (GSC) or multi-channel Wiener for the secondary, "orthogonal" channel. These are filters  $W^H_{v}B$ in the block diagram.

The code for generating these two sets of filters is confidential. <br> Theoretical aspects for computing the filters are presented in Ward [5], Chapter II: "Superdirective Microphone Arrays". <br> The computation of the filters is left as an exercise. Some experimentation will be necessary for regularizing the various ill-conditioned matrices.


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
    
###  Beampatterns

We can plot the beampatterns in the horizontal plane v. frequency : 
    
| <p align="center"> <img src="Directivity_HorizontalPlan.png" width="350"  /> </p> |  
|       ---       |       
| <center> <b><i> Directivity v. Frequency - Horizontal plane. </i></b> </center> |     
    

The main beam is steered at 000 degrees. In the valve direction. <br>

    
    
|<p align="center"> <img src="Beampattern_500Hz.png" width="250"  /> </p> |  <p align="center"> <img src="Beampattern_1000Hz.png" width="250"  /> </p> |   <p align="center"> <img src="Beampattern_4500Hz.png" width="250"  /> </p>                       |
|       ---       |         ---       |  ---  |
| <center> <b><i> Beampattern_500Hz  </i></b> </center> | <center> <b><i> Beampattern_1000Hz </i></b> </center> |   <center> <b><i> Beampattern_4500Hz </i></b> </center> |

<br>

### Generalized Side Lobe Canceller 

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

### Pseudo-Real-Time Implementation:

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

## Valve Activity Detection (VAD) 

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

We selected an 8-channel recording for processing: id00_n_00000117.

<b>Processing Applied:</b>

- MVDR beamforming was implemented, assuming the microphone array operates in a free field.
- Subsequently, a Generalized Sidelobe Canceller (GSC) strategy was employed, incorporating Valve Activity Detection.


<b>Conversion for GitHub Playback:</b>

The original .wav files were converted to .mp4 format using VLC media player for easier playback on GitHub.


<b>Observation from Processed Recordings:</b>

- MVDR Beamforming: This method showcases effective background noise attenuation at lower frequencies. However, there's a noticeable degradation in noise reduction at higher frequencies, likely due to the free-field assumption.
- GSC Strategy: Offers additional noise reduction beyond MVDR, though it introduces some distortion in valve sounds.


Further Information:  <br>

A detailed analysis of these techniques and results will be available in a forthcoming Jupyter Notebook.


<b><i>Microphone n. 1:</i></b>

https://github.com/user-attachments/assets/843f1dee-9837-4e55-85d5-2264ed983476

<b><i>MVDR Beamformer output:</i></b>

https://github.com/user-attachments/assets/ecfa8963-bb75-4864-99d7-4e31257d3b4a

<b><i>GSC output with Valve Activity Detection: </i></b> 

https://github.com/user-attachments/assets/c4e86685-4dbe-4a32-aa33-dfcb5771128d



##   References

<br>

https://github.com/MIMII-hitachi/mimii_baseline/
    

[1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, <i>“MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,”</i> arXiv preprint arXiv:1909.09347, 2019.

[2] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, <i>“MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” </i> in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.

[3] Y. Kawaguchi, R. Tanabe, T. Endo, K. Ichige, and K. Hamada, <i>“Anomaly detection based on an ensemble of dereverberation and anomalous sound extraction,”</i> in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 865–869

[5] M. Brandstein and D. Ward, “Microphone Arrays: Signal Processing Techniques and Applications,” Digital Signal Processing, 2001, Springer. 


[6] Nilesh Madhu and Rainer Martin, <i>"A Versatile Framework for Speaker Separation Using a Model-Based Speaker Localization Approach" </i>, October 2011 IEEE Transactions on Audio Speech and Language Processing 19(7):1900 - 1912, DOI:10.1109/TASL.2010.2102754




##   Notebooks


<b><i>Part I: Preliminary Activities</i></b>    (includes the General Introduction)

-**Part I: CNN Autoencoder for Valve Defect Detection**  (SNR=+6dB Valve dataset)
-**Part II:  Denoising Algorithms** 












