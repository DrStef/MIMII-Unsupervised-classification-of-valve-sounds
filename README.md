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
