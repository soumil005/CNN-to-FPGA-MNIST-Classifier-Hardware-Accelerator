# CNN-to-FPGA-MNIST-Classifier-Hardware-Accelerator
Deploying a Keras CNN for MNIST classification onto FPGA using hls4ml and Vitis HLS, covers model training, HLS conversion, fixed-point quantization, and Vivado RTL simulation on Xilinx Artix-7.
# Overview
This project demonstrates the full ML-to-hardware pipeline:
  1. **Train** a compact CNN in Keras/TensorFlow on a downscaled (8×8) MNIST dataset
  2. **Convert** the trained model to synthesisable HLS C++ using hls4ml
  3. **Synthesise** the HLS design in Vitis HLS with fixed-point quantization and pragma optimisations
  4. **Simulate** the generated RTL in Xilinx Vivado and verify outputs against the original Keras model

The model is intentionally tiny (1,722 parameters) to demonstrate FPGA-friendly, resource-constrained neural network deployment — the kind of design philosophy relevant to edge inference accelerators.

# Model Architecture
Input: 8×8×1 (MNIST images downscaled from 28×28)

<img width="583" height="344" alt="image" src="https://github.com/user-attachments/assets/7bdabce1-145e-4a5f-a2c0-aa6890d040ae" />

Total params: **1,722 (6.73 KB)**

Validation accuracy: ~89–90% (30 epochs, Adam, sparse categorical crossentropy)

# HLS Synthesis Results
Target device: **Xilinx Artix-7 (xc7a200t-fbg676-2)**
Tool: Vitis HLS 2023.1

<img width="758" height="540" alt="image" src="https://github.com/user-attachments/assets/bacd935f-f90b-431c-8071-0148a167a11f" />

# Optimisations Applied
* Fixed-point quantization: ap_fixed<16,6> throughout all layers, reduces resource usage vs. floating-point while preserving model accuracy
* Reuse factor: configured in hls4ml to trade off latency against DSP/LUT utilisation
* Dataflow / pipeline pragmas: applied to enable concurrent layer execution and improve throughput
* Input downscaling: 28×28 → 8×8 reduces input fan-in and convolutional compute significantly without a prohibitive accuracy penalty

# Requirements
Python          >= 3.8
TensorFlow      >= 2.x
tf-keras
hls4ml
Vitis HLS       2023.1
Xilinx Vivado   2023.1

Install Python Dependencies:

  a.	pip install tensorflow==2.13.0
  
  b.	pip install hls4ml[tf]==0.8.1
  
  c.	pip install numpy
  
  d.	pip install tf-keras 

# How to Run
  1. Train the model:

     python model/train.py
     
     #Outputs: small_cnn.h5, x_test_sample.npy, y_test_sample.npy
  2. Convert to HLS
     
     python hls4ml/convert.py
     
     #Generates: myproject_prj/ with HLS C++ sources
  3. Run C-synthesis in Vitis HLS
 
     Open myproject_prj in Vitis HLS 2023.1, set target device to xc7a200t-fbg676-2, and run C Synthesis. Review myproject_csynth.rpt for timing       and utilisation.
     
  4. Export IP and simulate in Vivado
     
     Export the synthesised design as a Vivado IP, instantiate it in a Vivado project, and run RTL Simulation to verify hardware outputs against       expected Keras model predictions. 

# Key Takeaways

  * hls4ml dramatically lowers the barrier to FPGA inference — a trained Keras model becomes synthesisable RTL in ~10 lines of Python
  * Fixed-point quantization (ap_fixed<16,6>) achieves a practical balance: negligible accuracy loss vs. the floating-point baseline, with       significant resource savings
  * The Artix-7 device has ample headroom (~88% LUT, ~88% DSP remaining), leaving room for larger networks or multi-instance parallelism
  * AXI-Stream interfacing makes the IP core composable with standard Vivado block design flows

# References

  * [hls4ml documentation](https://fastmachinelearning.org/hls4ml/)
  * [Vitis HLS User Guide (UG1399)](https://docs.amd.com/r/en-US/ug1399-vitis-hls)
  * [Vivado Design Suite User Guide](https://docs.amd.com/r/en-US/ug910-vivado-getting-started/What-is-the-Vivado-Design-Suite)
  * Duarte et al., Fast inference of deep neural networks in FPGAs for particle physics, JINST 2018


