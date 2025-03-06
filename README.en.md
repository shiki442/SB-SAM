# SB-SAM
### Introduction
**A Score-Based Small Atomic Model with Application to Stress Calculation**

SB-SAM is a molecular dynamics simulation framework based on Score Matching, designed for stress calculation.

### Related Paper
(Please add the paper link here)

### Data Generation
This project uses molecular dynamics (MD) to simulate the particle distribution of an NVT ensemble as training samples.

The data generation code is stored in the ./md directory.

For theoretical background, please refer to "Modeling Materials: Continuum, Atomistic and Multiscale Techniques."

### Model Training
The model is implemented using the PyTorch framework. Configuration parameters are stored in ./config/paramsxx.yml.

To run the training process, use the following command:

```python
python main.py --config=./config/paramsxx.yml