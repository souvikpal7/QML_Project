# Variational Quantum Classifier using EstimatorQNN (Implemented from scratch)

This Repository contains implementation of **Variational Quantum Classifier** using **EstimatorQNN** from scratch with Qiskit.

We have implemented 19 different Ansatz mentioned in [this paper](https://arxiv.org/abs/1905.10876). Alongside that we also implemented Angle Encoding. 

Given a dataset a user have choice of 
- 2 encoding (Angle Encoding and ZZ featuremap)
- 19 Custom Ansatz and RealAmplitude Ansatz
- Error Mitigation for EstimatorQNN
- Transpiled quantum circuit given a backend
- Flexible choice of Depth for each Ansatz
- Entangling Capacity of 19 Ansatz
- Results from all the circuits in a single place Results.csv
- currenly COBYLA is only supported as optimizer
- For COBYLA maximum iteration = number of weight parameters of Ansatz  x 3
#### Used Dataset
We have used [Rice Classification dataset](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) from UCI Machine Learning Repository. The dataset is pre-pocessed  and stored into .npy files for convenience

## 1. Introduction to Different Folders

### 1.1 Notebooks

The most important folder from a users perspective. 
- The 19 ansatz VQC execution for our dataset are made available in Ansatz\<k\>_execution.ipynb here k is the k-th Ansatz in the mentioned research paper
- RealAmplitude Notebooks are also made available is similar format
- Classical machine learning model is also trained on out dataset and it is available in Classical_ML.ipynb

Every ansatz is executed with 3 configurations.
1. Angle Encoding with depth 2
2. Angle Encoding with depth 5
3. ZZ Encoding with depth 5

Further variations user can add modifying the below code (which is available on all notebooks) 
```python
ANSATZ_NAME  =  "ansatz7"
ENCODING  = ["angle", "angle", "zz"]
DEPTH  = [2, 5, 5]
```
### 1.2 CustomQNN
We have implemented a class CustomEstimatorQNN which works as [EstimatorQNN](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html).

### 1.3 TranspiledCircuit
We have encapsulated creation of different quantum circuits in a single Class. It takes care of different encodings and ansatz, depth and also transpilation w.r.t the provided backend