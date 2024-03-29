# Model Parallelism for Edge Computing

## Overview
This project demonstrates the implementation of model parallelism techniques in machine learning, aimed at optimizing large model performance on edge devices. It involves partitioning ML models and distributing them across different computational resources, such as CPU cores or GPUs. The main focus is on enhancing object detection and inference time in edge computing scenarios.

## Code Samples

### MXNet Model Parallelism

`MXNet.py`:

- **Description**: Implements model parallelism using MXNet, splitting a multi-layer neural network across multiple GPUs. 
- **Key Components**:
  - Multi-layer model split between two GPUs.
  - Forward pass function distributing computation across GPUs.
  - Example training loop demonstrating the model's functionality.

### AlexNet with MXNet on CPU Cores

`MXNet_AlexNet.py`:

- **Description**: A simplified AlexNet model partitioned across different CPU cores using MXNet.
- **Key Components**:
  - AlexNet divided into feature extraction and classifier parts.
  - Initialization of different network parts on separate CPU cores.
  - A hybrid forward function to manage data across these cores.

### TensorFlow Multi-GPU Setup

`tensorflow.py`:

- **Description**: Demonstrates a basic multi-layer model using TensorFlow, manually placing each layer on a different GPU.
- **Key Components**:
  - Layer-wise GPU assignment for model components.
  - Function to create and compile a TensorFlow model.
  - Example of training the model on a simple dataset.

## Implementation Details

- The project applies model parallelism techniques, crucial for handling large models in resource-constrained environments like edge devices.
- It showcases optimal model partitioning, essential for distributing workloads effectively across multiple devices.
- Techniques demonstrated include manual layer assignment to different computational resources, such as GPUs or CPU cores, and managing data flow between these resources.
- The use of MXNet and TensorFlow frameworks highlights different approaches to model parallelism in practical scenarios.

## OpenMP for Parallel Processing in Multicore Processors

In addition to the use of MXNet and TensorFlow for model parallelism, this project also explores the utilization of OpenMP for effective parallel processing in multicore processor environments. The goal is to enhance computational efficiency by parallelizing operations that are otherwise executed serially.

### Experimentation with OpenMP

- **Description**: A part of the project involves experimenting with OpenMP to benchmark its performance in parallel processing scenarios. 
- **Key Components**:
  - **Serial Execution**: Baseline performance measurement with serial computation.
  - **Parallel Execution using OpenMP**: Implementing OpenMP to parallelize the computation and compare it against the serial execution.
  - **Optimized Parallel Execution**: Further optimization in parallel processing using advanced OpenMP techniques.

For a detailed comparison and insights into the experimentation with OpenMP, refer to the following GitHub repository: [Parallel Computing with OpenMP](https://github.com/mdalmaruf/ParallelComputingOpenMP.git)

This component of the project highlights the significance of parallel processing in modern computing environments, especially when dealing with large-scale data and intensive computational tasks. It provides valuable benchmarks and insights into the efficiency gains achievable with parallelization.

## Remarks

The combination of different parallel computing techniques, from model parallelism in machine learning to multicore processing with OpenMP, represents a comprehensive approach to optimizing computational tasks in various scenarios. This project not only demonstrates the practical application of these techniques but also serves as a foundation for future research and development in the field of parallel computing.




## Future Scope

- Explore automated approaches for model partitioning and resource allocation.
- Implement and test the models with larger, more complex datasets to evaluate performance.
- Investigate the integration of these models into real-world edge computing applications.
