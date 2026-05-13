# quantum_chaos_entropy_generation
# Generating Entropy via Quantum Chaos in Open Quantum Stochastic Walks

This repository contains the empirical hardware telemetry, exact state-vector simulations, and classical post-processing architecture for the manuscript: **"Generating Entropy via Quantum Chaos in Open Quantum Stochastic Walks for Post-Quantum Cryptographic Randomness."**

The framework demonstrates the extraction of cryptographic randomness from a composite chaotic Hamiltonian (Kicked Top coupled with Arnold's Cat Map) and establishes the strict necessity of classical privacy amplification (Leftover Hash Lemma) to overcome NISQ-era decoherence.

## Prerequisites & Installation

To reproduce the simulations and hardware evaluations, ensure you have an environment with GPU acceleration (CUDA/ROCm) for the PyTorch state-vector integrations.

```bash
pip install qiskit qiskit-ibm-runtime torch numpy scipy matplotlib
