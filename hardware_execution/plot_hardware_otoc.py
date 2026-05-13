import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from qiskit_ibm_runtime import RuntimeDecoder

def evaluate_empirical_otoc(filepath):
    print(f"Retrieving Local Data: {filepath}...")
    
    with open(filepath, 'r') as f:
        result = json.load(f, cls=RuntimeDecoder)
    
   
   
    pub_results = list(result)
    
    t_max = len(pub_results) - 1
    t_domain = np.arange(0, t_max + 1)
    otoc_vals = np.zeros(t_max + 1)
    
    for i, pub_result in enumerate(pub_results):
        if hasattr(pub_result.data, 'evs_extrapolated'):
            raw_val = pub_result.data.evs_extrapolated
        else:
            raw_val = pub_result.data.evs
            
        extracted_scalar = np.array(raw_val).flatten()[0]
        otoc_vals[i] = np.real(extracted_scalar)
        
    fit_start, fit_end = 2, 10 
    t_fit = t_domain[fit_start:fit_end]
    
    y_fit = np.log(np.abs(otoc_vals[fit_start:fit_end])) 
    
    def linear_model(t, lambda_L, c):
        return lambda_L * t + c
        
    popt, _ = curve_fit(linear_model, t_fit, y_fit)
    lambda_L = popt[0]
    print(f"Empirical Hardware Lyapunov Exponent (lambda_L): {lambda_L:.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(t_domain, otoc_vals, marker='o', linestyle='-', color='black', label='Hardware ZNE')
    plt.xlabel('Time (Trotter Steps)')
    plt.ylabel('OTOC $\\langle V(t) \\rangle$')
    plt.title(f'Hardware-Measured Information Scrambling ($\\lambda_L = {lambda_L:.4f}$)')
    plt.legend()
    plt.grid(True)
    
    output_filename = 'otoc_hardware_evaluation.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"Vector graphics successfully written to: {output_filename}")
    plt.close()


evaluate_empirical_otoc('ibm_fez_otoc_zne.json')
