import numpy as np
import matplotlib.pyplot as plt
import json

def extract_entropy_metrics(filepath):
    print(f"Retrieving Local Data: {filepath}...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)

    time_steps = data["scrambling_time_steps"]
    t_max = len(time_steps) - 1
    t_domain = np.arange(0, t_max + 1)
    
    min_entropies = np.zeros(t_max + 1)
    shannon_entropies = np.zeros(t_max + 1)
    
    print("\n--- Entropy Production Log ---")
    for step in time_steps:
        i = step["t"]
        counts = step["counts"]
        total_shots = sum(counts.values())
        
        
        probs = np.array([count / total_shots for count in counts.values()])
        
       
        max_p = np.max(probs)
        h_min = -np.log2(max_p) if max_p > 0 else 0
        
       
        h_shannon = -np.sum(probs * np.log2(probs + 1e-15)) 
        
        min_entropies[i] = h_min
        shannon_entropies[i] = h_shannon
        
        print(f"t={i:02d} | H_min: {h_min:.4f} bits | H_shannon: {h_shannon:.4f} bits")

   
    ideal_entropy = 4.0

   
    plt.figure(figsize=(9, 6))
    plt.plot(t_domain, shannon_entropies, marker='o', linestyle='-', color='black', label='Shannon Entropy ($S$)')
    plt.plot(t_domain, min_entropies, marker='s', linestyle='--', color='gray', label='Min-Entropy ($H_\\infty$)')
    plt.axhline(y=ideal_entropy, color='red', linestyle=':', label='Theoretical Maximum (4.0 bits)')
    
    plt.xlabel('Time (Trotter Steps)')
    plt.ylabel('Entropy (Bits)')
    plt.title('Cryptographic Entropy Production via Open Quantum Chaos')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_filename = 'entropy_generation_hardware.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"\nVector graphics successfully written to: {output_filename}")
    plt.close()


extract_entropy_metrics('ibm_fez_raw_counts.json')
