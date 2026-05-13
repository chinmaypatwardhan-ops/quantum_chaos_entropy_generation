import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.synthesis import LieTrotter
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ---------------------------------------------------------
# 1. System Parameters (N = 8 Qubits)
# ---------------------------------------------------------
N_DATA = 4
N_ANC = 4
N_TOTAL = N_DATA + N_ANC

dt = 0.1
alpha = np.pi / 2
kappa = 5.0
g = 1.5
gamma = 0.25  # Lindblad dissipation rate

# ---------------------------------------------------------
# 2. Unitary Floquet Step Assembly
# ---------------------------------------------------------
Jy_A = SparsePauliOp.from_list([("IIIY", 0.5), ("IIYI", 0.5)])
Jz_A = SparsePauliOp.from_list([("IIIZ", 0.5), ("IIZI", 0.5)])
p_B  = SparsePauliOp.from_list([("ZIII", 0.5), ("IZII", 0.5)])

H_cont = (alpha / 2.0) * (Jy_A @ Jy_A) + (kappa / 2.0) * Jz_A + g * (Jz_A @ p_B)

c1 = np.sqrt(2) / 4
cat_matrix = np.array([
    [ c1*(1-1j),  0.5+0.0j, c1*(-1+1j),  0.5+0.0j ],
    [ c1*(1+1j),  0.5+0.0j,  c1*(1+1j), -0.5+0.0j ],
    [ c1*(1-1j), -0.5+0.0j, c1*(-1+1j), -0.5+0.0j ],
    [ c1*(1+1j), -0.5+0.0j,  c1*(1+1j),  0.5+0.0j ]
], dtype=complex)

U_cat_gate = UnitaryGate(cat_matrix, label="U_Cat")

evol_gate = PauliEvolutionGate(H_cont, time=dt)
trotterized_cont = LieTrotter(reps=1).synthesize(evol_gate)

step_circ = QuantumCircuit(N_DATA, name="U_step")
step_circ.compose(trotterized_cont, inplace=True)
step_circ.append(U_cat_gate, [2, 3])
U_step = step_circ.to_gate()

# ---------------------------------------------------------
# 3. Lindblad Dissipation Function
# ---------------------------------------------------------
def apply_lindblad_dephasing(qc, data_indices, ancilla_indices, gamma_rate, dt_step):
    """Injects non-unitary environmental decoherence via ancilla reset."""
    p = 1 - np.exp(-gamma_rate * dt_step)
    theta = 2 * np.arcsin(np.sqrt(p))
    
    for d_idx, a_idx in zip(data_indices, ancilla_indices):
        qc.ry(theta, a_idx)
        qc.cz(a_idx, d_idx)
        qc.reset(a_idx)

# ---------------------------------------------------------
# 4. Open System Circuit Generation
# ---------------------------------------------------------
t_max = 10
circuits = []
data_qubits = [0, 1, 2, 3]
ancilla_qubits = [4, 5, 6, 7]

for t in range(t_max + 1):
    cr = ClassicalRegister(N_DATA, 'meas')
    qc = QuantumCircuit(N_TOTAL)
    qc.add_register(cr)
    
    # Initialize data register in |+> to maximize initial purity
    qc.h(data_qubits)
    
    for _ in range(t):
        # 1. Coherent Chaotic Evolution
        qc.append(U_step, data_qubits)
        # 2. Incoherent Environmental Dissipation
        apply_lindblad_dephasing(qc, data_qubits, ancilla_qubits, gamma, dt)
        
    # Measure Data Qubits
    qc.measure(data_qubits, cr)
    circuits.append(qc)

# ---------------------------------------------------------
# 5. Hardware Deployment via SamplerV2
# ---------------------------------------------------------
def deploy_entropy_sampler():
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=100)
    print(f"Acquired Backend: {backend.name}")

    print("Executing Level 3 Transpilation (Synchronous mapping)...")
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    
    isa_circuits = []
    for circ in circuits:
        isa_circuits.append(pm.run(circ))

    sampler = SamplerV2(mode=backend)
    
    # Set shots high to ensure statistical validity for probability distribution
    print(f"Dispatching payload of {len(circuits)} circuits for bitstring sampling...")
    job = sampler.run(isa_circuits, shots=10000)
    print(f"Job ID: {job.job_id()}")
    
    return job

execution_job = deploy_entropy_sampler()
