import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.synthesis import LieTrotter
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, EstimatorOptions
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ---------------------------------------------------------
# 1. System Parameters & Operator Definitions
# ---------------------------------------------------------
N_A = 2
N_B = 2
N_TOTAL = N_A + N_B
dt = 0.1
alpha = np.pi / 2
kappa = 5.0
g = 1.5

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

# ---------------------------------------------------------
# 2. Floquet Step Assembly (Forward and Backward)
# ---------------------------------------------------------
evol_gate = PauliEvolutionGate(H_cont, time=dt)
trotterized_cont = LieTrotter(reps=1).synthesize(evol_gate)

step_circ = QuantumCircuit(N_TOTAL, name="U_step")
step_circ.compose(trotterized_cont, inplace=True)
step_circ.append(U_cat_gate, [2, 3])
U_step = step_circ.to_gate()
U_step_dg = step_circ.inverse().to_gate(label="U_step_dg")

# ---------------------------------------------------------
# 3. Time-Domain Circuit Generation (t = 0 to 10)
# ---------------------------------------------------------
# W = Z on Qubit 0, V = X on Qubit 2
V_op = SparsePauliOp.from_list([("IXII", 1.0)])

t_max = 10
circuits = []

for t in range(t_max + 1):
    qc = QuantumCircuit(N_TOTAL)
    # Initialize in |+> state (eigenstate of V observable X)
    qc.h(range(N_TOTAL))
    
    # Forward Evolution
    for _ in range(t):
        qc.append(U_step, [0, 1, 2, 3])
        
    # Apply W Perturbation (Z on Qubit 0)
    qc.z(0)
    
    # Backward Evolution
    for _ in range(t):
        qc.append(U_step_dg, [0, 1, 2, 3])
        
    circuits.append(qc)

# ---------------------------------------------------------
# 4. Hardware Deployment & Error Mitigation
# ---------------------------------------------------------
def deploy_otoc_batch():
    service = QiskitRuntimeService()
    
    # Target >= 100 qubit operational processor
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=100)
    print(f"Acquired Backend: {backend.name}")

    # Aggressive Level 3 Transpilation
    print("Executing Level 3 Transpilation (Synchronous mapping to bypass forkserver crash)...")
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    
    # Force single-process execution to maintain thread stability
    isa_circuits = []
    for circ in circuits:
        isa_circuits.append(pm.run(circ))
    
    # Map observable to ISA layouts with automatic physical qubit padding
    pub_list = []
    for circ in isa_circuits:
        mapped_V_op = V_op.apply_layout(circ.layout)
        pub_list.append((circ, mapped_V_op))

    # Configuration: V2 Primitive standard
    options = EstimatorOptions()
    options.resilience.zne_mitigation = True # V2 syntax for Resilience Level 2
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XX"
    
    estimator = EstimatorV2(mode=backend, options=options)
    
    print(f"Dispatching payload of {len(circuits)} circuits...")
    job = estimator.run(pub_list)
    print(f"Job ID: {job.job_id()}")
    
    return job

execution_job = deploy_otoc_batch()
print("Execution sequence engaged. Await hardware processing.")
