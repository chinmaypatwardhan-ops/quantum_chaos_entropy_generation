import torch
import numpy as np
import time
import scipy.sparse as sp
import random

# 1. Hardware Verification
if not torch.cuda.is_available():
    raise SystemError("Fatal: GPU detached. Check Colab Runtime settings.")
device = torch.device('cuda')

# 2. Topologic Parameters (Exact Composite Model)
N_A = 8
N_B = 8
N = N_A + N_B
dim = 2**N
M_TRAJECTORIES = 100
DT = 0.05
T_TOTAL = 10.0

omega = 0.25
gamma = 0.25
alpha = np.pi / 2
kappa = 5.0
g = 1.5

print(f"Initiating Exact State-Vector MCWF: N={N}, Trajectories={M_TRAJECTORIES}")
print("-" * 50)

# 3. Sparse Operator Construction
def pauli_basis(op, idx, total_N):
    res = sp.eye(1)
    for i in range(total_N):
        res = sp.kron(res, op if i == idx else sp.eye(2), format='csr')
    return res

Y = sp.csr_matrix([[0, -1j], [1j, 0]])
Z = sp.csr_matrix([[1, 0], [0, -1]])
P1 = sp.csr_matrix([[0, 0], [0, 1]])

Jy_A = sum(pauli_basis(Y, i, N_A) for i in range(N_A)) * 0.5
Jz_A = sum(pauli_basis(Z, i, N_A) for i in range(N_A)) * 0.5
p_B = sum(pauli_basis(Z, i, N_B) for i in range(N_B)) * 0.5

print("Synthesizing Composite Hamiltonian...")
H_KT = (alpha / 2.0) * (Jy_A @ Jy_A) + (kappa / 2.0) * Jz_A
H_int = g * sp.kron(Jz_A, p_B, format='csr')
H_comp = sp.kron(H_KT, sp.eye(2**N_B), format='csr') + H_int

dissipation = sum(pauli_basis(P1, i, N) for i in range(N))
H_eff_sp = (1.0 - omega) * H_comp - 0.5j * omega * gamma * dissipation

H_eff_coo = H_eff_sp.tocoo()
indices = torch.tensor(np.vstack((H_eff_coo.row, H_eff_coo.col)), dtype=torch.long)
values = torch.tensor(H_eff_coo.data, dtype=torch.complex128)
H_eff = torch.sparse_coo_tensor(indices, values, torch.Size(H_eff_coo.shape)).to(device).to_sparse_csr()

print("Synthesizing Discrete Cat Map Unitary...")
cat_matrix = sp.dok_matrix((256, 256), dtype=np.complex128)
for q in range(16):
    for p in range(16):
        q_new = (q + p) % 16
        p_new = (q + 2*p) % 16
        old_idx = q * 16 + p
        new_idx = q_new * 16 + p_new
        cat_matrix[new_idx, old_idx] = 1.0

U_cat_sp = sp.kron(sp.eye(2**N_A), cat_matrix.tocsr(), format='coo')
U_indices = torch.tensor(np.vstack((U_cat_sp.row, U_cat_sp.col)), dtype=torch.long)
U_values = torch.tensor(U_cat_sp.data, dtype=torch.complex128)
U_cat = torch.sparse_coo_tensor(U_indices, U_values, torch.Size(U_cat_sp.shape)).to(device).to_sparse_csr()

# 4. Trajectory Batching (Symmetry-Broken Initialization)
print("Injecting symmetry-breaking thermal noise...")
initial_state = torch.ones(dim, dtype=torch.complex128, device=device)
noise_r = torch.randn(dim, dtype=torch.float64, device=device) * 1e-5
noise_i = torch.randn(dim, dtype=torch.float64, device=device) * 1e-5
initial_state += torch.complex(noise_r, noise_i)
initial_state = initial_state / torch.sqrt(torch.sum(torch.abs(initial_state)**2))
psi = initial_state.unsqueeze(1).repeat(1, M_TRAJECTORIES)

times = np.arange(0, T_TOTAL + DT, DT)
S_vn_log = np.zeros((M_TRAJECTORIES, len(times)))
H_inf_log = np.zeros((M_TRAJECTORIES, len(times)))

start_time = time.time()

def rk4_step(state, dt):
    k1 = -1j * torch.matmul(H_eff, state)
    k2 = -1j * torch.matmul(H_eff, state + 0.5 * dt * k1)
    k3 = -1j * torch.matmul(H_eff, state + 0.5 * dt * k2)
    k4 = -1j * torch.matmul(H_eff, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

print("Commencing Exact Markovian Integration...")
for i, t in enumerate(times):
    if t > 0.0:
        # A. Continuous Non-Hermitian Precession
        psi = rk4_step(psi, DT)

        # B. Discrete Scrambling
        if np.isclose(t % 1.0, 0.0) or np.isclose(t % 1.0, 1.0):
            psi = torch.matmul(U_cat, psi)

        # C. Stochastic Jump Evaluation
        norms = torch.sum(torch.abs(psi)**2, dim=0)
        dp = 1.0 - norms
        rands = torch.rand(M_TRAJECTORIES, device=device, dtype=torch.float64)
        jump_mask = dp > rands

        if jump_mask.any():
            jump_indices = jump_mask.nonzero(as_tuple=True)[0]
            for j_idx in jump_indices:
                psi_j = psi[:, j_idx].view([2]*N)
                valid_jump = False
                attempts = 0
                while not valid_jump and attempts < N * 2:
                    # CPU-native random to prevent PCI-e choke
                    q = random.randint(0, N - 1)
                    idx_1 = [slice(None)] * N; idx_1[q] = 1

                    pop = torch.sum(torch.abs(psi_j[tuple(idx_1)])**2).item()
                    if pop > 1e-10:
                        valid_jump = True
                        idx_0 = [slice(None)] * N; idx_0[q] = 0
                        psi_j[tuple(idx_0)] = psi_j[tuple(idx_1)].clone()
                        psi_j[tuple(idx_1)] = 0.0
                        psi[:, j_idx] = psi_j.view(-1)
                    attempts += 1

        # D. Safe Wavefunction Renormalization
        current_norms = torch.sqrt(torch.sum(torch.abs(psi)**2, dim=0))
        current_norms[current_norms < 1e-15] = 1.0
        psi = psi / current_norms

    # 5. Batched Bipartite Entanglement Extraction
    psi_reshaped = psi.t().view(M_TRAJECTORIES, 2**N_A, 2**N_B)
    S = torch.linalg.svdvals(psi_reshaped)

    probs = S**2
    p_max = torch.max(probs, dim=1).values
    S_vn = -torch.sum(probs * torch.log2(probs + 1e-15), dim=1)
    H_inf = -torch.log2(p_max)

    S_vn_log[:, i] = S_vn.cpu().numpy()
    H_inf_log[:, i] = H_inf.cpu().numpy()

    # Increased telemetry output for active monitoring
    if i % 4 == 0:
        print(f"  t={t:05.2f} | Avg S_vn: {torch.mean(S_vn):.4f} | Avg H_inf: {torch.mean(H_inf):.4f} bits")

avg_S_vn = np.mean(S_vn_log, axis=0)
avg_H_inf = np.mean(H_inf_log, axis=0)

print("-" * 50)
print(f"Simulation Complete. Wall Clock Time: {time.time() - start_time:.2f} seconds")
print(f"Terminal Deficit Extracted: {avg_H_inf[-1]:.4f} bits")

np.save('tebd_entropy_data.npy', np.array([times, avg_S_vn, avg_H_inf], dtype=object))
