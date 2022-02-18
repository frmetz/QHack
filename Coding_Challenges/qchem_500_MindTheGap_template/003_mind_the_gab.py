import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #

    
    qubits = 4
    number_layers = 16

    dev = qml.device("default.qubit", wires=qubits)
    #@jax.jit
    @qml.qnode(dev)
    def circuit_all(parameters):
        for i in range(0, number_layers):
            for j in range (0, 4):
                qml.Hadamard(wires = j)
            qml.CZ(wires = [2,3])
            qml.CZ(wires = [1,2])
            qml.CZ(wires = [0,1])
            for k in range (0, 4):
                qml.RX(parameters[i][k], wires = k)
        return qml.expval(H)
    
    init_params = np.random.rand(number_layers,qubits)
    max_iterations = 150
    conv_tol = 1e-04

    energy = [circuit_all(init_params)]
    angle = [init_params]
    
    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    for n in range(max_iterations):
        init_params, prev_energy = opt.step_and_cost(circuit_all, init_params)

        energy.append(circuit_all(init_params))
        angle.append(init_params)

        conv = np.abs(energy[-1] - prev_energy)

#         if n % 2 == 0:
#             print(f"Step = {n},  Energy = {energy[-1]:.8f}")

        if conv <= conv_tol:
            break

#     print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} ")
#     print (energy[-1])
    
    dev = qml.device("default.qubit", wires=qubits)
    @qml.qnode(dev)
    def circuit_state(parameters):
        for i in range(0, number_layers):
            for j in range (0, 4):
                qml.Hadamard(wires = j)
            qml.CZ(wires = [2,3])
            qml.CZ(wires = [1,2])
            qml.CZ(wires = [0,1])
            for k in range (0, 4):
                qml.RX(parameters[i][k], wires = k)
        return qml.probs(wires = [0,1,2,3])
    
    output = circuit_state(init_params)
    #print (output)
#     output[np.abs(output) < 1e-02] = 0
#     print (output)

    return (energy[-1], output)
    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """

    # QHACK #

    
    Hmat = qml.utils.sparse_hamiltonian(H).real
    
    ground_state[np.abs(ground_state) < 1e-02] = 0
    #print (ground_state)
    
    ground_state1 = np.array([ground_state])
    
    ground_stateT = ground_state1.transpose()
    gmat = np.dot(ground_stateT, ground_state1)
    gmat /= np.linalg.norm(gmat)
    
    #ground state
    fstate = np.array([[0, 0, 0, -0.104, 0, 0, 0, 0 ,   0, 0, 0, 0 ,  0.994, 0 , 0, 0 ]])
    fstateT = fstate.transpose()
    fmat = np.dot(fstateT, fstate)
    fmat /= np.linalg.norm(fmat)
    
    obs_matrix = fmat
    obs = qml.Hermitian(obs_matrix, wires=[0, 1, 2, 3])
    Hg = qml.Hamiltonian((beta, ), (obs, ))
    
    matrix = 0
    for coeff, op in zip(Hg.coeffs, Hg.ops):
        matrix += coeff * op.matrix
    
    Htot = Hmat + matrix
    #print (Htot)
    
    return (Htot)
    
    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """

    # QHACK #

    qubits = 4
    number_layers = 16

    dev = qml.device("default.qubit", wires=qubits)
    @qml.qnode(dev)
    def circuit_tot(parameters):
        for i in range(0, number_layers):
            for j in range (0, 4):
                qml.Hadamard(wires = j)
            qml.CZ(wires = [2,3])
            qml.CZ(wires = [1,2])
            qml.CZ(wires = [0,1])
            for k in range (0, 4):
                qml.RX(parameters[i][k], wires = k)
        return qml.expval(qml.Hermitian(H1, wires = [0,1,2,3]))
    
    init_params = np.random.rand(number_layers,qubits)
    max_iterations = 300
    conv_tol = 1e-04

    energy = [circuit_tot(init_params)]
    angle = [init_params]
    
    opt = qml.GradientDescentOptimizer(stepsize=0.01)

    for n in range(max_iterations):
        init_params, prev_energy = opt.step_and_cost(circuit_tot, init_params)

        energy.append(circuit_tot(init_params))
        angle.append(init_params)

        conv = np.abs(energy[-1] - prev_energy)

#         if n % 2 == 0:
#             print(f"Step = {n},  Energy = {energy[-1]:.8f}")

        if conv <= conv_tol:
            break

#     print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} ")
#     print (energy[-1])
    
    return (energy[-1])
    # QHACK #


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    
    coord = float(sys.stdin.read())
    #coord = 0.6614
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
    print("--- %s seconds ---" % (time.time() - start_time))