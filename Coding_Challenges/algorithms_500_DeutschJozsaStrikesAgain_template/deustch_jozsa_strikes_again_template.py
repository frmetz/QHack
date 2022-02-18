#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #

    dev = qml.device("default.qubit", wires=3+3, shots=1)
    wires = range(6)

    ops0 = qml.ctrl(fs[0], control=(3,4))
    ops1 = qml.ctrl(fs[1], control=(3,4))
    ops2 = qml.ctrl(fs[2], control=(3,4))
    ops3 = qml.ctrl(fs[3], control=(3,4))

    @qml.qnode(dev)
    def circuit():
        """Implements the Deutsch Jozsa algorithm."""

        qml.PauliX(wires=2)
        qml.PauliX(wires=5)
        for i in wires:
            qml.Hadamard(wires=i)

        ops0(wires)
        qml.PauliX(wires=3)
        ops1(wires)
        qml.PauliX(wires=3)
        qml.PauliX(wires=4)
        ops2(wires)
        qml.PauliX(wires=3)
        ops3(wires)

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)

        qml.PauliX(wires=0)
        qml.PauliX(wires=1)
        qml.Toffoli(wires=(0,1,5))
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)


        ops3(wires)
        qml.PauliX(wires=3)
        ops2(wires)
        qml.PauliX(wires=3)
        qml.PauliX(wires=4)
        ops1(wires)
        qml.PauliX(wires=3)
        ops0(wires)


        qml.Hadamard(wires=3)
        qml.Hadamard(wires=4)

        return qml.sample(wires=(3,4))

    sample = circuit()


    if np.sum(sample) == 0:
        return "4 same"
    else:
        return "2 and 2"

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
