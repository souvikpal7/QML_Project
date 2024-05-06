from qiskit.circuit import QuantumCircuit, ParameterVector


class AngleEncoding:
    def __init__(self, qubits):
        self.num_qubits = qubits
        self.parameters = ParameterVector("ip", qubits)

    def construct_circuit(self):
        # Create a QuantumCircuit object with 'num_qubits' qubits
        circuit = QuantumCircuit(self.num_qubits)

        for i in range(self.num_qubits):
            circuit.ry(self.parameters[i], i)

        for i in range(self.num_qubits - 1):
            circuit.cx(i, i + 1)

        return circuit


if __name__ == "__main__":
    ang = AngleEncoding(4)
    cir = ang.construct_circuit()
