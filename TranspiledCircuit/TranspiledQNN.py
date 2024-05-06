from Ansatzs.anstaz_19 import Ansatz19
from Embeddings.Custom_Embedding import CustomFeatureEmd
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit import transpile


class TranspiledQNN:
    def __init__(self, input_dim, encoding_scheme, ansatz_name, ansatz_repetition, backend):
        assert isinstance(ansatz_repetition, int)
        assert isinstance(input_dim, int)
        assert isinstance(encoding_scheme, str)
        assert isinstance(ansatz_name, str)

        self.feature_map = CustomFeatureEmd(input_dim, encoding_scheme).get_feturemap()
        n_qubit = self.feature_map.num_qubits
        if ansatz_name.lower().startswith("ansatz"):
            # custom ansatz should be of name format ansatz<I>, where I is ansatz id
            ansatz_id = int(ansatz_name[6:])
            self.ansatz = Ansatz19(ansatz_repetition, n_qubit, ansatz_id).get_ansatz()
        elif ansatz_name.lower() == "realamplitude":
            self.ansatz = RealAmplitudes(n_qubit, reps=ansatz_repetition)
        else:
            raise RuntimeError(f"{ansatz_name} currently not supported")

        self.qnn = QNNCircuit(
            feature_map=self.feature_map,
            ansatz=self.ansatz
            )
        self.transpiled_qc = transpile(self.qnn, backend)

        ip_params = []
        wei_params = []
        for param in self.transpiled_qc.parameters:
            if param.name.startswith('x'):
                ip_params.append(param)
            else:
                wei_params.append(param)

        self.transpiled_qc.input_parameters = ip_params
        self.transpiled_qc.weight_parameters = wei_params
        self.layout = self.transpiled_qc.layout

    def get_circuit(self):
        return self.transpiled_qc

    def get_circuit_layout(self):
        return self.layout


if __name__ == "__main__":
    trans_crc = TranspiledQNN(4, "zz", "RealAmplitude", 2, backend)