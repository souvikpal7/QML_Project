from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZZFeatureMap


class CustomFeatureEmd:
    supported_encodings = ["zz", "angle"]

    def __init__(self, input_dims, encoding):
        self.input_dims = input_dims
        assert encoding in CustomFeatureEmd.supported_encodings
        self.encoding = encoding
        if self.encoding in ["zz", "angle"]:
            self.output_dims = input_dims

    def get_feturemap(self):
        if self.encoding == "angle":
            circuit = self._create_angle_emd()
            return circuit
        elif self.encoding == "zz":
            return ZZFeatureMap(self.output_dims)
        else:
            raise RuntimeError(f"{self.supported_encodings} not expected")

    def _create_angle_emd(self):
        circuit = QuantumCircuit(self.output_dims)
        parameters = ParameterVector("x", self.input_dims)
        for i in range(self.output_dims):
            circuit.ry(parameters[i], i)

        for i in range(self.output_dims - 1):
            circuit.cx(i, i + 1)
        return circuit
