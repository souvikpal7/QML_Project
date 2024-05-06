from copy import copy
import numpy as np
from qiskit.circuit import QuantumCircuit
# from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2, EstimatorOptions
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_ibm_runtime import Estimator, Session, Options


class CustomEstimatorQNN():
    def __init__(
        self,
        circuit: QuantumCircuit,
        service: QiskitRuntimeService,
        backend: IBMBackend,
        resilience_level: int = 1,
        default_shots: int = 1000,
        enable_dynamic_decoupling: bool = True,
        dynamic_decoupling_seq_type="XY4"
    ):
        assert 0 <= resilience_level <= 2, "resilience_level should be from [0,1,2]"
        self._org_circuit = circuit

        observables = SparsePauliOp.from_list([("Z" * circuit.num_qubits, 1)])
        observables = (observables,)
        self._observables = observables

        self._input_params = list(circuit.input_parameters)
        self._weight_params = list(circuit.weight_parameters)

        self.num_inputs = len(self._input_params)
        self.num_weights = len(self._weight_params)
        self._output_shape = (len(self._observables),)
        assert self.num_inputs > 0, "Number of inputs should be more than 0"
        assert self.num_weights > 0, "Number of weights should be more than 0"

        self._circuit = circuit
        self.service = service
        self.backend = backend

        estimator_options = Options()
        estimator_options.resilience_level = resilience_level
        estimator_options.execution.shots = default_shots
        self._estimator_options = estimator_options

        # options = EstimatorOptions()
        # options.resilience_level = resilience_level
        # options.default_shots = default_shots
        # options.dynamical_decoupling.enable = enable_dynamic_decoupling
        # options.dynamical_decoupling.sequence_type = dynamic_decoupling_seq_type
        # self._estimator_options = options

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def circuit(self):
        return copy(self._org_circuit)

    @property
    def observables(self):
        return copy(self._observables)

    def _validate_forward_output(self, output_data, original_shape):
        if original_shape and len(original_shape) >= 2:
            output_data = output_data.reshape((*original_shape[:-1], *self._output_shape))

        return output_data

    def _preprocess_forward(self, input_data, weights):
        num_samples = input_data.shape[0]
        weights = np.broadcast_to(weights, (num_samples, len(weights)))
        parameters = np.concatenate((input_data, weights), axis=1)
        return parameters, num_samples

    def forward(self, input_data, weights):
        input_data = np.array(input_data)
        input_, shape = input_data, input_data.shape
        output_data = self._forward(input_, weights)
        return self._validate_forward_output(output_data, shape)

    def _forward(self, input_data, weights):
        parameter_values_, num_samples = self._preprocess_forward(
            input_data,
            weights
            )

        with Session(service=self.service, backend=self.backend) as session:
            estimator = Estimator(
                backend=self.backend,
                options=self._estimator_options
                )

            job = estimator.run(
                [self._circuit] * num_samples * self.output_shape[0],
                [op for op in self._observables for _ in range(num_samples)],
                np.tile(parameter_values_, (self.output_shape[0], 1)),
            )
            try:
                results = job.result()
            except Exception as exc:
                raise QiskitMachineLearningError("Estimator job failed.") from exc

        results_ = results.values.reshape(num_samples, -1)
        return results_
