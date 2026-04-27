import torch
import torch.nn as nn
import string

class CircuitLayer(nn.Module):
    def __init__(self, features, num_qubits, pos):
        super(CircuitLayer, self).__init__()
        self.num_qubits = num_qubits # number of qubits for logic part
        self.features = features # total number of features
        self.pos = pos # position of this layer in the circuit (0 to features-1)
        self.weights = nn.Parameter(torch.randn((num_qubits, features)) * 0.01) # (num_qubits, features)
        self.bias = nn.Parameter(torch.zeros((num_qubits, )) + 0.5 * 3.14) # (num_qubits, )

        temp_mask = torch.tensor([1 if i < pos else 0 for i in range(features)]).expand(num_qubits, features) # dont see itself
        self.register_buffer('mask', temp_mask)

    def get_thetas(self, x): # x is batched input (batch_size, features) 我们理想只使用前k个features, k为当前输出变量
        # print(x.dtype, self.weights.dtype)
        # x = x.to(self.weights.device)
        
        masked_weights = self.weights * self.mask
        thetas = torch.einsum('nf, bf->bn', masked_weights, x) + self.bias # (batch_size, num_qubits)
        return thetas

    def _batch_gate_matrix(self, thetas, gate_type): # thetas: (batch_size, num_qubits)
        gate_type = gate_type.lower()
        complex_dtype = torch.cfloat if thetas.dtype == torch.float32 else torch.cdouble
        matrices = torch.zeros((thetas.shape[0], self.num_qubits, 2, 2),  dtype=complex_dtype, device=thetas.device)
        cos_t = torch.cos(thetas / 2)
        sin_t = torch.sin(thetas / 2)
        if gate_type == 'rx':
            matrices[:, :, 0, 0] = cos_t
            matrices[:, :, 0, 1] = -1j * sin_t
            matrices[:, :, 1, 0] = -1j * sin_t
            matrices[:, :, 1, 1] = cos_t
        elif gate_type == 'ry':
            matrices[:, :, 0, 0] = cos_t
            matrices[:, :, 0, 1] = -sin_t
            matrices[:, :, 1, 0] = sin_t
            matrices[:, :, 1, 1] = cos_t
        elif gate_type == 'rz':
            matrices[:, :, 0, 0] = torch.exp(-1j * thetas / 2)
            matrices[:, :, 1, 1] = torch.exp(1j * thetas / 2)
        else:
            raise ValueError("Unsupported gate type")
        return matrices

    def forward(self, x, gate_type='rx'):
        thetas = self.get_thetas(x) # (batch_size, num_qubits)
        return self._batch_gate_matrix(thetas, gate_type)
        ### this function returns a batch of unitary matrices for each qubit (batch_size, num_qubits, 2, 2)

class Circuit(nn.Module):
    def __init__(self, features, num_qubits, mydevice='cpu', mydtype=torch.float32, depth=1, reset=False, proj='cx', gate='rx', measure_idx = -1):
        
        super(Circuit, self).__init__()
        self.complex_dtype = torch.cfloat if mydtype == torch.float32 else torch.cdouble
        self.features = features
        self.num_qubits = num_qubits
        self.mydevice = mydevice
        self.mydtype = mydtype
        self.depth = depth
        self.gate = gate.lower()
        self.reset = reset
        self.proj = proj.lower()
        self.measure_idx = measure_idx if measure_idx >=0 else num_qubits -1 # which qubit to measure, default the last logic qubit
        self.x_layers = nn.ModuleList([nn.ModuleList([CircuitLayer(features, num_qubits, pos = f) for i in range(depth)]) for f in range(features)]) # features x depth x_layers
        self.y_layers = nn.ModuleList([nn.ModuleList([CircuitLayer(features, num_qubits, pos = f) for i in range(depth)]) for f in range(features)]) # features x depth y_layers
        
        cx_gate = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=self.complex_dtype, device=mydevice)
        cz_gate = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=self.complex_dtype, device=mydevice)
        sqrt_swap = 0.5 * torch.tensor([[2, 0, 0, 0], [0, 1+1j, 1-1j, 0], [0, 1-1j, 1+1j, 0], [0, 0, 0, 2]], dtype=self.complex_dtype, device=mydevice)
        iswap = torch.tensor([[1, 0, 0, 0],[0, 0, 1j, 0],[0, 1j, 0, 0],[0, 0, 0, 1]], dtype=self.complex_dtype, device=mydevice)
        swap = torch.tensor([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, 1]], dtype=self.complex_dtype, device=mydevice)
        self.register_buffer('cx_gate', cx_gate)
        self.register_buffer('cz_gate', cz_gate)
        self.register_buffer('sqrt_swap', sqrt_swap)
        self.register_buffer('iswap', iswap)
        self.register_buffer('swap', swap)
        print('Num qubits:', num_qubits, 'Add data:', 'Depth:', depth, 'Reset:', reset, 'Projection gate:', proj, 'Gate:', gate, 'Measure idx:', self.measure_idx)
    def _block(self, states, x, features_idx):
        if self.num_qubits == 1: # no entangling gates
            rx_gates = self.x_layers[features_idx][0](x, gate_type='rx') # apply x_layers
            ry_gates = self.y_layers[features_idx][0](x, gate_type='ry') # apply y_layers
            if self.gate == 'rx':
                combine_gates = rx_gates[:, 0]
            elif self.gate == 'ry':
                combine_gates = ry_gates[:, 0]
            elif self.gate == 'rxry':
                combine_gates = ry_gates[:, 0] @ rx_gates[:, 0]
            for d in range(1, self.depth):
                rx_gates = self.x_layers[features_idx][d](x, gate_type='rx') # apply x_layers
                ry_gates = self.y_layers[features_idx][d](x, gate_type='ry') # apply y_layers
                if self.gate == 'rx':
                    combine_gates = rx_gates[:, 0] @ combine_gates
                elif self.gate == 'ry':
                    combine_gates = ry_gates[:, 0] @ combine_gates
                elif self.gate == 'rxry':
                    combine_gates = ry_gates[:, 0] @ rx_gates[:, 0] @ combine_gates
            states = self._apply_gate(states, combine_gates, 0)
            return states
        for d in range(self.depth):
            rx_gates = self.x_layers[features_idx][d](x, gate_type='rx') # apply x_layers
            ry_gates = self.y_layers[features_idx][d](x, gate_type='ry') # apply y_layers
            for q in range(self.num_qubits):
                if self.gate == 'rx':
                    combine_gates = rx_gates[:, q]
                elif self.gate == 'ry':
                    combine_gates = ry_gates[:, q]
                elif self.gate == 'rxry':
                    combine_gates = ry_gates[:, q] @ rx_gates[:, q]
                else:
                    raise ValueError("Unsupported gate type")
                states = self._apply_gate(states, combine_gates, q)
            for q in range(self.num_qubits - 1): # here we do nearest neighbor entangling, can be changed to all-to-all if needed
                control_idx = q
                target_idx = q + 1
                if self.proj == 'swap':
                    states = self._apply_su4(states, control_idx, target_idx, self.swap)
                elif self.proj == 'iswap':
                    states = self._apply_su4(states, control_idx, target_idx, self.iswap)
                elif self.proj == 'sqrt_swap':
                    states = self._apply_su4(states, control_idx, target_idx, self.sqrt_swap)
                elif self.proj == 'cz':
                    states = self._apply_control(states, control_idx, target_idx, type='cz')
                elif self.proj == 'rand':
                    states = self._apply_su4(states, control_idx, target_idx, self.rand_gate)
                else: # default cx
                    states = self._apply_control(states, control_idx, target_idx)
        return states

    def _apply_gate(self, states, unitary, qubit_idx):
        original_shape = states.shape
        states = states.movedim(qubit_idx + 1, 1).reshape(original_shape[0], 2, -1) # move qubit_idx to the front
        states = torch.bmm(unitary, states)
        return states.reshape(original_shape).movedim(1, qubit_idx + 1)

    def _apply_control(self, states, control_idx, target_idx, type='cx'):
        original_shape = states.shape
        batch_size = original_shape[0]
        # Move control and target qubits to the first two dimensions(without considering batch dimension)
        states = states.movedim([control_idx + 1, target_idx + 1], [1, 2]) # plus 1 is considering batch dimension
        states = states.reshape(batch_size, 4, -1)  # (batch_size, 4, rest)
        states = self.cx_gate @ states if type == 'cx' else self.cz_gate @ states
        return states.reshape(original_shape).movedim([1, 2], [control_idx + 1, target_idx + 1])
    
    def _apply_su4(self, states, qubit1_idx, qubit2_idx, gates):
        original_shape = states.shape
        batch_size = original_shape[0]
        # Move control and target qubits to the first two dimensions(without considering batch dimension)
        states = states.movedim([qubit1_idx + 1, qubit2_idx + 1], [1, 2]) # plus 1 is considering batch dimension
        states = states.reshape(batch_size, 4, -1)  # (batch_size, 4, rest)
        states = gates @ states
        return states.reshape(original_shape).movedim([1, 2], [qubit1_idx + 1, qubit2_idx + 1])

    def _get_qubit_probs(self, states, qubit_idx):
        # we must rember the batch dimension, so the qubit_idx is shifted by 1, 总维度为 batch, qubit1, qubit2, ..., qubitn = n+1维
        dims_to_sum = tuple(range(1, qubit_idx + 1)) + tuple((range(qubit_idx + 2, self.num_qubits + 1))) # 1 -- > 
        probs = states.abs()**2 if len(dims_to_sum) == 0 else torch.sum(states.abs()**2, dim=dims_to_sum)
        return probs

    def _collapse(self, states: torch.Tensor, outcomes: torch.Tensor, qubit_idx: int) -> torch.Tensor:

        qubit_dim = qubit_idx + 1
        projection_mask = torch.zeros_like(states)
        
        idx = outcomes.long().view(-1, *([1] * (states.dim() - 1)))
        src = torch.ones_like(idx, dtype=states.dtype)
        projection_mask.scatter_(qubit_dim, idx, src)
        unnormalized_states = states * projection_mask

        norm_dims = list(range(1, states.dim()))
        norms = torch.sqrt(torch.sum(unnormalized_states.abs()**2, dim=norm_dims, keepdim=True))

        collapsed_states = unnormalized_states / (norms + 1e-9) 

        return collapsed_states

    def _initial_state(self, batch_size):
        psi = torch.zeros((batch_size,) + (2,) * self.num_qubits, dtype=self.complex_dtype, device=self.mydevice) # considering ancilla
        psi[(slice(None),) + (0,) * self.num_qubits] = 1.0 # Initialize to |0...0>
        return psi
    
    def forward(self, data):
        '''
        data: (batch_size, features) with values in [0, 1]
        '''
        psi = self._initial_state(data.shape[0])
        data = data.to(dtype=self.mydtype, device=self.mydevice)

        log_probs = 0.
        for i in range(self.features):
            psi = self._block(psi, data, i)
            probs = self._get_qubit_probs(psi, self.measure_idx) # (batch_size, 2)
            data_idx = data[:, i].long().view(-1, 1)
            probs_data = probs.gather(1, data_idx).squeeze(-1)
            log_probs = log_probs + torch.log(probs_data + 1e-9)
            if self.reset:
                psi = self._initial_state(data.shape[0]) # reset all qubits
            else:
                # psi = self._collapse_and_reset_ancilla(psi, data_idx) # just reset ancilla and collapse the rest qubits
                psi = self._collapse(psi, data_idx, self.measure_idx) # just reset ancilla and collapse the rest qubits
            
        return log_probs
    
    @torch.no_grad()
    def sample(self, num_samples):
        psi = self._initial_state(num_samples)

        generated_samples = torch.zeros((num_samples, self.features), device=self.mydevice, dtype=torch.long)
        for i in range(self.features):
            psi = self._block(psi, generated_samples.to(dtype=torch.float32), i)
            probs = self._get_qubit_probs(psi, self.measure_idx) # (batch_size, 2)

            sampled_outcomes = torch.multinomial(probs, num_samples=1) # shape: (num_samples, 1)
            generated_samples[:, i] = sampled_outcomes.squeeze(-1)
            if self.reset:
                psi = self._initial_state(num_samples) # reset all qubits
            else:
                # psi = self._collapse_and_reset_ancilla(psi, sampled_outcomes, self.measure_idx) # just reset ancilla and collapse the rest qubits
                psi = self._collapse(psi, sampled_outcomes, self.measure_idx) # just reset ancilla and collapse the rest qubits
        return generated_samples
        
    def efficient_parameters(self):
        total_params = 0
        for layer in self.x_layers:
            for q_gate in layer:
                total_params += torch.sum(q_gate.mask).item() + 1 # weights + bias
        for layer in self.y_layers:
            for q_gate in layer:
                total_params += torch.sum(q_gate.mask).item() + 1 # weights + bias
        return total_params
