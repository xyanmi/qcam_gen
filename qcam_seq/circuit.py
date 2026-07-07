import torch
import torch.nn as nn

class CircuitLayer(nn.Module):
    def __init__(self, length, num_qubits, pos):
        '''
        data is sequence 
        (batch, seq_len, num_qubits)
        so, for each position, we have a circuit layer that need num_qubits gates
        '''
        super(CircuitLayer, self).__init__()
        self.length = length
        self.num_qubits = num_qubits
        self.pos = pos
        self.weights = nn.Parameter(torch.randn(num_qubits, length, num_qubits) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_qubits,) + 0.5 * 3.1415926)
        mask = torch.tensor([1 if i < pos else 0 for i in range(length)]).unsqueeze(1).unsqueeze(0).expand(num_qubits, length, num_qubits)
        # mask = torch.transpose(mask, dim0=1, dim1=0).reshape(num_qubits, length, num_qubits)
        self.register_buffer('mask', mask)
    
    def get_thetas(self, x): # x is (batch, seq_len, num_qubits) we just use the pos-th element
        x = x.to(device=self.weights.device, dtype=self.weights.dtype)
        masked_weights = self.weights * self.mask
        thetas = torch.einsum('nlq, blq->bn', masked_weights, x) + self.bias
        # thetas = torch.tanh(thetas) * torch.pi
        return thetas # (batch, num_qubits)
    
    def _batch_gate_matrix(self, thetas, gate_type):
        gate_type = gate_type.lower()
        complex_dtype = torch.complex64 if thetas.dtype == torch.float32 else torch.complex128
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
        thetas = self.get_thetas(x) # (batch, num_qubits)
        # print(self.pos, self.mask.sum())
        return self._batch_gate_matrix(thetas, gate_type) # (batch, num_qubits, 2, 2)
        # this function returns the gate matrices for each qubit in the batch

class Circuit(nn.Module):
    def __init__(self, length, num_qubits, mydevice='cpu', mydtype=torch.float32, depth=1, reset=False, proj='cx', gate='rx'):
        super(Circuit, self).__init__()
        self.length = length
        self.num_qubits = num_qubits
        self.mydevice = mydevice
        self.mydtype = mydtype
        self.depth = depth
        self.reset = reset
        self.proj = proj.lower()
        self.gate = gate.lower()
        self.complex_dtype = torch.cfloat if mydtype == torch.float32 else torch.cdouble
        self.x_layers = nn.ModuleList([nn.ModuleList([CircuitLayer(length, num_qubits, pos=f) for i in range(depth)]) for f in range(length)])
        self.y_layers = nn.ModuleList([nn.ModuleList([CircuitLayer(length, num_qubits, pos=f) for i in range(depth)]) for f in range(length)])

        cx_gate = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=self.complex_dtype, device=mydevice)
        self.register_buffer('cx_gate', cx_gate)
        print(f'Circuit initialized with length={length}, num_qubits={num_qubits}, depth={depth}, proj={proj}, gate={gate}')

    def efficient_parameters(self):
        total_params = 0
        for layer in self.x_layers:
            for q_gate in layer:
                total_params += torch.sum(q_gate.mask).item() + 1 # weights + bias
        for layer in self.y_layers:
            for q_gate in layer:
                total_params += torch.sum(q_gate.mask).item() + 1 # weights + bias
        return total_params
    def _block(self, states, x, seq_idx):
        for d in range(self.depth):
            rx_gates = self.x_layers[seq_idx][d](x, gate_type='rx')
            ry_gates = self.y_layers[seq_idx][d](x, gate_type='ry')
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
            ### apply controlled gates
            for q in range(self.num_qubits - 1):
                if self.proj == 'cx' or self.proj == 'cnot':
                    states = self._apply_control(states, q, q + 1)
        
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
    
    def _get_all_probs(self, states):
        return states.abs()**2 #(b, 2, 2, 2, ...)
    
    def _select_probs(self, probs, data): # states (b, 2, 2, ..., 2), data (b, num_qubits)
        batch_size = probs.shape[0]
        batch_indices = torch.arange(batch_size, device=probs.device)
        indices = (batch_indices,) + tuple(data[:, i] for i in range(self.num_qubits))
        return probs[indices] #(b,)

    # def _collapse(self, states: torch.Tensor, outcomes: torch.Tensor, qubit_idx: int) -> torch.Tensor:

    #     qubit_dim = qubit_idx + 1
    #     projection_mask = torch.zeros_like(states)
        
    #     idx = outcomes.long().view(-1, *([1] * (states.dim() - 1)))
    #     src = torch.ones_like(idx, dtype=states.dtype)
    #     projection_mask.scatter_(qubit_dim, idx, src)
    #     unnormalized_states = states * projection_mask

    #     norm_dims = list(range(1, states.dim()))
    #     norms = torch.sqrt(torch.sum(unnormalized_states.abs()**2, dim=norm_dims, keepdim=True))

    #     collapsed_states = unnormalized_states / (norms + 1e-9) 

    #     return collapsed_states
    def _initial_state(self, batch_size):
        psi = torch.zeros((batch_size,) + (2,) * self.num_qubits, dtype=self.complex_dtype, device=self.mydevice) # considering ancilla
        psi[(slice(None),) + (0,) * self.num_qubits] = 1.0 # Initialize to |0...0>
        return psi
    
    # def _collapse(self, states: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
    #     psi = torch.zeros_like(states)
    #     batch_size = states.shape[0]
    #     batch_indices = torch.arange(batch_size, device=states.device)
    #     indices = (batch_indices,) + tuple(outcomes[:, i] for i in range(self.num_qubits))
    #     psi[indices] = 1.0
    #     return psi
    def _collapse(self, states: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        # outcomes shape: (b, num_qubits)
        batch_size = states.shape[0]
        
        projection_mask = torch.zeros_like(states, dtype=torch.bool)

        batch_indices = torch.arange(batch_size, device=states.device)
        indices = (batch_indices,) + tuple(outcomes[:, i] for i in range(self.num_qubits))
        projection_mask[indices] = True
        
        unnormalized_states = states * projection_mask.to(states.dtype)

        # sum over all qubit dimensions (dims 1 to n)
        norm_dims = list(range(1, states.dim()))
        norms = torch.sqrt(torch.sum(unnormalized_states.abs()**2, dim=norm_dims, keepdim=True))
        collapsed_states = unnormalized_states / (norms + 1e-12)
        
        return collapsed_states
    
    def forward(self, data, mask=None, condition=False):
        '''
        data is (batch, seq_len, num_qubits)
        '''
        if mask is None: mask = torch.ones((data.shape[0], data.shape[1]), device=data.device, dtype=data.dtype)
        if condition: mask[:, 0] = 0  # do not consider the first token
        psi = self._initial_state(data.shape[0])
        log_probs = 0.
        for i in range(self.length):
            psi = self._block(psi, data, i)
            probs = self._get_all_probs(psi) # (b, 2, 2, ..., 2)
            data_int = data[:, i].long() # (b, num_qubits)
            selected_probs = self._select_probs(probs, data_int) # (b,)
            log_probs = log_probs + torch.log(selected_probs + 1e-9) * mask[:, i]
            if self.reset:
                psi = self._initial_state(data.shape[0])
            else:
                psi = self._collapse(psi, data_int)
        return log_probs
    
    @torch.no_grad()
    def sample(self, num_samples):
        psi = self._initial_state(num_samples)
        samples = torch.zeros((num_samples, self.length, self.num_qubits), dtype=torch.long, device=self.mydevice)
        for i in range(self.length):
            psi = self._block(psi, samples, i)
            probs = self._get_all_probs(psi) # (b, 2, 2, ..., 2)
            probs = probs.reshape(num_samples, -1) # (b, 2**num_qubits)
            sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(1) # (b,)
            for q in range(self.num_qubits):
                samples[:, i, q] = (sampled_indices >> (self.num_qubits - 1 - q)) & 1
            if self.reset:
                psi = self._initial_state(num_samples)
            else:
                psi = self._collapse(psi, samples[:, i])
        return samples
            
    @torch.no_grad()
    def cond_sample(self, prompt, num_samples=1):
        '''
        prompt is (prompt_len, num_qubits)
        '''
        prompt = prompt.to(device=self.mydevice, dtype=self.mydtype)
        prompt = prompt.unsqueeze(0).expand(num_samples, -1, -1) # (b, prompt_len, num_qubits)
        psi = self._initial_state(num_samples)
        samples = torch.zeros((num_samples, self.length, self.num_qubits), dtype=torch.long, device=self.mydevice)
        samples[:, :prompt.shape[1], :] = prompt
        for i in range(self.length):
            psi = self._block(psi, samples, i)
            if i >= prompt.shape[1]:
                probs = self._get_all_probs(psi) # (b, 2, 2, ..., 2)
                probs = probs.reshape(num_samples, -1) # (b, 2**num_qubits)
                sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(1) # (b,)
            # print(probs)
            
                for q in range(self.num_qubits):
                    samples[:, i, q] = (sampled_indices >> (self.num_qubits - 1 - q)) & 1
            if self.reset:
                psi = self._initial_state(num_samples)
            else:
                psi = self._collapse(psi, samples[:, i])
            
        return samples if num_samples > 1 else samples.squeeze(0)