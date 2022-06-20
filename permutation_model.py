import torch
from torch import nn

from naive_reconstructor import NaiveReconstructor
from attentor import Attentor
from control_unit import ControlUnit
from execution_unit import ExecutionUnit

class PermutationModel(nn.Module):
	def __init__(self, input_width: int, program_length: int, n_primitives: int, n_attention_heads: int, attention_head_characteristic: tuple[int, int], attention_hidden_width: int, attention_dropout: float = 0.0, execution_dropout: float = 0.0):
		super(PermutationModel, self).__init__()
		self.input_width = input_width
		
		self.naive_reconstructor = NaiveReconstructor(input_width)
		self.attentor = Attentor(input_width,
			n_attention_heads,
			attention_head_characteristic,
			output_characteristic=(attention_hidden_width, program_length * n_primitives),
			dropout=attention_dropout
		)
		self.control_unit = ControlUnit(program_length, n_primitives)
		self.execution_unit = ExecutionUnit(input_width, n_primitives, dropout=execution_dropout)

	def forward(self, input_tensor):
		inputs = input_tensor[:, 0:self.input_width]
		naive_reconstruction = self.naive_reconstructor(input_tensor)
		attentor_output = self.attentor(naive_reconstruction)
		choices, choices_max_powers = self.control_unit(attentor_output)
		execution_output = self.execution_unit(choices)
		output = torch.matmul(execution_output, inputs)
		return output, choices_max_powers
		