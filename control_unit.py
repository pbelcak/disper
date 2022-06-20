import torch
from torch import nn

class ControlUnit(nn.Module):
	def __init__(self, program_length, n_primitives):
		super(ControlUnit, self).__init__()
		self.program_length = program_length
		self.n_primitives = n_primitives

		# dim 0 batch entries, dim 1 program steps, dim 2 primitives
		self.choice_softmax = nn.Softmax(dim=2)	

	def forward(self, attentor_output):
		choice_scores = torch.reshape(attentor_output, (attentor_output.shape[0], self.program_length, self.n_primitives))
		choices = self.choice_softmax(choice_scores)
		choice_powers = torch.sum(choices, dim=1)
		choices_used_primitives_count = torch.sum(torch.sigmoid(10 * (choice_powers - 1)), dim=1)

		return choices, choices_used_primitives_count
			
