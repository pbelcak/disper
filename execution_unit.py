import torch
from torch import nn

import device_decision

class ExecutionUnit(nn.Module):
	def __init__(self, input_width, n_primitives, dropout=0.0):
		super(ExecutionUnit, self).__init__()
		self.input_width = input_width
		self.n_primitives = n_primitives

		self.identity = torch.eye(input_width).unsqueeze(0).to(device_decision.device)
		self.primitiveScores = nn.Parameter(torch.randn(n_primitives-1, input_width, input_width), requires_grad=True)
		self.dropout = nn.Dropout(dropout)
		self.primitiveSoftmax = nn.Softmax(dim=2)

	def forward(self, control):
		primitives = self.get_primitives()
		
		action = torch.eye(self.input_width).unsqueeze(0).to(device_decision.device)
		for program_step in range(0, control.size()[1]):
			step_mixing_weights = control[:, program_step] # (B, n_primitives)
			step_mixture = step_mixing_weights.unsqueeze(2).unsqueeze(2) * primitives
			step_action = torch.sum(step_mixture, dim=1)

			action = torch.matmul(action, step_action)

		return action # (B, input_width, input_width)
	
	def get_primitives(self):
		processedPrimitives = self.primitiveSoftmax(self.dropout(self.primitiveScores))
		primitives = torch.cat([self.identity, processedPrimitives], dim=0) # n_primitives

		return primitives
