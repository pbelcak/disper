import torch
from torch import nn

class Attentor(nn.Module):
	def __init__(self, input_width: int, n_heads: int, head_characteristic: tuple[int, int], output_characteristic: tuple[int, int], dropout=0.0):
		super(Attentor, self).__init__()
		self.input_width = input_width
		self.heads = nn.ModuleList([nn.Sequential(
				nn.Linear(input_width*input_width, head_characteristic[0], bias=True),
				nn.Dropout(dropout),
				nn.ReLU(),
				nn.Linear(head_characteristic[0], head_characteristic[1], bias=True),
				nn.Dropout(dropout),
			) for i in range(0, n_heads)
		])
		self.comprehensor = nn.Sequential(
			nn.Linear(n_heads * head_characteristic[1], output_characteristic[0], bias=True),
			nn.ReLU(),
			nn.Linear(output_characteristic[0], output_characteristic[1], bias=True),
		)

	def forward(self, naive_reconstruction):
		x = torch.flatten(naive_reconstruction, start_dim=1)

		heads_output = torch.cat([head(x) for head in self.heads], dim=1)
		comprehensor_output = self.comprehensor(heads_output)
		return comprehensor_output
			
