import torch
from torch import nn

class NaiveReconstructor(nn.Module):
	def __init__(self, input_width: int):
		super(NaiveReconstructor, self).__init__()
		self.input_width = input_width

	def forward(self, x):
		# this is to prevent division by zero
		__division_epsilon = 1e-10

		permutation_input = x[:, 0:self.input_width]
		permutation_output = x[:, self.input_width:]

		permutation_input_unsqueezed = permutation_input.unsqueeze(2)
		permutation_output_unsqueezed = permutation_output.unsqueeze(2)
		broadcast_differences = torch.transpose(permutation_input_unsqueezed, 1, 2) - permutation_output_unsqueezed
		skewed_differences = torch.square(broadcast_differences)
		normalising_constants = torch.amax(skewed_differences, dim=tuple(range(1, len(skewed_differences.shape))))
		non_batch_dims = len(skewed_differences.shape) - 1

		normalised_differences = skewed_differences / (normalising_constants + __division_epsilon)[(..., ) + (None, ) * non_batch_dims]
		naive_reconstruction = 1.0 - normalised_differences # shape (batch_size, input_width, input_width, ...)

		reduction_dimensions = tuple(range(3, len(skewed_differences.shape)))
		if reduction_dimensions:
			naive_reconstruction_processed = torch.amin(naive_reconstruction, dim=reduction_dimensions)
		else:
			naive_reconstruction_processed = naive_reconstruction
		return naive_reconstruction_processed 
