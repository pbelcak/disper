import torch
from torch import nn
import numpy as np

from data import *
from permutation_model import *
import curriculum

def dihedral_check():
	# LR = 1e-3
	# make data
	datapoint_size = 4
	permutation_a = [ 0, 1, 2, 3 ]
	permutation_b = [ 0, 3, 1, 2 ]
	permutation_c = [ 0, 2, 3, 1 ]
	permutation_sa = [ 3, 2, 1, 0]
	permutation_sb = [ 3, 0, 1, 2 ]
	#permutation_sc = [ 3, 2, 0, 1 ]
	permutations = [ permutation_a, permutation_b, permutation_c, permutation_sa, permutation_sb ]
	example_training_data = make_data(permutations, datapoint_count=4000, value_range_low=0, value_range_high=100)
	example_training_loader = torch.utils.data.DataLoader(example_training_data, batch_size=40, shuffle=True)
	example_test_data = make_data(permutations, datapoint_count=1000, value_range_low=200, value_range_high=400)
	example_test_loader = torch.utils.data.DataLoader(example_test_data, batch_size=1, shuffle=True)

	# construct the model
	model = PermutationModel(
		input_width=datapoint_size,
		program_length=4,
		n_primitives=6,
		n_attention_heads=6,
		attention_head_characteristic=(4*datapoint_size*datapoint_size, 4),
		attention_hidden_width=30,
	)

	# train&test the model
	epochs = 70
	best_test_loss = float("inf")
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		curriculum.train(model, example_training_loader)
		test_total_loss = curriculum.test(model, example_test_loader)
		if test_total_loss < best_test_loss:
			torch.save(model.state_dict(), "bestPermModel.pt")
	print("Done!")

	model.load_state_dict(torch.load("bestPermModel.pt"))
	model.eval()

	# print results on the internals of the model
	print("Learned permutations:")
	for primitive in model.execution_unit.get_primitives():
		print(primitive.round(decimals=2))

	print("\nOriginal permutations:")
	for permutation in permutations:
		print(torch.eye(datapoint_size)[permutation])

def decomposition_check():
	# make data
	datapoint_size = 4
	permutation_a = [ 0, 1, 2, 3 ]
	permutation_b = [ 0, 3, 1, 2 ]
	permutation_c = [ 0, 2, 3, 1 ]
	permutations = [ permutation_a, permutation_b, permutation_c ]
	example_training_data = make_data(permutations, datapoint_count=4000, value_range_low=0, value_range_high=100)
	example_training_loader = torch.utils.data.DataLoader(example_training_data, batch_size=80, shuffle=True)
	example_test_data = make_data(permutations, datapoint_count=400, value_range_low=100, value_range_high=200)
	example_test_loader = torch.utils.data.DataLoader(example_test_data, batch_size=4, shuffle=True)

	# construct the model
	model = PermutationModel(
		input_width=datapoint_size,
		program_length=2,
		n_primitives=4,
		n_attention_heads=4,
		attention_head_characteristic=(4*datapoint_size*datapoint_size, 4),
		attention_hidden_width=30,
	)

	# train&test the model
	epochs = 30
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		curriculum.train(model, example_training_loader)
		# curriculum.test(model, example_test_loader)
	print("Done!")

	print("Learned permutations:")
	for primitive in model.execution_unit.get_primitives():
		print(primitive.round(decimals=2))

	print("\nOriginal permutation:")
	for permutation in permutations:
		print(torch.eye(datapoint_size)[permutation])

def sanity_check():
	# make data
	datapoint_size = 5
	permutation = np.random.permutation(datapoint_size)
	example_training_data = make_data([ permutation ], datapoint_count=4000, value_range_low=0, value_range_high=100)
	example_training_loader = torch.utils.data.DataLoader(example_training_data, batch_size=4, shuffle=True)
	example_test_data = make_data([ permutation ], datapoint_count=400, value_range_low=100, value_range_high=200)
	example_test_loader = torch.utils.data.DataLoader(example_test_data, batch_size=4, shuffle=True)

	# construct the model
	model = PermutationModel(
		input_width=datapoint_size,
		program_length=3,
		n_primitives=3,
		n_attention_heads=5,
		attention_head_characteristic=(2*datapoint_size*datapoint_size, 2),
		attention_hidden_width=30,
	)

	# train&test the model
	epochs = 5
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		curriculum.train(model, example_training_loader)
		curriculum.test(model, example_test_loader)
	print("Done!")

	print("Learned permutations:")
	for primitive in model.execution_unit.get_primitives():
		print(primitive.round(decimals=2))

	print("\nOriginal permutation:")
	print(torch.eye(datapoint_size)[permutation])