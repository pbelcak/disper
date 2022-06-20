import itertools

import torch
from data import *
from permutation_model import *
import device_decision

def venn(a, b):
	only_a = []
	both = []
	only_b = []

	for element in a:
		for another in b:
			if torch.equal(element, another):
				both.append(element)
				break
		else:
			only_a.append(element)
	
	for element in b:
		for another in a:
			if torch.equal(element, another):
				break
		else:
			only_b.append(element)

	return only_a, both, only_b

def retrieve_unique_permutation_matrices(primitives):
	device = device_decision.device

	primitives = list(primitives)
	permutation_matrices = []

	eye = torch.eye(primitives[0].shape[0]).to(device)
	for primitive in primitives:
		permutation = torch.argmax(primitive, dim=1)
		if torch.unique(permutation).size() < permutation.size():
			# this is not a permutation (double-stochasticity has been violated)
			continue
		
		permutation_matrix = eye[permutation]
		permutation_matrices.append(permutation_matrix)

	unique_permutation_matrices = []
	for permutation_matrix in permutation_matrices:
		for unique_permutation_matrix in unique_permutation_matrices:
			if torch.equal(unique_permutation_matrix, permutation_matrix):
				break
		else:
			unique_permutation_matrices.append(permutation_matrix)
	
	return unique_permutation_matrices

def group_closure(permutation_matrices):
	device = device_decision.device
	
	group_order = permutation_matrices[0].shape[0]
	eye = torch.eye(group_order).to(device)

	element_orders = []
	for permutation_matrix in permutation_matrices:
		acc = torch.eye(group_order).to(device)
		for i in range(1, group_order+1):
			acc = torch.matmul(acc, permutation_matrix)
			if torch.equal(acc, eye):
				element_orders.append(i)
				break
		else:
			# raise Error("Could not find the order of the permutation given")
			# forget about the order, it can be bigger than the group order, and we don't actually need it
			# just peg at the group order
			element_orders.append(group_order)

	program_combinations = list(itertools.product(*(range(order) for order in element_orders)))

	precompiled_generator_powers = []
	for generator_index, generator_order in enumerate(element_orders):
		precompiled_generator_powers.append([
			torch.matrix_power(permutation_matrices[generator_index], power) for power in range(generator_order)
		])

	closure_elements = []
	for program_combination in program_combinations:
		acc = torch.eye(group_order).to(device)
		for generator_index, generator_power in enumerate(program_combination):
			acc = torch.matmul(acc, precompiled_generator_powers[generator_index][generator_power])

		for closure_element in closure_elements:
			if torch.equal(closure_element, acc):
				break
		else:
			closure_elements.append(acc)

	return closure_elements

def multiplication_table_to_permutation_matrices(table):
	device = device_decision.device

	permutation_matrices = []
	new_table = []

	for row in table:
		new_table_row = [e-1 for e in row] # this is because in the table, the first element is 1, but in the permutation matrix, it is 0
		new_table.append(new_table_row)

		permutation_against_identity = torch.tensor(new_table_row, dtype=torch.long)
		permutation_matrix = torch.eye(len(table))[permutation_against_identity].to(device)
		permutation_matrices.append(permutation_matrix)
	
	return permutation_matrices, new_table
