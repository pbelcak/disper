import random
import numpy as np
import itertools

def make_program_permutations(datapoint_size: int, min_span: int, max_span: int, atom_count: int, min_program_length: int, max_program_length: int, program_count: int):
	if max_span > datapoint_size:
		raise ValueError("max_span must be <= datapoint_size")

	# strategy: generate atoms, then the programs
	# atom generation
	atoms = []
	target_range = range(0, datapoint_size)
	for atom_index in range(0, atom_count):
		span = random.randint(min_span, max_span)
		atom_targets = random.sample(target_range, span)
		atom_sources = atom_targets.copy()
		atom_sources.sort()
		atom_map = dict(zip(atom_sources, atom_targets))
		atom = np.array([ (atom_map[i] if i in atom_map else i) for i in target_range])

		for already_an_atom in atoms:
			if (already_an_atom == atom).all():
				break
		else:
			atoms.append(atom)

	unique_atom_count = len(atoms)

	# program generation
	programs = []
	for program_index in range(0, program_count):
		program_length = random.randint(min_program_length, max_program_length)
		program_as_atom_indices = random.choices(range(0, unique_atom_count), k=program_length)

		program = np.array(target_range)
		for program_element_index in program_as_atom_indices:
			program = program[atoms[program_element_index]]

		for already_a_program in programs:
			if (already_a_program == program).all():
				break
		else:
			programs.append(program)

	return atoms, programs

def make_disjoint_cycle_permutations(datapoint_size: int, min_cycle_length: int, max_cycle_length: int, min_cycle_count: int, max_cycle_count: int, max_order: int):
	if max_cycle_count * max_cycle_length > datapoint_size:
		raise ValueError("max_cycle_count * max_cycle_length must be <= datapoint_size")
	
	# generate a base permutation
	cycle_count = random.randint(min_cycle_count, max_cycle_count)
	cycle_lengths = random.choices(range(min_cycle_length, max_cycle_length), k=cycle_count)
	cycles_seq = np.random.permutation(datapoint_size).tolist()

	cycles = []
	last_cycle_cut_index = 0
	for cycle_length in cycle_lengths:
		cycle = cycles_seq[last_cycle_cut_index:last_cycle_cut_index+cycle_length]
		cycles.append(cycle)
		last_cycle_cut_index += cycle_length
	
	# now, knowing the disjoint cycles, build the permutation
	cycle_permutations = []
	for cycle in cycles:
		base_permutation = np.array(range(0, datapoint_size))
		for cycle_element_current, cycle_element_next in pairwise_circle(cycle):
			base_permutation[cycle_element_next] = cycle_element_current
		cycle_permutations.append(base_permutation)

	# now, knowing the disjoint cycles, build the permutation
	permutations = []
	cycle_combinations = list(itertools.product(*(range(max_order+1) for cycle in cycles)))
	for cycle_combination in cycle_combinations:
		base_permutation = np.array(range(0, datapoint_size))
		for cycle_perm, order in zip(cycle_permutations, cycle_combination):
			program = np.array(range(0, datapoint_size))
			for _ in range(1, order+1):
				program = program[cycle_perm]

		for already_a_program in permutations:
			if (already_a_program == program).all():
				break
		else:
			permutations.append(program)

	return permutations, cycle_count

def pairwise_circle(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ... (s<last>,s0)
    a, b = itertools.tee(iterable)
    first_value = next(b, None)
    return itertools.zip_longest(a, b, fillvalue=first_value)