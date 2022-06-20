import os
import csv
import ast
from matplotlib import pyplot as plt

import torch
import wandb

import device_decision
import curriculum
from data import *
from permutation_model import *
from evaluation import *
from permutations import *

def run(group_entry, config, meta_config):
	device = device_decision.device
	
	if meta_config.verbosity >= 1:
		print(f"********************************************************************************")
		print(f"Group {group_entry['description']}")
		print(f"********************************************************************************")

	shortened_name = group_entry['description'].replace(' ', '')
	best_directory = os.path.join(meta_config.output_path, str(meta_config.job_id))
	best_checkpoint_file_path = f"{best_directory}//best_checkpoint_for_{shortened_name}.pt"
	if not os.path.exists(best_directory):
		os.makedirs(best_directory)
	plots_directory = f"plots//{meta_config.job_id}//{shortened_name}"
	if not os.path.exists(plots_directory):
		os.makedirs(plots_directory)
	
	group_order = group_entry['order']

	train_data_size = int((group_order ** config.data_exponent) * config.data_multiplier)
	test_data_size = int(train_data_size * 0.1)

	# make data
	dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}

	permutation_repositions = group_entry['table']
	datapoint_size = len(permutation_repositions[0])
	example_training_data = make_data(meta_config.data_form, permutation_repositions, datapoint_count=train_data_size)
	#example_training_loader = torch.utils.data.DataLoader(example_training_data, batch_size=int(config.batch_fraction * train_data_size), shuffle=True, **dataloader_kwargs)
	example_training_loader = torch.utils.data.DataLoader(example_training_data, batch_size=128, shuffle=True, **dataloader_kwargs)
	example_test_data = make_data(meta_config.data_form, permutation_repositions, datapoint_count=test_data_size)
	example_test_loader = torch.utils.data.DataLoader(example_test_data, batch_size=test_data_size, shuffle=True, **dataloader_kwargs)

	# construct the model
	model = PermutationModel(
		input_width=datapoint_size,
		program_length=config.program_length,
		n_primitives=config.n_primitives,
		n_attention_heads=group_order+4,
		attention_head_characteristic=(24, 8),
		attention_hidden_width=group_order*5,
		attention_dropout=config.attention_dropout,
		execution_dropout=config.execution_dropout
	)
	model = model.to(device)

	# train&test the model
	epochs = group_order * config.epochs_multiplier
	patience = int(epochs * config.patience_fraction)
	
	test_history = []
	for t in range(epochs):
		if meta_config.verbosity >= 1:
			print(f"Epoch {t+1}")
		
		curriculum.train(model, example_training_loader, config, meta_config)
		test_total_loss = curriculum.test(model, example_test_loader, config, meta_config)
		test_history.append(test_total_loss)

		if meta_config.plotlibsity >= 1 and t % 30 == 0:
			primitives_as_tensor = model.execution_unit.get_primitives()
			save_plot_of_primitives(primitives_as_tensor, os.path.join(plots_directory, f"primitives_epoch_{t+1}.png"))

		if curriculum.early_stopping(model, best_checkpoint_file_path, test_history, patience=patience):
			if meta_config.verbosity >= 1: print("Stopping early (ran out of patience)")
			break

	model.load_state_dict(torch.load(best_checkpoint_file_path))
	model.eval()

	learned_permutations = retrieve_unique_permutation_matrices(model.execution_unit.get_primitives())
	closure_of_learned_permutations = group_closure(learned_permutations)
	group_permutation_matrices = group_entry['permutations']

	# do venn diagram of the two
	only_a, both, only_b = venn(group_permutation_matrices, closure_of_learned_permutations)
	only_a_f, both_f, only_b_f = len(only_a) / group_order, len(both) / group_order, len(only_b) / group_order

	recall = both_f
	precision = len(both) / (len(both) + len(only_b))
	quality = len(group_entry['generators']) / max(len(learned_permutations) - 1, 1) # -1 for the identity matrix

	evaluation_metrics = {
		"rolling_id": group_entry['rolling_id'],
		"best_test_loss": min(test_history),

		"false_negatives": len(only_a),
		"false_positives": len(only_b),
		"true_positives": len(both),

		"recall": recall,
		"precision": precision,

		"only_a_f": only_a_f,
		"both_f": both_f,
		"only_b_f": only_b_f,

		"support": len(group_entry['generators']),
		"support_learned": len(learned_permutations),
		"support_quality": quality
	}
	if meta_config.wandbosity >= 2:
		wandb.log(evaluation_metrics)

	if meta_config.verbosity >= 1:
		print(f"\nRetrieval results for {group_entry['description']}: {only_a_f*100:.1f}% only in group, {both_f*100:.1f}% in both, {only_b_f*100:.1f}% only in learned")
		print("") # for a newline after each run

	return min(test_history), recall, precision, quality

group_entry_schema = [ "order", "id", "rolling_id", "description", "generators", "table" ]
def load_groups(filePath):
	entries = []

	with open(filePath, "r") as file:
		reader = csv.reader(file, delimiter=';', quotechar='"')
		for row in reader:
			entry = dict(zip(group_entry_schema, row))
			entry["order"] = int(entry["order"])
			entry["maximum_element_order"] = entry["order"]
			entry["id"] = int(entry["id"])
			entry["rolling_id"] = int(entry["rolling_id"])
			entry["generators"] = [part.replace('[', '').replace(']', '').strip() for part in entry["generators"].split(",")]
			table_indices_from_1 = ast.literal_eval(entry["table"])
			entry["permutations"], entry["table"] = multiplication_table_to_permutation_matrices(table_indices_from_1)

			entries.append(entry)
	
	return entries

def make_program_groups(datapoint_sizes: list, min_span: int, max_span: int, atom_count: int, min_program_length: int, max_program_length: int, program_counts: list):
	entries = []

	rolling_id = 1
	for datapoint_size_id, datapoint_size in enumerate(datapoint_sizes):
		for program_count_id, program_count in enumerate(program_counts):
			entry = dict(zip(group_entry_schema, ["" for i in range(len(group_entry_schema))]))
			entry["id"] = program_count_id
			entry["rolling_id"] = rolling_id

			atoms, permutation_schemas = make_program_permutations(datapoint_size, min_span, max_span, atom_count, min_program_length, max_program_length, program_count)
			unique_atom_count = len(atoms)
			unique_program_count = len(permutation_schemas)
			entry["order"] = unique_program_count
			entry["maximum_element_order"] = unique_program_count
			entry["generators"] = [ f"f{i}" for i in range(0, unique_atom_count)]
			entry["table"] = permutation_schemas
			entry["permutations"] = generate_permutation_matrices(permutation_schemas)

			entry["description"] = f"{datapoint_size}-{min_span}-{max_span}-{unique_atom_count}-{min_program_length}-{max_program_length}-{program_count}-{len(permutation_schemas)}"
			entries.append(entry)
			rolling_id += 1

	return entries

def make_cycle_groups(datapoint_sizes: list, min_cycle_length: int, max_cycle_length: int, min_cycle_count: int, max_cycle_count: int, max_orders: list):
	entries = []

	rolling_id = 1
	for datapoint_size_id, datapoint_size in enumerate(datapoint_sizes):
		for max_order_id, max_order in enumerate(max_orders):
			entry = dict(zip(group_entry_schema, ["" for i in range(len(group_entry_schema))]))
			entry["id"] = max_order_id
			entry["rolling_id"] = rolling_id

			permutation_schemas, cycle_count = make_disjoint_cycle_permutations(datapoint_size, min_cycle_length, max_cycle_length, min_cycle_count, max_cycle_count, max_order)
			unique_program_count = len(permutation_schemas)
			entry["order"] = unique_program_count
			entry["maximum_element_order"] = max_order
			entry["generators"] = [ f"f{i}" for i in range(0, cycle_count)]
			entry["table"] = permutation_schemas
			entry["permutations"] = generate_permutation_matrices(permutation_schemas)

			entry["description"] = f"{datapoint_size}-{min_cycle_length}-{max_cycle_length}-{min_cycle_count}-{max_cycle_count}-{max_order}-{unique_program_count}"
			entries.append(entry)
			rolling_id += 1

	return entries

def generate_permutation_matrices(permutation_schemas):
	device = device_decision.device

	permutation_matrices = []
	for permutation_schema in permutation_schemas:
		permutation_matrix = torch.eye(len(permutation_schema))[permutation_schema].to(device)
		permutation_matrices.append(permutation_matrix)

	return permutation_matrices

def save_plot_of_primitives(primitives: list, file_path: str):
	primitives_list = list(primitives)
	fig, axs = plt.subplots(len(primitives_list))
	for primitive_index, primitive in enumerate(primitives_list):
		#pass
		with torch.no_grad():
			axs[primitive_index].imshow(primitive, cmap='Reds')
		axs[primitive_index].grid(color='black', linestyle='-', linewidth=1)
		axs[primitive_index].axis('off')

	fig.savefig(file_path)
	
	plt.close(fig)