import time
import sys
import argparse
from os import path
from collections import namedtuple
import pickle

import wandb
import experiment

def main():
	# meta config zone
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-j',
		'--job-id',
		type=int,
		default=int(time.time()),
		help='The job id (and the name of the wandb group)'
	)
	parser.add_argument(
		'-o',
		'--output-path',
		type=str,
		default="bests",
		help='The directory which will contain job sub-directories containing the checkpoints of best models'
	)

	parser.add_argument(
		'-t',
		'--task',
		type=str,
		default="full_recovery",
		help='The task to perform'
	)
	parser.add_argument(
		'-df',
		'--data-form',
		type=str,
		default="unique_n",
		help='The data form to use when creating the training (default: unique_n)'
	)
	parser.add_argument(
		'-ddp',
		'--data-directory-path',
		type=str,
		default="./data",
		help='The path to the directory containing the data to use (default: ./data)'
	)
	parser.add_argument(
		'-da',
		'--data-action',
		type=str,
		default="load",
		choices=[ "load", "generate", "generate_and_save" ],
		help='The action to perform when preparing the data (choices: load, generate, generate_and_save; default: load)'
	)
	parser.add_argument(
		'-f',
		'--focus',
		type=int,
		default=0,
		help='The rolling ID of the group to focus at (default 0, means all groups being loaded)'
	)

	parser.add_argument(
		'-m',
		'--mode',
		type=str,
		default='single',
		choices=['single', 'sweep'],
		help='Choose whether to do a single evaluation run or whether to start a sweep agent (default: single)'
	)
	parser.add_argument(
		'--sweep-id',
		type=str,
		default=0,
		help='The id of the sweep to connect to (usually a string rather than a number)'
	)
	parser.add_argument(
		'--sweep-runs',
		type=int,
		default=1,
		help='The number of sweep runs to do in this job (default: 1)'
	)
	parser.add_argument(
		'--verbosity',
		type=int,
		default=2,
		help='The terminal output verbosity level (0 is min, 2 is max, default: 2)'
	)
	parser.add_argument(
		'--wandbosity',
		type=int,
		default=2,
		help='The level of verbosity for wandb (0 is min, 2 is max, default: 2)'
	)
	parser.add_argument(
		'--plotlibsity',
		type=int,
		default=1,
		help='The level of verbosity for matplotlib (0 is min, 1 is max, default: 1)'
	)
	meta_config = parser.parse_args()
	gettrace = getattr(sys, 'gettrace', None)
	meta_config.is_debug_instance = False if gettrace is None or not gettrace() else True
	
	# experiment config zone
	default_experiment_config = {
		'optimizer': 'adamw',
		'lr': 0.01,

		'data_multiplier': 100,
		'data_exponent': 1.0,
		'batch_size': 128,
		'epochs_multiplier': 100,
		'patience_fraction': 0.15,

		'n_primitives': 7,
		'program_length': 7,
		'beta': 0.025,
		'gamma': 0.005,
		'delta': 0.0000,

		'attention_dropout': 0.25,
		'execution_dropout': 0.0
	}
	ExperimentConfig = namedtuple('ExperimentConfig', default_experiment_config.keys())
	experiment_config = ExperimentConfig(**default_experiment_config)

	# run the job conditional on the mode
	if meta_config.mode == 'single':
		job(experiment_config, meta_config)
	elif meta_config.mode == 'sweep':
		wandb.agent(meta_config.sweep_id, lambda: job(experiment_config, meta_config), count=meta_config.sweep_runs)

	
def job(experiment_config, meta_config):
	if meta_config.task == 'full_recovery':
		group_entries = experiment.load_groups(path.join(meta_config.data_directory_path, 'gens10.csv'))
	elif meta_config.task == 'partial_recovery':
		if meta_config.data_action == 'load':
			with open(path.join(meta_config.data_directory_path, 'partial_recovery_data.pickle'), 'rb') as partial_recovery_data_file:
				group_entries = pickle.load(partial_recovery_data_file)
		else:
			group_entries = experiment.make_program_groups([5, 6, 7, 8, 9], min_span=2, max_span=5, atom_count=5, min_program_length=1, max_program_length=4, program_counts=[4, 5, 6, 7, 8]*4)
			if meta_config.action == 'generate_and_save':
				with open(path.join(meta_config.data_directory_path, 'partial_recovery_data.pickle'), 'wb') as partial_recovery_data_file:
					pickle.dump(group_entries, partial_recovery_data_file)
		
	elif meta_config.task == 'smallest_generator_finding':
		if meta_config.data_action == 'load':
			with open(path.join(meta_config.data_directory_path, 'smallest_generator_finding_data.pickle'), 'rb') as smallest_generator_finding_data_file:
				group_entries = pickle.load(smallest_generator_finding_data_file)
		else:
			group_entries = experiment.make_cycle_groups(datapoint_sizes=[12, 14], min_cycle_length=2, max_cycle_length=4, min_cycle_count=1, max_cycle_count=3, max_orders=[1, 2, 3, 4]*10)
			if meta_config.action == 'generate_and_save':
				with open(path.join(meta_config.data_directory_path, 'smallest_generator_finding_data.pickle'), 'wb') as smallest_generator_finding_data_file:
					pickle.dump(group_entries, smallest_generator_finding_data_file)
	else:
		raise Exception(f"Unknown task '{meta_config.task}', expected full_recovery, partial_recovery, or smallest_generator_finding")
	
	# actual job
	best_test_losses, recalls, precisions, qualities = [], [], [], []
	sum_of_orders_so_far = 0
	best_test_loss_cum, recall_cum, precision_cum, quality_cum = 0, 0, 0, 0
	for group_entry in group_entries:
		# apply focus
		if meta_config.focus > 0 and group_entry['rolling_id'] != meta_config.focus:
			continue

		# wandb config zone
		if meta_config.wandbosity >= 1:
			group_information = { 'group': str(meta_config.job_id) }  if meta_config.mode == 'single' else { 'group': str(meta_config.job_id) }
			name_information = { 'name': group_entry['description'] } if meta_config.mode == 'single' else { 'name': str(meta_config.job_id) }

			if meta_config.wandbosity >= 2 or meta_config.mode != 'sweep' or group_entry['rolling_id'] == 1:
				# if wandbosity is 1 for sweeps then init only on the first group and then roll on with it
				run = wandb.init(
					project="disper",
					tags=[
						str(meta_config.job_id),
						group_entry['description']
					],
					config=dict(experiment_config._asdict()) if type(experiment_config).__name__ == 'ExperimentConfig' else dict(experiment_config._as_dict()),
					reinit=True,
					settings=wandb.Settings(start_method='thread'),
					**name_information,
					**group_information
				)
			else:
				run = None

			if meta_config.mode == 'sweep':
				experiment_config = wandb.config
		else:
			run = None


		best_test_loss, recall, precision, quality = experiment.run(group_entry, experiment_config, meta_config)
		sum_of_orders_so_far += group_entry['order']
		quality = min(quality, 1.0) # quality can be >1 and it makes sense in the partial recovery task; it just breakes things visually in wandb kinda
		
		best_test_losses.append(best_test_loss)
		recalls.append(recall)
		precisions.append(precision)
		qualities.append(quality)

		best_test_loss_cum += float(best_test_loss * group_entry['order'])
		recall_cum += float(recall * group_entry['order'])
		precision_cum += float(precision * group_entry['order'])
		quality_cum += float(quality * group_entry['order'])

		if meta_config.wandbosity >= 1:
			step_setter = { 'step': int(group_entry['rolling_id']) } if meta_config.wandbosity==1 else {}
			wandb.log({
				"rolling_id": group_entry["rolling_id"],

				"rolling_average_best_test_loss": mean(float(bts) for bts in best_test_losses),
				"rolling_average_recall": mean(float(recall) for recall in recalls),
				"rolling_average_precision": mean(float(precision) for precision in precisions),
				"rolling_average_quality": mean(float(quality) for quality in qualities),

				"rolling_weighted_best_test_loss": best_test_loss_cum / sum_of_orders_so_far,
				"rolling_weighted_recall": recall_cum / sum_of_orders_so_far,
				"rolling_weighted_precision": precision_cum / sum_of_orders_so_far,
				"rolling_weighted_quality": quality_cum / sum_of_orders_so_far,
			}, **step_setter)
		
		if meta_config.wandbosity >= 1:
			if meta_config.wandbosity >= 2 or meta_config.mode != 'sweep':
				run.finish()

main()