import torch
from torch import nn

import wandb
import device_decision

def train(model, dataloader, experiment_config, meta_config):
	device = device_decision.device

	# get optimizer from string
	optimizer = get_optimizer_from_string(model, experiment_config.optimizer, lr=experiment_config.lr)

	# training begins here
	size = len(dataloader.dataset)
	last_print_point = 0
	epoch_loss_total, epoch_loss_reconstruction, epoch_loss_eye_divergence, epoch_loss_double_stochasticity, epoch_loss_pow, epoch_definiteness, def_count = 0, 0, 0, 0, 0, 0, 0

	model.train()
	lossobj = nn.MSELoss() if meta_config.data_form == 'integer_100' else nn.BCELoss()
	for batch, (X, y) in enumerate(dataloader):
		current_point = batch * len(X)
		X, y = X.to(device), y.to(device)

		# Compute prediction error
		pred, choices_max_powers = model(X)
		pred = torch.clamp(pred, 0, 1) if meta_config.data_form != 'integer_100' else pred

		loss_total, loss_reconstruction, loss_eye_divergence, loss_double_stochasticity, loss_pow \
			 = multihead_permutation_loss(model, pred, y, choices_max_powers, beta=experiment_config.beta, gamma=experiment_config.gamma, delta=experiment_config.delta, lossobj=lossobj)
		
		epoch_loss_total += loss_total
		epoch_loss_reconstruction += loss_reconstruction
		epoch_loss_eye_divergence += loss_eye_divergence
		epoch_loss_double_stochasticity += loss_double_stochasticity
		epoch_loss_pow += loss_pow
		
		# Backpropagation
		optimizer.zero_grad()
		loss_total.backward()
		optimizer.step()

		# Definitiveness calculation
		definitenesses = []
		for primitive in model.execution_unit.get_primitives():
			best_entries, _ = torch.max(primitive, dim=1)
			definiteness = torch.mean(best_entries)
			definitenesses.append(definiteness)
		definiteness = torch.mean(torch.tensor(definitenesses))
		epoch_definiteness += definiteness
		def_count += 1

		# Print progress at ~10 checkpoints
		if meta_config.verbosity >= 2 and current_point - last_print_point > size//10:
			last_print_point = current_point
			loss_total, current = loss_total.item(), batch * len(X)
			print(f" - loss: total {loss_total: >7.3f}, (rec {loss_reconstruction/loss_total:.2f}, eye {loss_eye_divergence/loss_total:.2f}, ds {loss_double_stochasticity/loss_total:.2f}, pow {loss_pow/loss_total:.2f}), def {definiteness:.3f}  [{current:>5d}/{size:>5d}]", end="\t\t\r")
	
	epoch_loss_total /= batch
	epoch_loss_reconstruction /= batch
	epoch_loss_eye_divergence /= batch
	epoch_loss_double_stochasticity /= batch
	epoch_loss_pow /= batch
	epoch_definiteness /= def_count

	epoch_fraction_reconstruction = epoch_loss_reconstruction / epoch_loss_total
	epoch_fraction_eye_divergence = epoch_loss_eye_divergence / epoch_loss_total
	epoch_fraction_double_stochasticity = epoch_loss_double_stochasticity / epoch_loss_total
	epoch_fraction_pow = epoch_loss_pow / epoch_loss_total

	if meta_config.wandbosity >= 2:
		metrics = {
			'total_loss': epoch_loss_total,
			'reconstruction_loss': epoch_loss_reconstruction,
			'eye_divergence_loss': epoch_loss_eye_divergence,
			'double_stochasticity_loss': epoch_loss_double_stochasticity,
			'fraction_reconstruction': epoch_fraction_reconstruction,
			'fraction_eye_divergence': epoch_fraction_eye_divergence,
			'fraction_double_stochasticity': epoch_fraction_double_stochasticity,
			'definiteness': epoch_definiteness
		}
		wandb.log(metrics)

	if meta_config.verbosity >= 1:
		print('\x1b[2K', end="\r") # line clear
		print(
			f" - mean train loss: \t{epoch_loss_total: >7.3f} (rec {epoch_fraction_reconstruction:.2f}, eye {epoch_fraction_eye_divergence:.2f}, ds {epoch_fraction_double_stochasticity:.2f}, pow {epoch_fraction_pow:.2f}), def {epoch_definiteness:.3f}",
			end="\t\t\t\n"
		)

		
def test(model, dataloader, experiment_config, meta_config):
	device = device_decision.device

	num_batches = len(dataloader)
	model.eval()

	test_total, test_reconstruction, test_eye_divergence, test_double_stochasticity, test_pow = 0, 0, 0, 0, 0
	lossobj = nn.MSELoss() if meta_config.data_form == 'integer_100' else nn.BCELoss()
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred, choices_max_powers = model(X)
			pred = torch.clamp(pred, 0, 1) if meta_config.data_form != 'integer_100' else pred

			loss_total, loss_reconstruction, loss_eye_divergence, loss_double_stochasticity, loss_pow \
				= multihead_permutation_loss(model, pred, y, choices_max_powers, beta=experiment_config.beta, gamma=experiment_config.gamma, delta=experiment_config.delta, lossobj=lossobj)
			
			test_total += loss_total
			test_reconstruction += loss_reconstruction
			test_eye_divergence += loss_eye_divergence
			test_double_stochasticity += loss_double_stochasticity
			test_pow += loss_pow

	test_total /= num_batches
	test_reconstruction /= num_batches
	test_eye_divergence /= num_batches
	test_double_stochasticity /= num_batches
	test_pow /= num_batches

	if meta_config.verbosity >= 1:
		print(
			f" - mean test loss: \t{test_total: >7.3f} (rec {test_reconstruction/test_total:.2f}, eye {test_eye_divergence/test_total:.2f}, ds {test_double_stochasticity/test_total:.2f}, pow {test_pow/test_total:.2f})",
			end="\t\t\t\n")

	return test_total

def multihead_permutation_loss(model, y_predicted, y_true, choices_max_powers, beta, gamma, delta, lossobj):
	reconstruction_loss = lossobj(y_predicted, y_true)
	
	model_heads = model.execution_unit.get_primitives()	# (n_primitives, input_width, input_width)
	head_column_sums = model_heads.sum(dim=1)			# (n_primitives, input_width)
	head_deviations_from_double_stochasticity = \
		head_column_sums - 1 # (n_primitives, input_width)
	column_losses = torch.square(head_deviations_from_double_stochasticity)
	double_stochasticity_loss = column_losses.sum()

	head_diagonals = torch.diagonal(model_heads, offset=0, dim1=-2, dim2=-1)	# (n_primitives, input_width)
	eye_divergence_loss = torch.mean(torch.square((head_diagonals - 1)))

	choice_max_power_loss = torch.mean(choices_max_powers)
	
	total_loss = reconstruction_loss + beta * eye_divergence_loss + gamma * double_stochasticity_loss + delta * choice_max_power_loss

	return total_loss, reconstruction_loss, beta * eye_divergence_loss,  gamma * double_stochasticity_loss, delta * choice_max_power_loss

def early_stopping(model, checkpointFilePath: str, validation_history: list, patience: int):
	least_validation_loss_in_history =  min(validation_history)

	if validation_history[-1] == least_validation_loss_in_history:
		torch.save(model.state_dict(), checkpointFilePath)

	if len(validation_history) < patience:
		return False

	should_stop_early = validation_history[-patience] == least_validation_loss_in_history
	return should_stop_early

def get_optimizer_from_string(model, optimizer_string: str, **kwargs):
	if optimizer_string == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), **kwargs)
	elif optimizer_string == 'adamw':
		optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
	elif optimizer_string == 'adadelta':
		optimizer = torch.optim.Adadelta(model.parameters(), **kwargs)
	elif optimizer_string == 'rmsprop':
		optimizer = torch.optim.RMSprop(model.parameters(), **kwargs)
	else:
		raise Exception(f"Uknown optimizer {optimizer_string}")

	return optimizer