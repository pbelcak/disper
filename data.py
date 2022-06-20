import torch
import numpy as np

torch.set_default_dtype(torch.float64)

class PermutationDataset(torch.utils.data.Dataset):
	def __init__(self, x, y, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.x = x
		self.y = y

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		sample = [ self.x[idx], self.y[idx] ]
		#if self.transform:
		#	sample = self.transform(sample)
		return sample

	def __len__(self):
		return len(self.x)

def make_data(dataset_name: str, permutations: list, datapoint_count: int):
	n = len(permutations[0])

	if dataset_name == "integer_100":
		return make_data_integer_k(permutations, datapoint_count, k=100)
	elif dataset_name == "unique_n":
		return make_data_unique_k(permutations, datapoint_count, k=n)
	elif dataset_name == "unique_2n":
		return make_data_unique_k(permutations, datapoint_count, k=2*n)
	elif dataset_name == "arbitrary_n":
		return make_data_arbitrary_k(permutations, datapoint_count, k=n)
	elif dataset_name == "arbitrary_2n":
		return make_data_arbitrary_k(permutations, datapoint_count, k=2*n)
	else:
		raise Exception("Unknown dataset name: {}".format(dataset_name))

def make_data_integer_k(permutations, datapoint_count, k):
	datapoint_x_list = []
	datapoint_y_list = []

	i = 0
	while i < datapoint_count:
		permutation_to_use = permutations[i % len(permutations)]

		x = np.random.randint(low=0, high=k, size=len(permutation_to_use)).astype(np.float64, copy=False)
		# if x.max() == x.min():
		#	continue

		y = np.expand_dims(x[permutation_to_use], axis=1)
		x = np.expand_dims(x, axis=1)
		
		xandy = np.concatenate((x, y), axis=0)
		datapoint_x_list.append(torch.from_numpy(xandy))
		datapoint_y_list.append(torch.from_numpy(y))
		i += 1

	return PermutationDataset(datapoint_x_list, datapoint_y_list)

def make_data_unique_k(permutations, datapoint_count, k):
	# k is the number of candidates
	# n is the input width
	n = len(permutations[0])

	if k < n:
		raise ValueError("k must be >= n (you cant uniquely choose n elements from k candidates)")

	datapoint_x_list = []
	datapoint_y_list = []

	i = 0
	while i < datapoint_count:
		permutation_to_use = permutations[i % len(permutations)]

		# take a permutation of the numbers 0, 1, 2, ..., k-1
		#  - these are all of the elements we can choose from to fill the x vector
		x_scheme = np.random.permutation(k)[0:n]

		# compute the y vector from x under the permutation_to_use
		y_scheme = x_scheme[permutation_to_use]

		# convert the schemes to one-hot vectors
		x = np.eye(k)[x_scheme].astype(np.float64, copy=False)
		y = np.eye(k)[y_scheme].astype(np.float64, copy=False)

		# form input and output pairs for PermutationModel
		xandy = np.concatenate((x, y), axis=0)
		datapoint_x_list.append(torch.from_numpy(xandy))
		datapoint_y_list.append(torch.from_numpy(y))
		i += 1

	return PermutationDataset(datapoint_x_list, datapoint_y_list)

def make_data_arbitrary_k(permutations, datapoint_count, k):
	# k is the number of candidates
	# n is the input width
	n = len(permutations[0])

	if k < n:
		raise ValueError("k must be >= n (you cant uniquely choose n elements from k candidates)")

	datapoint_x_list = []
	datapoint_y_list = []

	i = 0
	while i < datapoint_count:
		permutation_to_use = permutations[i % len(permutations)]

		# take random numbers between 0 and k-1
		#  - these are all of the elements we can choose from to fill the x vector
		x_scheme = np.random.randint(low=0, high=k, size=n)

		# compute the y vector from x under the permutation_to_use
		y_scheme = x_scheme[permutation_to_use]

		# convert the schemes to one-hot vectors
		x = np.eye(k)[x_scheme].astype(np.float64, copy=False)
		y = np.eye(k)[y_scheme].astype(np.float64, copy=False)

		# form input and output pairs for PermutationModel
		xandy = np.concatenate((x, y), axis=0)
		datapoint_x_list.append(torch.from_numpy(xandy))
		datapoint_y_list.append(torch.from_numpy(y))
		i += 1

	return PermutationDataset(datapoint_x_list, datapoint_y_list)