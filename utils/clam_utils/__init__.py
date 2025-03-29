import collections
import numpy as np
import torch
from itertools import islice
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
  # for other MIL methods
  if len(batch) == 1: return batch[0]
  else: raise RuntimeError

def generate_split(
  cls_ids,
  val_num, 
  test_num, 
  samples,
  n_splits = 5,
	seed = 7,
):
  indices = np.arange(samples).astype(int)
  np.random.seed(seed)

  for i in range(n_splits):
    all_val_ids = []
    all_test_ids = []
    sampled_train_ids = []
		
    for c in range(len(val_num)):
      possible_indices = np.intersect1d(cls_ids[c], indices) # all indices of this class
      val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

      remaining_ids = np.setdiff1d(possible_indices, val_ids) # indices of this class left after validation
      all_val_ids.extend(val_ids)

      test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
      remaining_ids = np.setdiff1d(remaining_ids, test_ids)
      all_test_ids.extend(test_ids)

      sampled_train_ids.extend(remaining_ids)

    yield sampled_train_ids, all_val_ids, all_test_ids

def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))
	weight_per_class = [
    N / len(dataset.slide_cls_ids[c]) if len(dataset.slide_cls_ids[c]) > 0 else 1e-8
    for c in range(len(dataset.slide_cls_ids))
  ]

	weight = [0] * int(N)
	for idx in range(len(dataset)):
		y = dataset.getlabel(idx)
		weight[idx] = weight_per_class[y]

	return torch.DoubleTensor(weight)

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, batch_size=1):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 0} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error