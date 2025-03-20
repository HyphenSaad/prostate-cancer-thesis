import math
import collections
import numpy as np
from itertools import islice

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