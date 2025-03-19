# !python "/kaggle/working/GPFM/create_splits_seq.py" \
#     --prefix splits712 \
#     --task PANDA \
#     --seed 1 \
#     --label_frac 1.0 \
#     --val_frac 0.1 \
#     --test_frac 0.2 \
#     --k 1

import pandas as pd
import numpy as np

def save_splits(
  dataset_splits,
  column_keys,
  filename,
  boolean_style = False
):
  splits = [dataset_splits[i].slide_data['slide_id'] for i in range(len(dataset_splits))]
  if not boolean_style:
    df = pd.concat(splits, ignore_index = True, axis = 1)
    df.columns = column_keys
  else:
    df = pd.concat(splits, ignore_index = True, axis = 0)
    one_hot = np.eye(len(dataset_splits)).astype(bool)
    bool_array = np.repeat(one_hot, [len(dset) for dset in dataset_splits], axis = 0)
    df = pd.DataFrame(
      bool_array,
      index = df.values.tolist(), 
      columns = ['train', 'val', 'test']
    )

  df.to_csv(filename)