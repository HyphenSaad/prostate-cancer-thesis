import warnings
warnings.filterwarnings('ignore')

import os 
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import OUTPUT_BASE_DIRECTORY, DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME
from utils.helper import create_directories
from utils.data_loader import GenericMILDataset

CONFIG = {
  'validation_fraction': 0.1,
  'test_fraction': 0.1,
  'k_fold': 10,
  'dataset_info_csv': os.path.join(DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME),
  'directories': {
    'save_base_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_splits'),
    'create_patches_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches', 'patches'),
    'extract_patches_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_patches'),
    'features_pt_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_features', 'pt_files'),
  },
  'verbose': False,
}

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

def main():
  create_directories(CONFIG['directories'])

  dataset = GenericMILDataset(
    patches_dir = CONFIG['directories']['create_patches_directory'],
    extract_patches_dir = CONFIG['directories']['extract_patches_directory'],
    features_pt_directory = CONFIG['directories']['features_pt_directory'],
    csv_path = CONFIG['dataset_info_csv'],
    label_column = 'isup_grade',
    label_dict = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5 },
    verbose = CONFIG['verbose'],
  )

  num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
  dataset.create_splits(
    k = CONFIG['k_fold'],
    val_num = np.round(num_slides_cls * CONFIG['validation_fraction']).astype(int),
    test_num = np.round(num_slides_cls * CONFIG['test_fraction']).astype(int)
  )
  
  # print(dataset.patient_cls_ids)
  # print(np.round(num_slides_cls * CONFIG['validation_fraction']).astype(int))
  # print(np.round(num_slides_cls * CONFIG['test_fraction']).astype(int))
  # return

  for k in range(CONFIG['k_fold']):
    dataset.set_splits()
    splits = dataset.return_splits()

    save_splits(
      splits, 
      ['train', 'val', 'test'], 
      os.path.join(CONFIG['directories']['save_base_directory'], 'splits_{}.csv'.format(k))
    )
    
    save_splits(
      splits, 
      ['train', 'val', 'test'],
      os.path.join(CONFIG['directories']['save_base_directory'], 'splits_{}_boolean.csv'.format(k)),
      boolean_style = True
    )

    descriptor_df = dataset.test_split_gen(return_descriptor = True, verbose = CONFIG['verbose'])
    descriptor_df.to_csv(os.path.join(CONFIG['directories']['save_base_directory'], 'splits_{}_descriptor.csv'.format(k)))

if __name__ == '__main__':
  main()