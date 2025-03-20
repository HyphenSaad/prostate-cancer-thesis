import warnings
warnings.filterwarnings('ignore')

import os 
import sys
import pandas as pd
import numpy as np
import time
import argparse
import random
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import OUTPUT_BASE_DIRECTORY
from constants.mil_models import MILModels
from constants.encoders import Encoders
from utils.helper import create_directories
from utils.data_loader import GenericMILDataset
from utils.train_engine import TrainEngine
from utils.file_utils import save_pkl
from utils.logger import Logger

logger = Logger()

CONFIG = {
  'backbone': Encoders.RESNET50.value,
  'mil_model': MILModels.TRANS_MIL.value,
  'drop_out': True,
  'n_classes': 6,
  'learning_rate': 1e-4,
  'k_fold': 1,
  'patch_size': 512,
  'in_dim': 1024,
  'max_epochs': 10,
  'directories': {
    'save_base_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'train'),
    'create_splits_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_splits'),
  },
  'verbose': True,
}

def seed_torch(seed = 1):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if we are using multi-GPU.
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

def main():
  create_directories(CONFIG['directories'])

  folds = np.arange(0, CONFIG['k_fold'])
  dataset = GenericMILDataset(
    patches_dir = CONFIG['directories']['create_patches_directory'],
    extract_patches_dir = CONFIG['directories']['extract_patches_directory'],
    features_pt_directory = CONFIG['directories']['features_pt_directory'],
    csv_path = CONFIG['dataset_info_csv'],
    label_column = 'isup_grade',
    label_dict = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5 },
    verbose = CONFIG['verbose'],
  )
  
  all_test_auc = []
  all_val_auc = []
  all_test_acc = []
  all_val_acc = []
  all_test_f1 = []
  all_val_f1 = []

  for fold in folds:
    seed_torch()

    dataset_split = dataset.return_splits(
      CONFIG['backbone'],
      CONFIG['patch_size'], 
      from_id = False, 
      csv_path='{}/splits_{}.csv'.format(CONFIG['directories']['create_splits_directory'], fold)
    )

    drop_out = 0.25 if CONFIG['drop_out'] else 0.0
    train_engine = TrainEngine(
      datasets = dataset_split,
      fold = fold,
      drop_out = drop_out,
      result_directory = CONFIG['directories']['save_base_directory'],
      mil_model_name = CONFIG['mil_model'],
      learning_rate = CONFIG['learning_rate'],
      max_epochs = CONFIG['max_epochs'],
      in_dim = CONFIG['in_dim'],
      n_classes = CONFIG['n_classes'],
      verbose = CONFIG['verbose'],
    )

    results, test_auc, val_auc, test_acc, val_acc, test_f1, val_f1  = train_engine.train_model(fold)
    all_test_auc.append(test_auc)
    all_val_auc.append(val_auc)
    all_test_acc.append(test_acc)
    all_val_acc.append(val_acc)
    all_test_f1.append(test_f1)
    all_val_f1.append(val_f1)

    # filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
    filename = os.path.join(CONFIG['directories']['save_base_directory'], 'split_{}_results.pkl'.format(fold))
    save_pkl(filename, results)

  final_df = pd.DataFrame({
    'folds': folds,
    'test_auc': all_test_auc, 
    'val_auc': all_val_auc,
    'test_acc': all_test_acc,
    'val_acc' : all_val_acc,
    'test_f1': all_test_f1,
    'val_f1': all_val_f1
  })

  if len(folds) != CONFIG['k_fold']:
    save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
  else:
    save_name = 'summary.csv'
  final_df.to_csv(os.path.join(CONFIG['directories']['save_base_directory'], save_name))

if __name__ == '__main__':
  main()