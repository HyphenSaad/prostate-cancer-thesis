import warnings
warnings.filterwarnings('ignore')

import os 
import sys
import pandas as pd
import numpy as np
import time
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import OUTPUT_BASE_DIRECTORY, DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME
from utils.helper import create_directories
from utils.data_loader import GenericMILDataset
from utils.logger import Logger

logger = Logger()

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

def show_configs():
  logger.empty_line()
  logger.info("Using Configurations;")
  logger.text(f"> Validation Fraction: {CONFIG['validation_fraction']}")
  logger.text(f"> Test Fraction: {CONFIG['test_fraction']}")
  logger.text(f"> K-Fold: {CONFIG['k_fold']}")
  logger.text(f"> Dataset Info CSV: {CONFIG['dataset_info_csv']}")
  logger.text(f"> Verbose: {CONFIG['verbose']}")
  logger.empty_line()

def load_arguments():
  parser = argparse.ArgumentParser(description = "Create Dataset Splits")
  parser.add_argument(
    "--validation-fraction",
    type = float,
    default = CONFIG['validation_fraction'],
    help = f"Validation Fraction (default: {CONFIG['validation_fraction']})"
  )
  parser.add_argument(
    "--test-fraction",
    type = float,
    default = CONFIG['test_fraction'],
    help = f"Test Fraction (default: {CONFIG['test_fraction']})"
  )
  parser.add_argument(
    "--k-fold",
    type = int,
    default = CONFIG['k_fold'],
    help = f"K-Fold (default: {CONFIG['k_fold']})"
  )
  parser.add_argument(
    "--verbose",
    type = bool,
    default = CONFIG['verbose'],
    help = f"Verbose (default: {CONFIG['verbose']})"
  )
  parser.add_argument(
    "--dataset-base-directory",
    type = str,
    default = DATASET_BASE_DIRECTORY,
    help = f"Dataset Base Directory (default: {DATASET_BASE_DIRECTORY})"
  )
  parser.add_argument(
    "--output-base-directory",
    type = str,
    default = OUTPUT_BASE_DIRECTORY,
    help = f"Output Base Directory (default: {OUTPUT_BASE_DIRECTORY})"
  )
  parser.add_argument(
    "--dataset-info-file-name",
    type = str,
    default = DATASET_INFO_FILE_NAME,
    help = f"Dataset Info File Name (default: {DATASET_INFO_FILE_NAME})"
  )

  args = parser.parse_args()
  
  CONFIG['validation_fraction'] = args.validation_fraction
  CONFIG['test_fraction'] = args.test_fraction
  CONFIG['k_fold'] = args.k_fold
  CONFIG['verbose'] = args.verbose
  
  dataset_base_dir = args.dataset_base_directory
  output_base_dir = args.output_base_directory
  dataset_info_file = args.dataset_info_file_name
  
  CONFIG['dataset_info_csv'] = os.path.join(dataset_base_dir, dataset_info_file)
  CONFIG['directories']['save_base_directory'] = os.path.join(output_base_dir, 'create_splits')
  CONFIG['directories']['create_patches_directory'] = os.path.join(output_base_dir, 'create_patches', 'patches')
  CONFIG['directories']['extract_patches_directory'] = os.path.join(output_base_dir, 'extract_patches')
  CONFIG['directories']['features_pt_directory'] = os.path.join(output_base_dir, 'extract_features', 'pt_files')

def main():
  logger.draw_header("Create Dataset Splits")
  load_arguments()
  
  logger.info("Creating Dataset Splits...")
  start_time = time.time()

  logger.info("Creating Directories...")
  create_directories(CONFIG['directories'])

  show_configs()

  logger.info("Loading dataset from CSV...")
  df = pd.read_csv(CONFIG['dataset_info_csv'])
  if len(df) < 100:
    logger.error('Dataset is too small to create splits (size: {})', len(df))
    raise ValueError('Dataset is too small to create splits')
  logger.info("Loaded {} samples from dataset", len(df))

  logger.info("Initializing MIL Dataset...")
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
  logger.info("Creating {} splits with validation fraction {} and test fraction {}", 
             CONFIG['k_fold'], CONFIG['validation_fraction'], CONFIG['test_fraction'])
  
  dataset.create_splits(
    k = CONFIG['k_fold'],
    val_num = np.round(num_slides_cls * CONFIG['validation_fraction']).astype(int),
    test_num = np.round(num_slides_cls * CONFIG['test_fraction']).astype(int)
  )

  logger.info("Generating and saving splits:")
  for k in range(CONFIG['k_fold']):
    logger.text(f"> Processing fold {k+1}/{CONFIG['k_fold']}")
    dataset.set_splits()
    splits = dataset.return_splits()

    splits_path = os.path.join(CONFIG['directories']['save_base_directory'], 'splits_{}.csv'.format(k))
    save_splits(
      splits, 
      ['train', 'val', 'test'], 
      splits_path
    )
    
    bool_splits_path = os.path.join(CONFIG['directories']['save_base_directory'], 'splits_{}_boolean.csv'.format(k))
    save_splits(
      splits, 
      ['train', 'val', 'test'],
      bool_splits_path,
      boolean_style = True
    )

    descriptor_path = os.path.join(CONFIG['directories']['save_base_directory'], 'splits_{}_descriptor.csv'.format(k))
    descriptor_df = dataset.test_split_gen(return_descriptor = True, verbose = CONFIG['verbose'])
    descriptor_df.to_csv(descriptor_path)

  end_time = time.time()
  total_time = end_time - start_time

  logger.empty_line()
  logger.info("Total Processing Time: {:.2f} seconds ({:.2f} minutes)", total_time, total_time/60)
  logger.empty_line()
  logger.success("Created All Dataset Splits!")

if __name__ == '__main__':
  main()