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

from constants.misc import OUTPUT_BASE_DIRECTORY, DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME
from constants.mil_models import MILModels
from constants.encoders import Encoders
from utils.helper import create_directories
from utils.data_loader import GenericMILDataset
from utils.train_engine import TrainEngine
from utils.file_utils import save_pkl, load_pkl
from utils.logger import Logger

logger = Logger()

CONFIG = {
  'backbone': Encoders.RESNET50.value,
  'mil_model': MILModels.TRANS_MIL.value,
  'drop_out': True,
  'n_classes': 6,
  'learning_rate': 1e-4,
  'k_fold': 4,
  'k_fold_start': -1,
  'k_fold_end': -1,
  'patch_size': 512,
  'in_dim': 1024,
  'max_epochs': 10,
  'start_epoch': -1,
  'end_epoch': -1,
  'kaggle_feature_path': None,
  'dataset_info_csv': os.path.join(DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME),
  'directories': {
    'save_base_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'train'),
    'create_splits_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_splits'),
    'create_patches_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches', 'patches'),
    'extract_patches_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_patches'),
    'features_pt_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_features', 'pt_files'),
  },
  'verbose': False,
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

def show_configs():
  logger.empty_line()
  logger.info("Using Configurations:")
  logger.text(f"> Backbone: {CONFIG['backbone']}")
  logger.text(f"> MIL Model: {CONFIG['mil_model']}")
  logger.text(f"> Drop Out: {CONFIG['drop_out']}")
  logger.text(f"> Number of Classes: {CONFIG['n_classes']}")
  logger.text(f"> Learning Rate: {CONFIG['learning_rate']}")
  logger.text(f"> K-Fold: {CONFIG['k_fold']}")
  logger.text(f"> K-Fold Start: {CONFIG['k_fold_start']}")
  logger.text(f"> K-Fold End: {CONFIG['k_fold_end']}")
  logger.text(f"> Patch Size: {CONFIG['patch_size']}")
  logger.text(f"> Input Dimension: {CONFIG['in_dim']}")
  logger.text(f"> Max Epochs: {CONFIG['max_epochs']}")
  logger.text(f"> Start Epoch: {CONFIG['start_epoch']}")
  logger.text(f"> End Epoch: {CONFIG['end_epoch']}")
  logger.text(f"> Kaggle Feature Path: {CONFIG['kaggle_feature_path']}")
  logger.text(f"> Dataset Info CSV: {CONFIG['dataset_info_csv']}")
  logger.text(f"> Verbose: {CONFIG['verbose']}")
  logger.empty_line()

def load_arguments():
  parser = argparse.ArgumentParser(description="Train MIL Models")
  parser.add_argument(
    "--backbone",
    type=str,
    default=CONFIG['backbone'],
    help=f"Backbone encoder model (default: {CONFIG['backbone']})"
  )
  parser.add_argument(
    "--mil-model",
    type=str,
    default=CONFIG['mil_model'],
    help=f"MIL model type (default: {CONFIG['mil_model']})"
  )
  parser.add_argument(
    "--drop-out",
    type=bool,
    default=CONFIG['drop_out'],
    help=f"Use dropout (default: {CONFIG['drop_out']})"
  )
  parser.add_argument(
    "--n-classes",
    type=int,
    default=CONFIG['n_classes'],
    help=f"Number of classes (default: {CONFIG['n_classes']})"
  )
  parser.add_argument(
    "--learning-rate",
    type=float,
    default=CONFIG['learning_rate'],
    help=f"Learning rate (default: {CONFIG['learning_rate']})"
  )
  parser.add_argument(
    "--k-fold",
    type=int,
    default=CONFIG['k_fold'],
    help=f"Number of folds (default: {CONFIG['k_fold']})"
  )
  parser.add_argument(
    "--k-fold-start",
    type=int,
    default=CONFIG['k_fold_start'],
    help=f"Start fold (default: {CONFIG['k_fold_start']})"
  )
  parser.add_argument(
    "--k-fold-end",
    type=int,
    default=CONFIG['k_fold_end'],
    help=f"End fold (default: {CONFIG['k_fold_end']})"
  )
  parser.add_argument(
    "--patch-size",
    type=int,
    default=CONFIG['patch_size'],
    help=f"Patch size (default: {CONFIG['patch_size']})"
  )
  parser.add_argument(
    "--in-dim",
    type=int,
    default=CONFIG['in_dim'],
    help=f"Input dimension (default: {CONFIG['in_dim']})"
  )
  parser.add_argument(
    "--max-epochs",
    type=int,
    default=CONFIG['max_epochs'],
    help=f"Maximum epochs (default: {CONFIG['max_epochs']})"
  )
  parser.add_argument(
    "--start-epoch",
    type=int,
    default=CONFIG['start_epoch'],
    help=f"Start epoch (default: {CONFIG['start_epoch']})"
  )
  parser.add_argument(
    "--end-epoch",
    type=int,
    default=CONFIG['end_epoch'],
    help=f"End epoch (default: {CONFIG['end_epoch']})"
  )
  parser.add_argument(
    "--kaggle-feature-path",
    type=str,
    default=CONFIG['kaggle_feature_path'],
    help=f"Kaggle feature path (default: {CONFIG['kaggle_feature_path']})"
  )
  parser.add_argument(
    "--dataset-info-csv",
    type=str,
    default=CONFIG['dataset_info_csv'],
    help=f"Dataset info CSV path (default: {CONFIG['dataset_info_csv']})"
  )
  parser.add_argument(
    "--output-base-directory",
    type=str,
    default=OUTPUT_BASE_DIRECTORY,
    help=f"Output base directory (default: {OUTPUT_BASE_DIRECTORY})"
  )
  parser.add_argument(
    "--verbose",
    type=bool,
    default=CONFIG['verbose'],
    help=f"Verbose mode (default: {CONFIG['verbose']})"
  )
  
  args = parser.parse_args()
  
  CONFIG['backbone'] = args.backbone
  CONFIG['mil_model'] = args.mil_model
  CONFIG['drop_out'] = args.drop_out
  CONFIG['n_classes'] = args.n_classes
  CONFIG['learning_rate'] = args.learning_rate
  CONFIG['k_fold'] = args.k_fold
  CONFIG['k_fold_start'] = args.k_fold_start
  CONFIG['k_fold_end'] = args.k_fold_end
  CONFIG['patch_size'] = args.patch_size
  CONFIG['in_dim'] = args.in_dim
  CONFIG['max_epochs'] = args.max_epochs
  CONFIG['start_epoch'] = args.start_epoch
  CONFIG['end_epoch'] = args.end_epoch
  CONFIG['dataset_info_csv'] = args.dataset_info_csv
  CONFIG['kaggle_feature_path'] = args.kaggle_feature_path
  CONFIG['verbose'] = args.verbose
  
  output_base_dir = args.output_base_directory
  
  CONFIG['directories']['save_base_directory'] = os.path.join(output_base_dir, 'train')
  CONFIG['directories']['create_splits_directory'] = os.path.join(output_base_dir, 'create_splits')
  CONFIG['directories']['create_patches_directory'] = os.path.join(output_base_dir, 'create_patches', 'patches')
  CONFIG['directories']['extract_patches_directory'] = os.path.join(output_base_dir, 'extract_patches')
  CONFIG['directories']['features_pt_directory'] = os.path.join(output_base_dir, 'extract_features', 'pt_files')

def main():
  logger.draw_header("Train MIL Model")
  load_arguments()
  
  logger.info("Training MIL model...")
  start_time = time.time()

  logger.info("Creating Directories...")
  create_directories(CONFIG['directories'])

  show_configs()

  start_k_fold = CONFIG['k_fold_start'] if CONFIG['k_fold_start'] != -1 else 0
  end_k_fold = CONFIG['k_fold_end'] if CONFIG['k_fold_end'] != -1 else CONFIG['k_fold']
  folds = np.arange(start_k_fold, end_k_fold, 1)

  if CONFIG['kaggle_feature_path'] is not None:
    __path = os.path.join(os.getcwd(), 'config.txt')
    with open(__path, 'w') as f:
      f.write(CONFIG['kaggle_feature_path'])
  
  logger.info("Loading dataset...")
  dataset = GenericMILDataset(
    patches_dir = CONFIG['directories']['create_patches_directory'],
    extract_patches_dir = CONFIG['directories']['extract_patches_directory'],
    features_pt_directory = CONFIG['directories']['features_pt_directory'],
    csv_path = CONFIG['dataset_info_csv'],
    label_column = 'isup_grade',
    label_dict = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5 },
    verbose = CONFIG['verbose']
  )
  
  all_test_auc = []
  all_val_auc = []
  all_test_acc = []
  all_val_acc = []
  all_test_f1 = []
  all_val_f1 = []
  all_test_precision = []
  all_val_precision = []
  all_test_recall = []
  all_val_recall = []
  all_test_kappa = []
  all_val_kappa = []

  if CONFIG['k_fold_start'] != -1 and CONFIG['k_fold_end'] != -1:
    for fold in range(0, CONFIG['k_fold']):
      filename = os.path.join(CONFIG['directories']['save_base_directory'], 'split_{}_metrics.pkl'.format(fold))
      if os.path.exists(filename):
        logger.info("Loading results from fold {}...", fold)
        results = load_pkl(filename)
        all_test_auc.append(results['test']['auc'])
        all_val_auc.append(results['val']['auc'])
        all_test_acc.append(results['test']['accuracy'])
        all_val_acc.append(results['val']['accuracy'])
        all_test_f1.append(results['test']['f1_macro'])
        all_val_f1.append(results['val']['f1_macro'])
        all_test_precision.append(results['test']['precision_macro'])
        all_val_precision.append(results['val']['precision_macro'])
        all_test_recall.append(results['test']['recall_macro'])
        all_val_recall.append(results['val']['recall_macro'])
        all_test_kappa.append(results['test']['cohens_kappa'])
        all_val_kappa.append(results['val']['cohens_kappa'])

  for fold in folds:
    logger.info("Training fold {}/{}...", fold + 1, CONFIG['k_fold'])
    fold_start_time = time.time()
    
    seed_torch()
    
    logger.info("Setting up dataset split for fold {}...", fold)
    dataset_split = dataset.return_splits(
      CONFIG['backbone'],
      CONFIG['patch_size'], 
      from_id = False, 
      csv_path='{}/splits_{}.csv'.format(CONFIG['directories']['create_splits_directory'], fold)
    )

    drop_out = 0.25 if CONFIG['drop_out'] else 0.0
    logger.info("Initializing training engine...")
    train_engine = TrainEngine(
      datasets = dataset_split,
      fold = fold,
      drop_out = drop_out,
      result_directory = CONFIG['directories']['save_base_directory'],
      mil_model_name = CONFIG['mil_model'],
      learning_rate = CONFIG['learning_rate'],
      max_epochs = CONFIG['max_epochs'],
      start_epoch = CONFIG['start_epoch'],
      end_epoch = CONFIG['end_epoch'],
      in_dim = CONFIG['in_dim'],
      n_classes = CONFIG['n_classes'],
      verbose = CONFIG['verbose'],
    )

    logger.info("Starting model training for fold {}...", fold)
    results, test_auc, val_auc, test_acc, val_acc, test_f1, val_f1, test_metrics, val_metrics = train_engine.train_model(fold)
    all_test_auc.append(test_auc)
    all_val_auc.append(val_auc)
    all_test_acc.append(test_acc)
    all_val_acc.append(val_acc)
    all_test_f1.append(test_f1)
    all_val_f1.append(val_f1)
    all_test_precision.append(test_metrics['precision_macro'])
    all_val_precision.append(val_metrics['precision_macro'])
    all_test_recall.append(test_metrics['recall_macro'])
    all_val_recall.append(val_metrics['recall_macro'])
    all_test_kappa.append(test_metrics['cohens_kappa'])
    all_val_kappa.append(val_metrics['cohens_kappa'])

    fold_metrics = {
      'test': test_metrics,
      'val': val_metrics
    }
    save_pkl(os.path.join(CONFIG['directories']['save_base_directory'], 'split_{}_metrics.pkl'.format(fold)), fold_metrics)
    
    filename = os.path.join(CONFIG['directories']['save_base_directory'], 'split_{}_results.pkl'.format(fold))
    save_pkl(filename, results)
    
    fold_time = time.time() - fold_start_time
    logger.info("Fold {}/{} completed in {:.2f} seconds ({:.2f} minutes)", fold + 1, CONFIG['k_fold'], fold_time, fold_time/60)
    logger.info("Fold {} results - Test AUC: {:.4f}, Val AUC: {:.4f}", fold, test_auc, val_auc)
    logger.empty_line()

  logger.info("Saving summary results...")
  final_df = pd.DataFrame({
    'folds': np.arange(0, CONFIG['k_fold_end'], 1),
    'test_auc': all_test_auc, 
    'val_auc': all_val_auc,
    'test_acc': all_test_acc,
    'val_acc': all_val_acc,
    'test_f1': all_test_f1,
    'val_f1': all_val_f1,
    'test_precision': all_test_precision,
    'val_precision': all_val_precision,
    'test_recall': all_test_recall,
    'val_recall': all_val_recall,
    'test_kappa': all_test_kappa,
    'val_kappa': all_val_kappa
  })

  if len(folds) != CONFIG['k_fold']:
    save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
  else:
    save_name = 'summary.csv'
  
  final_df.to_csv(os.path.join(CONFIG['directories']['save_base_directory'], save_name))
  
  total_time = time.time() - start_time
  logger.empty_line()
  logger.info("Training completed!")
  logger.info("Total folds processed: {}", len(folds))
  
  logger.info("===== Performance Metrics Summary =====")
  logger.info("--- Test Metrics ---")
  logger.info("Average Accuracy: {:.4f}", np.mean(all_test_acc))
  logger.info("Average AUC: {:.4f}", np.mean(all_test_auc))
  logger.info("Average F1 Score: {:.4f}", np.mean(all_test_f1))
  logger.info("Average Precision: {:.4f}", np.mean(all_test_precision))
  logger.info("Average Recall: {:.4f}", np.mean(all_test_recall))
  logger.info("Average Cohen's Kappa: {:.4f}", np.mean(all_test_kappa))
  
  logger.info("--- Validation Metrics ---")
  logger.info("Average Accuracy: {:.4f}", np.mean(all_val_acc))
  logger.info("Average AUC: {:.4f}", np.mean(all_val_auc))
  logger.info("Average F1 Score: {:.4f}", np.mean(all_val_f1))
  logger.info("Average Precision: {:.4f}", np.mean(all_val_precision))
  logger.info("Average Recall: {:.4f}", np.mean(all_val_recall))
  logger.info("Average Cohen's Kappa: {:.4f}", np.mean(all_val_kappa))
  
  logger.info("Total processing time: {:.2f} seconds ({:.2f} minutes)", 
              total_time, total_time/60)
  logger.success("MIL model training completed successfully!")

  __path = os.path.join(os.getcwd(), 'config.txt')
  if os.path.exists(__path):
    os.remove(__path)

if __name__ == '__main__':
  main()