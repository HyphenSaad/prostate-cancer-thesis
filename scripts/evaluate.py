import warnings
warnings.filterwarnings('ignore')

import os 
import sys
import time
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import OUTPUT_BASE_DIRECTORY, DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME
from constants.mil_models import MILModels
from constants.encoders import Encoders
from utils.helper import create_directories
from utils.data_loader import GenericMILDataset
from utils.train_engine import TrainEngine
from utils.file_utils import save_pkl, save_json
from utils.logger import Logger

logger = Logger()

CONFIG = {
  'backbone': Encoders.RESNET50.value,
  'mil_model': MILModels.TRANS_MIL.value,
  'drop_out': True,
  'n_classes': 6,
  'learning_rate': 1e-4,
  'fold': 0,
  'patch_size': 512,
  'in_dim': 1024,
  'dataset_info_csv': os.path.join(DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME),
  'checkpoint_path': None, 
  'verbose': False,
  'save_predictions': True,
  'save_attention_maps': False,
  'directories': {
    'results_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'evaluation'),
    'train_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'train'),
    'create_splits_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_splits'),
    'create_patches_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches', 'patches'),
    'extract_patches_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_patches'),
    'features_pt_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_features', 'pt_files'),
  }
}

def show_configs():
  logger.empty_line()
  logger.info("Using Configurations:")
  logger.text(f"> Backbone: {CONFIG['backbone']}")
  logger.text(f"> MIL Model: {CONFIG['mil_model']}")
  logger.text(f"> Drop Out: {CONFIG['drop_out']}")
  logger.text(f"> Number of Classes: {CONFIG['n_classes']}")
  logger.text(f"> Learning Rate: {CONFIG['learning_rate']}")
  logger.text(f"> Fold: {CONFIG['fold']}")
  logger.text(f"> Patch Size: {CONFIG['patch_size']}")
  logger.text(f"> Input Dimension: {CONFIG['in_dim']}")
  logger.text(f"> Dataset Info CSV: {CONFIG['dataset_info_csv']}")
  logger.text(f"> Checkpoint Path: {CONFIG['checkpoint_path']}")
  logger.text(f"> Verbose: {CONFIG['verbose']}")
  logger.text(f"> Save Predictions: {CONFIG['save_predictions']}")
  logger.text(f"> Save Attention Maps: {CONFIG['save_attention_maps']}")
  logger.empty_line()

def load_arguments():
  parser = argparse.ArgumentParser(description="Evaluate MIL Models")
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
    "--fold",
    type=int,
    default=CONFIG['fold'],
    help=f"Fold to evaluate (default: {CONFIG['fold']})"
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
    "--dataset-info-csv",
    type=str,
    default=CONFIG['dataset_info_csv'],
    help=f"Dataset info CSV path (default: {CONFIG['dataset_info_csv']})"
  )
  parser.add_argument(
    "--checkpoint-path",
    type=str,
    default=CONFIG['checkpoint_path'],
    help="Path to model checkpoint (default: auto-generated based on other parameters)"
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
  parser.add_argument(
    "--save-predictions",
    type=bool,
    default=CONFIG['save_predictions'],
    help=f"Save individual predictions (default: {CONFIG['save_predictions']})"
  )
  parser.add_argument(
    "--save-attention-maps",
    type=bool,
    default=CONFIG['save_attention_maps'],
    help=f"Save attention maps for interpretability (default: {CONFIG['save_attention_maps']})"
  )
  
  args = parser.parse_args()
  
  CONFIG['backbone'] = args.backbone
  CONFIG['mil_model'] = args.mil_model
  CONFIG['drop_out'] = args.drop_out
  CONFIG['n_classes'] = args.n_classes
  CONFIG['learning_rate'] = args.learning_rate
  CONFIG['fold'] = args.fold
  CONFIG['patch_size'] = args.patch_size
  CONFIG['in_dim'] = args.in_dim
  CONFIG['dataset_info_csv'] = args.dataset_info_csv
  CONFIG['checkpoint_path'] = args.checkpoint_path
  CONFIG['verbose'] = args.verbose
  CONFIG['save_predictions'] = args.save_predictions
  CONFIG['save_attention_maps'] = args.save_attention_maps
  
  output_base_dir = args.output_base_directory
  
  CONFIG['directories']['results_directory'] = os.path.join(output_base_dir, 'evaluation')
  CONFIG['directories']['train_directory'] = os.path.join(output_base_dir, 'train')
  CONFIG['directories']['create_splits_directory'] = os.path.join(output_base_dir, 'create_splits')
  CONFIG['directories']['create_patches_directory'] = os.path.join(output_base_dir, 'create_patches', 'patches')
  CONFIG['directories']['extract_patches_directory'] = os.path.join(output_base_dir, 'extract_patches')
  CONFIG['directories']['features_pt_directory'] = os.path.join(output_base_dir, 'extract_features', 'pt_files')

  # If checkpoint_path not provided, generate based on other parameters
  if CONFIG['checkpoint_path'] is None:
    CONFIG['checkpoint_path'] = os.path.join(
      CONFIG['directories']['train_directory'],
      f"s_{CONFIG['fold']}_checkpoint.pt"
    )

def plot_confusion_matrix(cm, classes, save_path):
  plt.figure(figsize=(10, 8))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
  disp.plot(cmap=plt.cm.Blues, values_format='d')
  plt.title('Confusion Matrix')
  plt.savefig(save_path, bbox_inches='tight')
  plt.close()

def main():
  logger.draw_header("Evaluate MIL Model")
  load_arguments()
  
  logger.info("Evaluating MIL model...")
  start_time = time.time()

  logger.info("Creating directories...")
  create_directories(CONFIG['directories'])
  
  model_output_dir = os.path.join(
    CONFIG['directories']['results_directory'],
    f"{CONFIG['mil_model']}_{CONFIG['backbone']}_fold_{CONFIG['fold']}"
  )
  os.makedirs(model_output_dir, exist_ok=True)

  show_configs()

  if not os.path.exists(CONFIG['checkpoint_path']):
    logger.error(f"Checkpoint not found at: {CONFIG['checkpoint_path']}")
    logger.error("Please provide a valid checkpoint path or ensure the model was trained")
    return
  
  logger.info(f"Loading dataset for fold {CONFIG['fold']}...")
  dataset = GenericMILDataset(
    patches_dir = CONFIG['directories']['create_patches_directory'],
    extract_patches_dir = CONFIG['directories']['extract_patches_directory'],
    features_pt_directory = CONFIG['directories']['features_pt_directory'],
    csv_path = CONFIG['dataset_info_csv'],
    label_column = 'isup_grade',
    label_dict = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5 },
    verbose = CONFIG['verbose'],
  )
  
  logger.info(f"Setting up dataset split for fold {CONFIG['fold']}...")
  dataset_split = dataset.return_splits(
    CONFIG['backbone'],
    CONFIG['patch_size'], 
    from_id = False, 
    csv_path='{}/splits_{}.csv'.format(CONFIG['directories']['create_splits_directory'], CONFIG['fold'])
  )
  
  drop_out = 0.25 if CONFIG['drop_out'] else 0.0
  logger.info("Initializing evaluation engine...")
  train_engine = TrainEngine(
    datasets = dataset_split,
    fold = CONFIG['fold'],
    drop_out = drop_out,
    result_directory = CONFIG['directories']['train_directory'],
    mil_model_name = CONFIG['mil_model'],
    learning_rate = CONFIG['learning_rate'],
    max_epochs = 1,
    in_dim = CONFIG['in_dim'],
    n_classes = CONFIG['n_classes'],
    verbose = CONFIG['verbose'],
  )

  logger.info(f"Evaluating model from checkpoint: {CONFIG['checkpoint_path']}")
  _, test_error, test_auc, acc_logger, df, test_f1, test_metrics = train_engine.eval_model(CONFIG['checkpoint_path'])
  
  if CONFIG['save_predictions']:
    predictions_path = os.path.join(model_output_dir, "predictions.csv")
    logger.info(f"Saving predictions to {predictions_path}")
    df.to_csv(predictions_path, index=False)
  
  metrics_path = os.path.join(model_output_dir, "metrics.json")
  metrics_to_save = {
    'accuracy': 1 - test_error,
    'error': test_error,
    'auc': test_auc,
    'f1_score': test_f1,
    'precision': test_metrics['precision_macro'],
    'recall': test_metrics['recall_macro'],
    'cohens_kappa': test_metrics['cohens_kappa'],
  }
  
  metrics_to_save['class_metrics'] = {}
  for i in range(CONFIG['n_classes']):
    class_metrics = {}
    acc, correct, count = acc_logger.get_summary(i)
    class_metrics['accuracy'] = acc
    class_metrics['count'] = count
    
    if 'f1_per_class' in test_metrics and i < len(test_metrics['f1_per_class']):
      class_metrics['f1'] = float(test_metrics['f1_per_class'][i])
    
    if 'precision_per_class' in test_metrics and i < len(test_metrics['precision_per_class']):
      class_metrics['precision'] = float(test_metrics['precision_per_class'][i])
    
    if 'recall_per_class' in test_metrics and i < len(test_metrics['recall_per_class']):
      class_metrics['recall'] = float(test_metrics['recall_per_class'][i])
    
    metrics_to_save['class_metrics'][f'class_{i}'] = class_metrics
  
  logger.info(f"Saving metrics to {metrics_path}")
  save_json(metrics_path, metrics_to_save)
  
  if 'confusion_matrix' in test_metrics:
    cm_path = os.path.join(model_output_dir, "confusion_matrix.png")
    logger.info(f"Saving confusion matrix to {cm_path}")
    class_names = [str(i) for i in range(CONFIG['n_classes'])]
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, cm_path)

  full_metrics_path = os.path.join(model_output_dir, "full_metrics.pkl")
  logger.info(f"Saving full metrics to {full_metrics_path}")
  save_pkl(full_metrics_path, test_metrics)
  
  end_time = time.time()
  total_time = end_time - start_time
  
  logger.empty_line()
  logger.info("===== Evaluation Results Summary =====")
  logger.info(f"Model: {CONFIG['mil_model']}, Backbone: {CONFIG['backbone']}, Fold: {CONFIG['fold']}")
  logger.success(f"Accuracy: {1 - test_error:.4f}")
  logger.success(f"AUC: {test_auc:.4f}")
  logger.success(f"F1 Score: {test_f1:.4f}")
  logger.success(f"Precision: {test_metrics['precision_macro']:.4f}")
  logger.success(f"Recall: {test_metrics['recall_macro']:.4f}")
  logger.success(f"Cohen's Kappa: {test_metrics['cohens_kappa']:.4f}")
  
  logger.info("Class-specific results:")
  for i in range(CONFIG['n_classes']):
    acc, correct, count = acc_logger.get_summary(i)
    logger.info(f"  Class {i}: Accuracy {acc:.4f}, Correct {correct}/{count}")
  
  logger.info(f"Evaluation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
  logger.success("Evaluation completed successfully!")

if __name__ == '__main__':
  main()