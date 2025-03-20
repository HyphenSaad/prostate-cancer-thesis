import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import time
import h5py
import pandas as pd
import random
import argparse
from PIL import Image
from multiprocessing import Process
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME, OUTPUT_BASE_DIRECTORY
from constants.encoders import Encoders
from encoders import get_encoder, get_custom_transformer
from utils.helper import create_directories
from utils.logger import Logger

logger = Logger()

CONFIG = {
  'encoder': Encoders.RESNET50.value,
  'batch_size': 128,
  'patch_size': 512,
  'dataset_info_csv': os.path.join(DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME),
  'processed_dataset_info_csv': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_features', 'processed_dataset_info.csv'),
  'directories': {
    'patches_h5_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches', 'patches'),
    'extract_patches_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_patches'),
    'save_base_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_features'),
    'features_pt_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_features', 'pt_files'),
  },
  'verbose': False
}

def prepare_csv(
  label_csv_path: str,
  patches_h5_path: str,
  csv_save_path: str
) -> None:
  random.seed(0)
  data_items = []
  
  df = pd.read_csv(label_csv_path)
  for idx, row in df.iterrows():
    slide_id = row['image_id']
    p = os.path.join(patches_h5_path, f'{slide_id}.h5')
    if os.path.exists(p):
      data_items.append([slide_id, row['isup_grade']])

  random.shuffle(data_items)

  new_df_columns = ['dir', 'case_id', 'slide_id', 'label']
  new_df = pd.DataFrame(columns = new_df_columns)

  rows = []
  for slide_id, isup in data_items:
    rows.append({
      'dir': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches'),
      'case_id': slide_id,
      'slide_id': slide_id,
      'label': isup
    })
  
  new_df = pd.concat([new_df, pd.DataFrame(rows)], ignore_index=True)
  new_df.to_csv(csv_save_path, index = False)

class DatasetAllBags(Dataset):
	def __init__(self, csv_path):
		self.df = pd.read_csv(
      csv_path,
      dtype = { 'case_id': str, 'slide_id': str }
    )
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['dir'][idx], self.df['slide_id'][idx]

class PatchDataset(Dataset):
  def __init__(
    self,
    img_root,
    patch_h5_path,
    transform = None
  ) -> None:
    super().__init__()

    self.img_root = img_root
    self.transform = transform
    
    with h5py.File(patch_h5_path, 'r') as hf:
      self.coords = hf['coords'][()]
    
    actual_files = os.listdir(img_root)
    assert len(actual_files) + 1 >= len(self.coords), \
      'real patch {} not match h5 patch number {}'.format(len(actual_files), len(self.coords))

  def __len__(self):
    return len(self.coords)

  def __getitem__(self, index):
    x, y = self.coords[index]
    
    img_name = f'{x}_{y}_{CONFIG["patch_size"]}_{CONFIG["patch_size"]}.jpg'
    image_path = os.path.join(self.img_root, img_name)
    img = Image.open(image_path)
    
    if self.transform is not None:
      img = self.transform(img)
      
    return img

def save_feature(path, feature):
  s = time.time()
  torch.save(feature, path)
  e = time.time()
  if CONFIG['verbose']: logger.info(f'Feature Sucessfully Saved At: {path}, Time: {e-s:.1f}s')

def save_feature_subprocess(path, feature):
  kwargs = {
    'feature': feature, 
    'path': path
  }
  
  process = Process(target = save_feature, kwargs = kwargs)
  process.start()

def light_compute_w_loader(
  loader,
  encoder,
  device,
  print_every = 20,
):
  features_list = []
  _start_time = time.time()
  for count, batch in enumerate(loader):
    with torch.no_grad():	
      if count % print_every == 0:
        batch_time = time.time()
        if CONFIG['verbose']: 
          logger.info(f'batch {count}/{len(loader)}, {count * len(batch)} files processed, used_time: {batch_time - _start_time:.2f}s')

      batch = batch.to(device, non_blocking=True)
      features = encoder(batch)
      features = features.cpu()
      features_list.append(features)

  features = torch.cat(features_list, dim=0)
  return features

def show_configs():
  logger.empty_line()
  logger.info("Using Configurations;")
  logger.text(f"> Encoder: {CONFIG['encoder']}")
  logger.text(f"> Batch Size: {CONFIG['batch_size']}")
  logger.text(f"> Patch Size: {CONFIG['patch_size']}")
  logger.empty_line()

def load_arguments():
  parser = argparse.ArgumentParser(description = "Extract Features From Patches")
  parser.add_argument(
    "--encoder",
    type = str,
    default = CONFIG['encoder'],
    help = f"Encoder Model Name (default: {CONFIG['encoder']}) ({', '.join([e.value for e in Encoders])})"
  )
  parser.add_argument(
    "--batch-size",
    type = int,
    default = CONFIG['batch_size'],
    help = f"Batch Size (default: {CONFIG['batch_size']})"
  )
  parser.add_argument(
    "--patch-size",
    type = int,
    default = CONFIG['patch_size'],
    help = f"Patch Size (default: {CONFIG['patch_size']})"
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
    "--dataset-info-file-name",
    type = str,
    default = DATASET_INFO_FILE_NAME,
    help = f"Dataset Info File Name (default: {DATASET_INFO_FILE_NAME})"
  )
  parser.add_argument(
    "--output-base-directory",
    type = str,
    default = OUTPUT_BASE_DIRECTORY,
    help = f"Output Base Directory (default: {OUTPUT_BASE_DIRECTORY})"
  )

  args = parser.parse_args()

  if args.encoder not in [e.value for e in Encoders]:
    raise ValueError(f"Invalid Encoder: {args.encoder}")
  
  CONFIG['encoder'] = args.encoder
  CONFIG['batch_size'] = args.batch_size
  CONFIG['patch_size'] = args.patch_size
  CONFIG['verbose'] = args.verbose
  
  dataset_base_dir = args.dataset_base_directory
  dataset_info_file = args.dataset_info_file_name
  output_base_dir = args.output_base_directory
  
  CONFIG['dataset_info_csv'] = os.path.join(dataset_base_dir, dataset_info_file)
  CONFIG['processed_dataset_info_csv'] = os.path.join(output_base_dir, 'extract_features', 'processed_dataset_info.csv')
  CONFIG['directories']['patches_h5_directory'] = os.path.join(output_base_dir, 'create_patches', 'patches')
  CONFIG['directories']['extract_patches_directory'] = os.path.join(output_base_dir, 'extract_patches')
  CONFIG['directories']['save_base_directory'] = os.path.join(output_base_dir, 'extract_features')
  CONFIG['directories']['features_pt_directory'] = os.path.join(output_base_dir, 'extract_features', 'pt_files')

def main():
  logger.draw_header("Extract Features From Patches")
  load_arguments()
  
  logger.info("Extracting Features From Patches...")
  process_start_time = time.time()

  logger.info("Creating Directories...")
  create_directories(CONFIG['directories'])

  logger.info("Preparing Processed Dataset Info CSV...")
  prepare_csv(
    label_csv_path = CONFIG['dataset_info_csv'],
    patches_h5_path = CONFIG['directories']['patches_h5_directory'],
    csv_save_path = CONFIG['processed_dataset_info_csv']
  )

  show_configs()
  
  dataset_bags = DatasetAllBags(CONFIG['processed_dataset_info_csv'])

  encoder_pt_path = os.path.join(CONFIG['directories']['features_pt_directory'], CONFIG['encoder'])
  os.makedirs(encoder_pt_path, exist_ok = True)
  destination_files = os.listdir(encoder_pt_path)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(f'Using Device: {device}' + (f' ({torch.cuda.device_count()} GPUs)' if torch.cuda.is_available() else ""))

  encoder = get_encoder(CONFIG['encoder'], device, torch.cuda.device_count())
  custom_transformer = get_custom_transformer(CONFIG['encoder'])

  exiting_indexes = []
  for bag_candidate_index in range(len(dataset_bags)):
    dataset_dir, slide_id = dataset_bags[bag_candidate_index]
    bag_name = f'{slide_id}.h5'
    h5_file_path = os.path.join(CONFIG['directories']['patches_h5_directory'], bag_name)
    if not os.path.exists(h5_file_path):
      if CONFIG['verbose']: logger.error(f'{h5_file_path} does not exist...')
      continue
    elif f'{slide_id}.pt' in destination_files:
      if CONFIG['verbose']: logger.info(f'Skipped {slide_id}')
      continue
    else:
      exiting_indexes.append(bag_candidate_index)

  total_features = 0
  total_slides_processed = 0

  logger.empty_line()
  for bag_candidate_index in tqdm(exiting_indexes, desc="Extracting Features", unit="slide"):
    dataset_dir, slide_id = dataset_bags[bag_candidate_index]
    if CONFIG['verbose']: logger.info(f'\nSlide ID: {slide_id}')
    
    output_feature_path = os.path.join(encoder_pt_path, f'{slide_id}.pt')
    h5_file_path = os.path.join(CONFIG['directories']['patches_h5_directory'], f'{slide_id}.h5')

    one_slide_start = time.time()

    if not os.path.exists(h5_file_path):
      if CONFIG['verbose']: logger.error(f'{h5_file_path} does not exist, skipping...')
      continue

    images_path = os.path.join(CONFIG['directories']['extract_patches_directory'], slide_id)
    patch_dataset = PatchDataset(images_path, h5_file_path, transform = custom_transformer)
    loader = DataLoader(
      patch_dataset,
      batch_size = CONFIG['batch_size'],
      shuffle = False,
      num_workers = os.cpu_count()
    )

    # created a temporary file to help other processes
    with open(f'{output_feature_path}.partial', 'w') as f: 
      f.write('')

    features = light_compute_w_loader(
      loader = loader,
      encoder = encoder,
      device = device
    )

    save_feature_subprocess(output_feature_path, features)
    os.remove(f'{output_feature_path}.partial')
    
    if CONFIG['verbose']: logger.info(f'Coords Shape: {features.shape}')
    if CONFIG['verbose']: logger.info(f'Elapsed Time: {time.time() - one_slide_start:.2f}s')
    
    total_features += features.shape[0]
    total_slides_processed += 1

  end_time = time.time()
  total_time = end_time - process_start_time
  avg_time_per_slide = total_time / total_slides_processed if total_slides_processed > 0 else 0

  logger.empty_line()
  logger.info("Total Slides Processed: {}", total_slides_processed)
  logger.info("Total Processing Time: {:.2f} minutes", total_time / 60)
  logger.info("Average Time Per Slide: {:.2f} seconds", avg_time_per_slide)
  logger.empty_line()
  logger.success("Extracted All Features!")

if __name__ == '__main__':
  try: main()
  except Exception as e: logger.error(e)