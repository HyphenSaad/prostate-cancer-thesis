import warnings
warnings.filterwarnings('ignore')

import os
import sys
import h5py
import glob
import time
import argparse
import openslide
from multiprocessing.pool import Pool
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import DATASET_BASE_DIRECTORY, DATASET_SLIDES_FOLDER_NAME, OUTPUT_BASE_DIRECTORY
from utils.helper import create_directories
from utils.wsi_core.whole_slide_image import ImgReader
from utils.logger import Logger

logger = Logger()

CONFIG = {
  'slides_format': 'tiff',
  'patch_level': 0,
  'patch_size': 512,
  'num_workers': os.cpu_count(),
  'directories': {
    'slides_directory': os.path.join(DATASET_BASE_DIRECTORY, DATASET_SLIDES_FOLDER_NAME),
    'patches_h5_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches', 'patches'),
    'save_base_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_patches'),
  },
  'verbose': False
}

def get_wsi_handle(
  wsi_path,
  verbose = False
):
  if not os.path.exists(wsi_path):
    raise FileNotFoundError(f'{wsi_path} is not found')

  postfix = wsi_path.split('.')[-1]
  if postfix.lower() in ['svs', 'mrxs']:
    handle = openslide.OpenSlide(wsi_path)
  else:
    handle = ImgReader(wsi_path, verbose = verbose)

  return handle

def read_images(args):
  h5_path, save_root, wsi_path, size, level, verbose = args
  if wsi_path is None: return 0

  try: 
    h5 = h5py.File(h5_path)
    os.makedirs(save_root, exist_ok=True)
  except:
    if verbose: logger.error("{} is not readable....", h5_path)
    return 0
  
  if len(h5['coords']) == len(os.listdir(save_root)): 
    return len(h5['coords'])
  
  wsi_handle = get_wsi_handle(wsi_path, verbose)
  patch_count = 0
  for x, y in h5['coords']:
    patch_path = os.path.join(save_root, f'{x}_{y}_{size[0]}_{size[1]}.jpg')
    if os.path.exists(patch_path):
      patch_count += 1
      continue

    try:
      img = wsi_handle.read_region((x, y), level, size).convert('RGB')
      img.save(patch_path)
      patch_count += 1
    except:
      if verbose: logger.error("Failed to read: {}, {}, {}", wsi_path, x, y)

  return patch_count

def get_wsi_path(
  wsi_root,
  h5_files,
  wsi_format
):
  kv = {}
  all_paths = glob.glob(os.path.join(wsi_root, f'*.{wsi_format}'), recursive=True)
  for h5_file in h5_files:
    slide_id = os.path.basename(h5_file).split('.')[0]
    for path in all_paths:
      if slide_id in path:
        kv[h5_file] = path
        break

  wsi_paths = [kv[h5_file] for h5_file in h5_files]
  return wsi_paths

def show_configs():
  logger.empty_line()
  logger.info("Using Configurations;")
  logger.text(f"> Slides Format: {CONFIG['slides_format']}")
  logger.text(f"> Patch Level: {CONFIG['patch_level']}")
  logger.text(f"> Patch Size: {CONFIG['patch_size']}")
  logger.text(f"> Number of Threads: {CONFIG['num_workers']}")
  logger.empty_line()

def load_arguments():
  parser = argparse.ArgumentParser(description = "Extract Patches From Slides")
  parser.add_argument(
    "--slides-format",
    type = str,
    default = CONFIG['slides_format'],
    help = f"Slides Format (default: {CONFIG['slides_format']})"
  )
  parser.add_argument(
    "--patch-level",
    type = int,
    default = CONFIG['patch_level'],
    help = f"Patch Level (default: {CONFIG['patch_level']})"
  )
  parser.add_argument(
    "--patch-size",
    type = int,
    default = CONFIG['patch_size'],
    help = f"Patch Size (default: {CONFIG['patch_size']})"
  )
  parser.add_argument(
    "--num-workers",
    type = int,
    default = CONFIG['num_workers'],
    help = f"Number of Workers (default: {CONFIG['num_workers']})"
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
    "--dataset-slides-folder-name",
    type = str,
    default = DATASET_SLIDES_FOLDER_NAME,
    help = f"Dataset Slides Folder Name (default: {DATASET_SLIDES_FOLDER_NAME})"
  )
  parser.add_argument(
    "--output-base-directory",
    type = str,
    default = OUTPUT_BASE_DIRECTORY,
    help = f"Output Base Directory (default: {OUTPUT_BASE_DIRECTORY})"
  )

  args = parser.parse_args()
  
  CONFIG['slides_format'] = args.slides_format
  CONFIG['patch_level'] = args.patch_level
  CONFIG['patch_size'] = args.patch_size
  CONFIG['num_workers'] = args.num_workers
  CONFIG['verbose'] = args.verbose
  
  dataset_base_dir = args.dataset_base_directory
  dataset_slides_folder = args.dataset_slides_folder_name
  output_base_dir = args.output_base_directory
  
  CONFIG['directories']['slides_directory'] = os.path.join(dataset_base_dir, dataset_slides_folder)
  CONFIG['directories']['patches_h5_directory'] = os.path.join(output_base_dir, 'create_patches', 'patches')
  CONFIG['directories']['save_base_directory'] = os.path.join(output_base_dir, 'extract_patches')

def main():
  logger.draw_header("Extract Patches From Slides")
  load_arguments()
  
  logger.info("Extracting Patches From Slides...")
  start_time = time.time()

  logger.info("Creating Directories...")
  create_directories(CONFIG['directories'])

  show_configs()
  
  h5_files = glob.glob(os.path.join(CONFIG['directories']['patches_h5_directory'], '*.h5'))
  wsi_paths = get_wsi_path(
    CONFIG['directories']['slides_directory'],
    h5_files,
    CONFIG['slides_format']
  )

  args = [
    (
      h5_file, 
      os.path.join(
        CONFIG['directories']['save_base_directory'], 
        os.path.basename(h5_file).split('.')[0]
      ),
      wsi_path,
      (CONFIG['patch_size'], CONFIG['patch_size']),
      CONFIG['patch_level'],
      CONFIG['verbose']
    )
    for h5_file, wsi_path in zip(h5_files, wsi_paths)
  ]

  logger.info("Using {} workers to extract patches from slides...\n", CONFIG["num_workers"])
  with Pool(CONFIG['num_workers']) as p:
    patch_counts = list(tqdm(
      p.imap_unordered(read_images, args),
      total=len(args),
      desc="Extracting Patches",
      unit="slide"
    ))
  
  end_time = time.time()
  total_time = end_time - start_time
  slide_count = len(args)
  valid_counts = [count for count in patch_counts if count is not None]

  total_patches = sum(valid_counts)
  avg_patches_per_slide = total_patches / slide_count if slide_count > 0 else 0
  avg_time_per_slide = total_time / slide_count if slide_count > 0 else 0

  logger.empty_line()
  logger.info("Total Slides Processed: {}", slide_count)
  logger.info("Total Patches Extracted: {}", total_patches)
  logger.info("Average Patches Per Slide: {:.2f}", avg_patches_per_slide)
  logger.info("Total Processing Time: {:.2f} seconds ({:.2f} minutes)", total_time, total_time/60)
  logger.info("Average Time Per Slide: {:.2f} seconds", avg_time_per_slide)
  logger.empty_line()
  logger.success("Extracted All Patches From Slides!")

if __name__ == '__main__':
  main()