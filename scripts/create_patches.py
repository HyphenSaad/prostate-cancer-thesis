import warnings
warnings.filterwarnings('ignore')

import os 
import sys
import time
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import DATASET_BASE_DIRECTORY, OUTPUT_BASE_DIRECTORY
from constants.configs import CREATE_PATCHES_PRESET
from utils.helper import create_directories
from utils.wsi_core.batch_process_utils import initialize_dataframe
from utils.wsi_core.whole_slide_image import WholeSlideImage
from utils.wsi_core.wsi_utils import stitch_coords
from utils.logger import Logger

logger = Logger()

CONFIG = {
  'slides_format': 'tiff',
  'patch_level': 0,
  'patch_size': 512,
  'stride_size': 512,
  'skip_existing': True,
  'directories': {
    'slides_directory': os.path.join(DATASET_BASE_DIRECTORY, 'slides'),
    'save_base_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches'),
    'patches_save_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches', 'patches'),
    'masks_save_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches', 'masks'),
    'stitches_save_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches', 'stitches'),
  },
  'verbose': False,
}

def transform_presets(preset: dict) -> tuple:
  seg_params = {
    'seg_level': int(preset['seg_level']),
    'sthresh': int(preset['sthresh']),
    'mthresh': int(preset['mthresh']),
    'close': int(preset['close']),
    'use_otsu': bool(preset['use_otsu']),
    'keep_ids': preset['keep_ids'],
    'exclude_ids': preset['exclude_ids'],
  }

  filter_params = {
    'a_t': int(preset['a_t']),
    'a_h': int(preset['a_h']),
    'max_n_holes': int(preset['max_n_holes']),
  }

  vis_params = {
    'vis_level': int(preset['vis_level']),
    'line_thickness': int(preset['line_thickness']),
  }

  patch_params = {
    'use_padding': bool(preset['use_padding']),
    'contour_fn': preset['contour_fn'],
  }

  return seg_params, filter_params, vis_params, patch_params

def stitching(
  file_path,
  wsi_object: WholeSlideImage,
  downscale = 64,
  verbose = False
):
	start = time.time()

	heatmap = stitch_coords(
    file_path,
    wsi_object,
    downscale = downscale,
    bg_color = (0, 0, 0),
    alpha = -1,
    draw_grid = False,
    verbose = verbose
  )

	total_time = time.time() - start	
	return heatmap, total_time

def segment(
  wsi_object: WholeSlideImage,
  seg_params = None,
  filter_params = None,
  mask_file = None
):
	start_time = time.time()

	if mask_file is not None: 
		wsi_object.init_segmentation(mask_file)
	else:
		wsi_object.segment_tissue(**seg_params, filter_params = filter_params)

	seg_time_elapsed = time.time() - start_time
	return wsi_object, seg_time_elapsed

def patching(
  wsi_object: WholeSlideImage,
  **kwargs
):
	start_time = time.time()
	file_path = wsi_object.process_contours(**kwargs)
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed

def segment_and_patch(
  slides_directory: str,
  save_base_directory: str,
  patches_save_directory: str,
  masks_save_directory: str,
  stitches_save_directory: str,
  patch_size: int = 512,
  step_size: int = 512,
  preset: dict = {
    'seg_level': -1,
    'sthresh': 8,
    'mthresh': 7, 
    'close': 4,
    'use_otsu': False,
    'keep_ids': 'none', 
    'exclude_ids': 'none',
    'a_t': 100,
    'a_h': 16,
    'max_n_holes': 8,
    'vis_level': -1,
    'line_thickness': 500,
    'use_padding': True,
    'contour_fn': 'four_pt',
  },
  patch_level: int = 0,
  slides_format: str = 'tiff',
  skip_existing: bool = True,
  do_segmentation: bool = True,
  do_save_masks: bool = True,
  do_patch: bool = True,
  do_stitching: bool = True,
  verbose: bool = False
) -> dict:
  slides = []
  for root, dirs, filenames in os.walk(slides_directory):
    for filename in filenames:
      if filename.endswith(slides_format):
        slides.append(os.path.join(root, filename))

  seg_params, filter_params, vis_params, patch_params = transform_presets(preset)

  df = initialize_dataframe(
    slides = slides,
    seg_params = seg_params,
    filter_params = filter_params,
    vis_params = vis_params,
    patch_params = patch_params,
  )

  process_stack = df[df['process'] == 1]
  total_slides = len(process_stack)

  times = {
    'segmentation': 0,
    'patching': 0,
    'stitching': 0,
  }
  
  counts = {
    'segmentation': 0,
    'patching': 0,
    'stitching': 0,
  }
  
  start_time = time.time()

  for i in tqdm(range(total_slides), desc="Creating Patches", unit="slide"):
    process_list_autogen_path = os.path.join(save_base_directory, 'process_list_autogen.csv')
    df.to_csv(process_list_autogen_path, index=False)

    index = process_stack.index[i]
    slide_id = os.path.basename(process_stack.loc[index, 'slide_id']).split('.')[0]
    if verbose: logger.info(f'Processing Slide ID: {slide_id} ({i + 1}/{total_slides})')

    df.loc[index, 'process'] = 0

    if skip_existing and os.path.isfile(os.path.join(patches_save_directory, slide_id + '.h5')):
      if verbose:  logger.warning(f'(SKIPPED) {slide_id} already exists in destination location!')
      df.loc[index, 'status'] = 'already_exist'
      continue

    full_slide_path = os.path.join(slides_directory, slide_id + '.' + slides_format)
    try:
      wsi_object = WholeSlideImage(full_slide_path, verbose)
    except:
      if verbose: logger.error(f'{slide_id} failed to load!')
      continue

    current_vis_params = {}
    for key in vis_params.keys():
      current_vis_params.update({ key: df.loc[index, key] })
    
    current_filter_params = {}
    for key in filter_params.keys():
      current_filter_params.update({ key: df.loc[index, key] })
    
    current_seg_params = {}
    for key in seg_params.keys():
      current_seg_params.update({ key: df.loc[index, key] })
      
    current_patch_params = {}
    for key in patch_params.keys():
      current_patch_params.update({ key: df.loc[index, key] })

    if current_vis_params['vis_level'] < 0:
      if len(wsi_object.level_dim) == 1:
        current_vis_params['vis_level'] = 0
      else:
        wsi = wsi_object.get_open_slide()
        best_level = wsi.get_best_level_for_downsample(64)
        current_vis_params['vis_level'] = best_level

    if current_seg_params['seg_level'] < 0:
      if len(wsi_object.level_dim) == 1:
        current_seg_params['seg_level'] = 0
      else:
        wsi = wsi_object.get_open_slide()
        best_level = wsi.get_best_level_for_downsample(64)
        current_seg_params['seg_level'] = best_level

    keep_ids = str(current_seg_params['keep_ids'])
    if keep_ids != 'none' and len(keep_ids) > 0:
      str_ids = current_seg_params['keep_ids']
      current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
    else: current_seg_params['keep_ids'] = []

    exclude_ids = str(current_seg_params['exclude_ids'])
    if exclude_ids != 'none' and len(exclude_ids) > 0:
      str_ids = current_seg_params['exclude_ids']
      current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
    else: current_seg_params['exclude_ids'] = []

    w, h = wsi_object.level_dim[current_seg_params['seg_level']]
    if w * h > 1e10: # 10 billion pixels
      if verbose: logger.error('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
      df.loc[index, 'status'] = 'failed_seg'
      continue

    df.loc[index, 'vis_level'] = current_vis_params['vis_level']
    df.loc[index, 'seg_level'] = current_seg_params['seg_level']

    seg_time_elapsed = 0
    if do_segmentation:
      try:
        wsi_object, seg_time_elapsed = segment(wsi_object, current_seg_params, current_filter_params)
        seg_time_elapsed = max(0, seg_time_elapsed)
        counts['segmentation'] += 1
      except Exception as e:
        if verbose: logger.error(str(e))
        if verbose: logger.error('OpenSlideError, skipped')
        df.loc[index, 'status'] = 'failed_seg'
        continue

    if do_save_masks:
      mask = wsi_object.vis_wsi(**current_vis_params)
      mask_path = os.path.join(masks_save_directory, slide_id + '.jpg')
      if process_stack.loc[index, 'slide_id'].split('.')[-1] in ['jpg']:
        mask = mask.resize([i // 8 for i in mask.size])
      mask.save(mask_path)

    patch_time_elapsed = 0
    if do_patch:
      current_patch_params.update({
        'patch_level': patch_level,
        'patch_size': patch_size,
        'step_size': step_size,
        'save_path': patches_save_directory,
      })
      file_path, patch_time_elapsed = patching(wsi_object, **current_patch_params)
      patch_time_elapsed = max(0, patch_time_elapsed)
      counts['patching'] += 1

    stitch_time_elapsed = 0
    if do_stitching:
      file_path = os.path.join(patches_save_directory, slide_id + '.h5')
      if os.path.isfile(file_path):
        heatmap, stitch_time_elapsed = stitching(file_path, wsi_object, downscale=64, verbose=verbose)
        stitch_time_elapsed = max(0, stitch_time_elapsed)
        counts['stitching'] += 1
        stitch_path = os.path.join(stitches_save_directory, slide_id + '.jpg')
        if process_stack.loc[index, 'slide_id'].split('.')[-1] in ['jpg']:
          heatmap = heatmap.resize([i // 8 for i in heatmap.size])
        heatmap.save(stitch_path)

    if verbose: logger.info(f'\tSegmentation: {seg_time_elapsed:.2f}s')
    if verbose: logger.info(f'\tPatching: {patch_time_elapsed:.2f}s')
    if verbose: logger.info(f'\tStitching: {stitch_time_elapsed:.2f}s')
    df.loc[index, 'status'] = 'processed'

    times['segmentation'] += seg_time_elapsed
    times['patching'] += patch_time_elapsed
    times['stitching'] += stitch_time_elapsed

  end_time = time.time()
  total_time = end_time - start_time

  avg_seg_time = times['segmentation'] / counts['segmentation'] if counts['segmentation'] > 0 else 0
  avg_patch_time = times['patching'] / counts['patching'] if counts['patching'] > 0 else 0
  avg_stitch_time = times['stitching'] / counts['stitching'] if counts['stitching'] > 0 else 0

  df.to_csv(process_list_autogen_path, index=False)
  
  logger.empty_line()
  logger.info("Total Slides Processed: {}", total_slides)
  if counts['segmentation'] > 0:
    logger.info("Segmentation Average Time: {:.2f} seconds ({} slides)", avg_seg_time, counts["segmentation"])
  if counts['patching'] > 0:
    logger.info("Patching Average Time: {:.2f} seconds ({} slides)", avg_patch_time, counts["patching"])
  if counts['stitching'] > 0:
    logger.info("Stitching Average Time: {:.2f} seconds ({} slides)", avg_stitch_time, counts["stitching"])
  logger.info("Total Processing Time: {:.2f} seconds ({:.2f} minutes)", total_time, total_time/60)
  logger.empty_line()
  logger.success("Created Patches Successfully!")

  return {
    'segmentation': avg_seg_time,
    'patching': avg_patch_time, 
    'stitching': avg_stitch_time
  }

def show_configs():
  logger.empty_line()
  logger.info("Using Configurations;")
  logger.text(f"> Slides Format: {CONFIG['slides_format']}")
  logger.text(f"> Patch Level: {CONFIG['patch_level']}")
  logger.text(f"> Patch Size: {CONFIG['patch_size']}")
  logger.text(f"> Stride Size: {CONFIG['stride_size']}")
  logger.empty_line()

def main():
  logger.draw_header("Create Patches From Slides")
  
  logger.info("Creating Directories...")
  create_directories(CONFIG['directories'])

  show_configs()
  
  logger.info("Starting Patch Creation Process...")
  logger.empty_line()
  
  segment_and_patch(
    slides_directory = CONFIG['directories']['slides_directory'],
    save_base_directory = CONFIG['directories']['save_base_directory'],
    patches_save_directory = CONFIG['directories']['patches_save_directory'],
    masks_save_directory = CONFIG['directories']['masks_save_directory'],
    stitches_save_directory = CONFIG['directories']['stitches_save_directory'],
    patch_size = CONFIG['patch_size'],
    step_size = CONFIG['stride_size'],
    preset = CREATE_PATCHES_PRESET,
    patch_level = CONFIG['patch_level'],
    slides_format = CONFIG['slides_format'],
    skip_existing = CONFIG['skip_existing'],
    verbose = CONFIG['verbose']
  )
  
  logger.success("Patch Creation Completed Successfully!")

if __name__ == '__main__':
  main()