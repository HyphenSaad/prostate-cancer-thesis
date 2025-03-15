import pandas as pd
import numpy as np

def initialize_dataframe(
  slides: list,
  seg_params: dict,
  filter_params: dict,
  vis_params: dict,
  patch_params: dict,
	use_heatmap_args: bool = False,
  save_patches: bool = False
):
	total_slides = len(slides)
	default_df_dict = {
    'slide_id': slides,
    'process': np.full((total_slides), 1, dtype = np.uint8)
  }

	# initiate empty labels in case not provided
	if use_heatmap_args:
		default_df_dict.update({'label': np.full((total_slides), -1)})
	
	default_df_dict.update({
		'status': np.full((total_slides), 'tbp'),

    # segmentation params
		'seg_level': np.full((total_slides), int(seg_params['seg_level']), dtype = np.int8),
		'sthresh': np.full((total_slides), int(seg_params['sthresh']), dtype = np.uint8),
		'mthresh': np.full((total_slides), int(seg_params['mthresh']), dtype = np.uint8),
		'close': np.full((total_slides), int(seg_params['close']), dtype = np.uint32),
		'use_otsu': np.full((total_slides), bool(seg_params['use_otsu']), dtype = bool),
		'keep_ids': np.full((total_slides), seg_params['keep_ids']),
		'exclude_ids': np.full((total_slides), seg_params['exclude_ids']),
		
		# filter params
		'a_t': np.full((total_slides), int(filter_params['a_t']), dtype = np.float32),
		'a_h': np.full((total_slides), int(filter_params['a_h']), dtype = np.float32),
		'max_n_holes': np.full((total_slides), int(filter_params['max_n_holes']), dtype = np.uint32),

		# vis params
		'vis_level': np.full((total_slides), int(vis_params['vis_level']), dtype = np.int8),
		'line_thickness': np.full((total_slides), int(vis_params['line_thickness']), dtype = np.uint32),

		# patching params
		'use_padding': np.full((total_slides), bool(patch_params['use_padding']), dtype = bool),
		'contour_fn': np.full((total_slides), patch_params['contour_fn'])
  })

	if save_patches:
		default_df_dict.update({
			'white_thresh': np.full((total_slides), int(patch_params['white_thresh']), dtype = np.uint8),
			'black_thresh': np.full((total_slides), int(patch_params['black_thresh']), dtype = np.uint8)
    })

	if use_heatmap_args:
		# initiate empty x,y coordinates in case not provided
		default_df_dict.update({
      'x1': np.empty((total_slides)).fill(np.NaN), 
			'x2': np.empty((total_slides)).fill(np.NaN), 
			'y1': np.empty((total_slides)).fill(np.NaN), 
			'y2': np.empty((total_slides)).fill(np.NaN)
    })
	
	return pd.DataFrame(default_df_dict)