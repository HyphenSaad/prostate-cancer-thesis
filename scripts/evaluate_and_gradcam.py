import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import time
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import OUTPUT_BASE_DIRECTORY, DATASET_BASE_DIRECTORY, DATASET_SLIDES_FOLDER_NAME
from constants.mil_models import MILModels
from constants.encoders import Encoders
from constants.configs import CREATE_PATCHES_PRESET
from utils.helper import create_directories
from utils.wsi_core.whole_slide_image import WholeSlideImage
from utils.wsi_core.wsi_utils import save_hdf5
from encoders import get_encoder, get_custom_transformer
from mil_models import find_mil_model
from utils.logger import Logger

logger = Logger()

# Hardcoded configuration
CONFIG = {
    'backbone': Encoders.RESNET50.value,
    'mil_model': MILModels.TRANS_MIL.value,
    'n_classes': 6,
    'patch_size': 512,
    'in_dim': 1024,
    'drop_out': 0.25,
    'slide_id': '00bbc1482301d16de3ff63238cfd0b34',  # Hardcoded slide ID as requested
    'checkpoint_path': None,
    'fold': 0,
    'slides_format': 'tiff',
    'attention_heatmap': True,
    'gradcam': True,
    'alpha': 0.5,  # Transparency of heatmap overlay
    'directories': {
        'slides_directory': os.path.join(DATASET_BASE_DIRECTORY, DATASET_SLIDES_FOLDER_NAME),
        'output_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'evaluation_visualizations'),
        'temp_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'temp'),
        'train_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'train'),
    },
    'verbose': True
}

def show_configs():
    logger.empty_line()
    logger.info("Using Configurations:")
    logger.text(f"> Backbone: {CONFIG['backbone']}")
    logger.text(f"> MIL Model: {CONFIG['mil_model']}")
    logger.text(f"> Number of Classes: {CONFIG['n_classes']}")
    logger.text(f"> Patch Size: {CONFIG['patch_size']}")
    logger.text(f"> Input Dimension: {CONFIG['in_dim']}")
    logger.text(f"> Slide ID: {CONFIG['slide_id']}")
    if CONFIG['checkpoint_path']:
        logger.text(f"> Checkpoint Path: {CONFIG['checkpoint_path']}")
    else:
        logger.text(f"> Using fold {CONFIG['fold']} checkpoint")
    logger.text(f"> Attention Heatmap: {CONFIG['attention_heatmap']}")
    logger.text(f"> GradCAM: {CONFIG['gradcam']}")
    logger.empty_line()

def create_patches_for_slide(slide_id, slide_path, output_dir):
    """
    Create patches for a specific slide with improved error handling
    """
    logger.info(f"Creating patches for slide {slide_id}...")
    patches_save_path = os.path.join(output_dir, f"{slide_id}.h5")
    
    # Skip if patches already exist
    if os.path.exists(patches_save_path):
        logger.info(f"Patches already exist at {patches_save_path}")
        return patches_save_path
    
    # Initialize WSI object
    try:
        wsi_object = WholeSlideImage(slide_path, verbose=CONFIG['verbose'])
    except Exception as e:
        logger.error(f"Failed to load slide: {e}")
        return None

    # Modified approach for this specific slide - skip tissue segmentation
    # and use the whole slide instead
    try:
        # Try normal segmentation first
        seg_params = {
            'seg_level': int(CREATE_PATCHES_PRESET['seg_level']),
            'sthresh': int(CREATE_PATCHES_PRESET['sthresh']),
            'mthresh': int(CREATE_PATCHES_PRESET['mthresh']),
            'close': int(CREATE_PATCHES_PRESET['close']),
            'use_otsu': bool(CREATE_PATCHES_PRESET['use_otsu']),
            'keep_ids': CREATE_PATCHES_PRESET['keep_ids'],
            'exclude_ids': CREATE_PATCHES_PRESET['exclude_ids'],
        }
        
        filter_params = {
            'a_t': int(CREATE_PATCHES_PRESET['a_t']),
            'a_h': int(CREATE_PATCHES_PRESET['a_h']),
            'max_n_holes': int(CREATE_PATCHES_PRESET['max_n_holes']),
        }
        
        wsi_object.segment_tissue(**seg_params, filter_params=filter_params)
    except Exception as e:
        logger.warning(f"Standard tissue segmentation failed: {e}")
        logger.info("Falling back to simplified segmentation approach")
        
        # Simplified approach: create a single contour covering the whole slide
        w, h = wsi_object.level_dim[0]
        wsi_object.contours_tissue = [np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)]
        wsi_object.holes_tissue = [[]]  # No holes
    
    # Process contours to extract patches
    patch_params = {
        'patch_level': 0,
        'patch_size': CONFIG['patch_size'],
        'step_size': CONFIG['patch_size'],
        'save_path': output_dir,
        'use_padding': bool(CREATE_PATCHES_PRESET['use_padding']),
        'contour_fn': CREATE_PATCHES_PRESET['contour_fn'],
    }
    
    try:
        patch_file = wsi_object.process_contours(**patch_params)
        logger.success(f"Created patches for slide {slide_id}")
        return patches_save_path
    except Exception as e:
        logger.error(f"Failed to process contours: {e}")
        
        # If normal method fails, try direct patch extraction
        logger.info("Attempting direct grid-based patch extraction")
        
        w, h = wsi_object.level_dim[0]
        patch_size = CONFIG['patch_size']
        step_size = CONFIG['patch_size']
        
        # Generate grid coordinates
        coords = []
        for x in range(0, w-patch_size+1, step_size):
            for y in range(0, h-patch_size+1, step_size):
                coords.append([x, y])
        
        if len(coords) == 0:
            logger.error("No valid coordinates generated")
            return None
            
        coords = np.array(coords)
        
        # Save coordinates to H5 file
        asset_dict = {'coords': coords}
        attr_dict = {
            'coords': {
                'patch_size': patch_size,
                'patch_level': 0,
                'downsample': wsi_object.level_downsamples[0],
                'downsampled_level_dim': wsi_object.level_dim[0],
                'level_dim': wsi_object.level_dim[0],
                'name': slide_id
            }
        }
        
        save_hdf5(patches_save_path, asset_dict, attr_dict, mode='w')
        logger.success(f"Created grid-based patches for slide {slide_id}")
        return patches_save_path

def extract_patches_from_h5(slide_id, h5_path, extract_dir):
    """
    Extract actual image patches from H5 file
    """
    logger.info(f"Extracting patches for slide {slide_id}...")
    
    output_dir = os.path.join(extract_dir, slide_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if patches already extracted
    if len(os.listdir(output_dir)) > 0:
        logger.info(f"Patches already extracted at {output_dir}")
        return output_dir
    
    try:
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][()]
        
        slide_path = os.path.join(CONFIG['directories']['slides_directory'], f"{slide_id}.{CONFIG['slides_format']}")
        wsi_object = WholeSlideImage(slide_path, verbose=CONFIG['verbose'])
        
        for idx, (x, y) in enumerate(coords):
            img_name = f'{x}_{y}_{CONFIG["patch_size"]}_{CONFIG["patch_size"]}.jpg'
            img_path = os.path.join(output_dir, img_name)
            
            if os.path.exists(img_path):
                continue
                
            patch = wsi_object.wsi.read_region((x, y), 0, (CONFIG['patch_size'], CONFIG['patch_size'])).convert('RGB')
            patch.save(img_path)
            
            if CONFIG['verbose'] and idx % 100 == 0:
                logger.info(f"Extracted {idx+1}/{len(coords)} patches")
                
        logger.success(f"Extracted {len(coords)} patches for slide {slide_id}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Failed to extract patches: {e}")
        return None

def extract_features(slide_id, extract_dir, output_dir):
    """
    Extract features from patches using a pretrained encoder
    """
    logger.info(f"Extracting features for slide {slide_id} using {CONFIG['backbone']}...")
    
    # Create output directory for features
    encoder_pt_path = os.path.join(output_dir, CONFIG['backbone'])
    os.makedirs(encoder_pt_path, exist_ok=True)
    feature_path = os.path.join(encoder_pt_path, f"{slide_id}.pt")
    
    # Check if features already extracted
    if os.path.exists(feature_path):
        logger.info(f"Features already extracted at {feature_path}")
        return feature_path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize encoder and transformer
    encoder = get_encoder(CONFIG['backbone'], device, 1 if device.type == "cpu" else torch.cuda.device_count())
    transformer = get_custom_transformer(CONFIG['backbone'])
    
    # Load patch coordinates
    h5_path = os.path.join(CONFIG['directories']['temp_directory'], 'patches', f"{slide_id}.h5")
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][()]
    
    # Process patches
    features_list = []
    patches_dir = os.path.join(extract_dir, slide_id)
    
    with torch.no_grad():
        for idx, (x, y) in enumerate(coords):
            img_name = f'{x}_{y}_{CONFIG["patch_size"]}_{CONFIG["patch_size"]}.jpg'
            img_path = os.path.join(patches_dir, img_name)
            
            if not os.path.exists(img_path):
                logger.warning(f"Patch {img_path} not found, skipping")
                continue
                
            img = Image.open(img_path).convert('RGB')
            img_tensor = transformer(img).unsqueeze(0).to(device)
            feature = encoder(img_tensor).cpu()
            features_list.append(feature)
            
            if CONFIG['verbose'] and idx % 100 == 0:
                logger.info(f"Processed {idx+1}/{len(coords)} patches")
    
    # Concatenate and save features
    if len(features_list) > 0:
        features = torch.cat(features_list, dim=0)
        torch.save(features, feature_path)
        logger.success(f"Extracted features for {len(features_list)} patches, saved to {feature_path}")
        return feature_path
    else:
        logger.error("No features extracted")
        return None

def load_mil_model():
    """
    Load the MIL model from checkpoint
    """
    if CONFIG['checkpoint_path'] is None:
        CONFIG['checkpoint_path'] = os.path.join(
            CONFIG['directories']['train_directory'],
            f"s_{CONFIG['fold']}_checkpoint.pt"
        )
        
    logger.info(f"Loading {CONFIG['mil_model']} model from {CONFIG['checkpoint_path']}...")
    
    drop_out = CONFIG['drop_out']
    model = find_mil_model(
        CONFIG['mil_model'],
        CONFIG['in_dim'],
        CONFIG['n_classes'],
        drop_out=drop_out
    )
    
    if hasattr(model, 'relocate'):
        model.relocate()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
    try:
        checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
        model.eval()
        logger.success(f"Successfully loaded model from checkpoint")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def get_attention_map(model, features, coords):
    """
    Generate attention map using the MIL model's attention mechanism
    """
    logger.info("Generating attention map...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    
    # For TransMIL, we need to extract attention weights
    attention_scores = None
    
    if CONFIG['mil_model'] == MILModels.TRANS_MIL.value:
        # Forward pass with hook to capture attention
        outputs = {}
        attention_hook_handle = None
        
        def hook_fn(module, input, output):
            # For TransMIL, this will capture attention weights
            if isinstance(output, tuple) and len(output) > 1:
                outputs['attention'] = output[1]
            else:
                # If output structure is different, try to get attention from module
                if hasattr(module, 'attention_weights'):
                    outputs['attention'] = module.attention_weights
        
        # Register hook (this is model-specific and might need adjustment)
        if hasattr(model, 'layer2') and hasattr(model.layer2, 'attn'):
            attention_hook_handle = model.layer2.attn.register_forward_hook(hook_fn)
        elif hasattr(model, 'attn'):
            attention_hook_handle = model.attn.register_forward_hook(hook_fn)
            
        # Run inference
        with torch.no_grad():
            _ = model(features.unsqueeze(0))
            
        if attention_hook_handle:
            attention_hook_handle.remove()
            
        if 'attention' in outputs:
            # Process attention scores
            attention = outputs['attention']
            # Average across heads if multi-head attention
            if len(attention.shape) > 3:
                attention = attention.mean(dim=1)
            # Take the scores related to the CLS token attending to patches
            attention_scores = attention[0, 0, 1:].cpu().numpy()
            
            # Make sure we have the right number of attention scores
            if len(attention_scores) < len(features):
                # Pad with zeros if necessary
                attention_scores = np.pad(attention_scores, 
                                          (0, len(features) - len(attention_scores)), 
                                          'constant')
            elif len(attention_scores) > len(features):
                # Truncate if necessary
                attention_scores = attention_scores[:len(features)]
        else:
            # Fallback method if hook doesn't capture attention
            logger.warning("Couldn't extract attention through hook, using uniform attention")
            attention_scores = np.ones(len(features))
    else:
        # Fallback to uniform attention if model type not supported
        logger.warning(f"Attention extraction not implemented for {CONFIG['mil_model']}, using uniform attention")
        attention_scores = np.ones(len(features))
    
    # Normalize attention scores
    if attention_scores.max() > attention_scores.min():
        attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
    else:
        # If all values are the same, use uniform attention
        attention_scores = np.ones(len(features))
    
    return attention_scores

def predict_slide(model, features):
    """
    Generate prediction for a slide
    """
    logger.info("Generating prediction...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    
    with torch.no_grad():
        outputs = model(features.unsqueeze(0))
        logits = outputs['wsi_logits']
        probs = outputs['wsi_prob']
        pred_class = outputs['wsi_label'].item()
        
    class_names = [f"Grade {i}" for i in range(CONFIG['n_classes'])]
    probabilities = probs.cpu().numpy()[0]
    
    result = {
        'predicted_class': pred_class,
        'predicted_class_name': class_names[pred_class],
        'probabilities': {class_names[i]: float(probabilities[i]) for i in range(CONFIG['n_classes'])}
    }
    
    logger.success(f"Prediction: {result['predicted_class_name']} with probability {result['probabilities'][result['predicted_class_name']]:.4f}")
    return result

def create_visualization(slide_id, attention_scores, coords, result):
    """
    Create and save heatmap visualization overlaid on the original slide
    """
    logger.info("Creating visualization...")
    
    slide_path = os.path.join(CONFIG['directories']['slides_directory'], f"{slide_id}.{CONFIG['slides_format']}")
    wsi_object = WholeSlideImage(slide_path, verbose=CONFIG['verbose'])
    
    # Determine visualization level
    vis_level = wsi_object.wsi.get_best_level_for_downsample(32)
    
    try:
        # Create heatmap visualization with error handling
        heatmap = wsi_object.vis_heatmap(
            scores=attention_scores, 
            coords=coords,
            vis_level=vis_level,
            patch_size=(CONFIG['patch_size'], CONFIG['patch_size']),
            blank_canvas=False,
            cmap='jet',
            alpha=CONFIG['alpha'],
            use_holes=True,
            binarize=False,
            segment=False  # Disable segmentation to avoid errors
        )
    except Exception as e:
        logger.warning(f"Failed to create heatmap with original settings: {e}")
        # Fallback to simpler visualization
        try:
            heatmap = wsi_object.vis_heatmap(
                scores=attention_scores, 
                coords=coords,
                vis_level=vis_level,
                patch_size=(CONFIG['patch_size'], CONFIG['patch_size']),
                blank_canvas=True,  # Use blank canvas
                cmap='jet',
                alpha=CONFIG['alpha'],
                use_holes=False,
                binarize=False,
                segment=False
            )
        except Exception as e:
            logger.error(f"Failed to create heatmap: {e}")
            # Return fallback paths even though visualization failed
            output_path = os.path.join(CONFIG['directories']['output_directory'], f"{slide_id}_prediction_{result['predicted_class']}.txt")
            with open(output_path, 'w') as f:
                f.write(f"Prediction: {result['predicted_class_name']} with probability {result['probabilities'][result['predicted_class_name']]:.4f}")
            return output_path, output_path
    
    # Save visualization
    output_filename = f"{slide_id}_prediction_{result['predicted_class']}.png"
    output_path = os.path.join(CONFIG['directories']['output_directory'], output_filename)
    heatmap.save(output_path)
    
    # Add labels and legend to the visualization
    img = np.array(heatmap)
    h, w, _ = img.shape
    
    # Create a labeled version with predictions
    labeled_img = Image.fromarray(img)
    labeled_path = os.path.join(CONFIG['directories']['output_directory'], f"{slide_id}_labeled.png")
    
    # Add prediction text directly to the image using PIL
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(labeled_img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    # Add prediction text
    text = f"Prediction: {result['predicted_class_name']} ({result['probabilities'][result['predicted_class_name']]:.2f})"
    text_color = (255, 255, 255)
    text_position = (20, 20)
    
    # Add text shadow for better visibility
    draw.text((text_position[0]+2, text_position[1]+2), text, fill=(0, 0, 0), font=font)
    draw.text(text_position, text, fill=text_color, font=font)
    
    # Add probability bars
    bar_start_y = 60
    bar_height = 20
    bar_width = 200
    for i, (class_name, prob) in enumerate(sorted(result['probabilities'].items(), key=lambda x: x[0])):
        # Draw label
        draw.text((20, bar_start_y + i*30), f"{class_name}:", fill=text_color, font=font)
        # Draw background bar
        draw.rectangle([(120, bar_start_y + i*30), (120 + bar_width, bar_start_y + i*30 + bar_height)], fill=(50, 50, 50))
        # Draw filled bar based on probability
        draw.rectangle([(120, bar_start_y + i*30), (120 + int(bar_width * prob), bar_start_y + i*30 + bar_height)], 
                     fill=(int(255*(1-prob)), int(255*prob), 0))
        # Draw probability text
        draw.text((330, bar_start_y + i*30), f"{prob:.2f}", fill=text_color, font=font)
    
    labeled_img.save(labeled_path)
    
    logger.success(f"Saved visualization to {output_path}")
    logger.success(f"Saved labeled visualization to {labeled_path}")
    return output_path, labeled_path

def main():
    logger.info(f"Processing slide {CONFIG['slide_id']}...")
    show_configs()
    
    # Create necessary directories
    create_directories({
        'temp_patches': os.path.join(CONFIG['directories']['temp_directory'], 'patches'),
        'temp_extracted': os.path.join(CONFIG['directories']['temp_directory'], 'extracted_patches'),
        'temp_features': os.path.join(CONFIG['directories']['temp_directory'], 'features'),
        'output': CONFIG['directories']['output_directory']
    })
    
    # Step 1: Find the slide file
    slide_path = os.path.join(CONFIG['directories']['slides_directory'], f"{CONFIG['slide_id']}.{CONFIG['slides_format']}")
    if not os.path.exists(slide_path):
        logger.error(f"Slide not found at {slide_path}")
        return
    
    # Step 2: Create patches
    patches_path = create_patches_for_slide(
        CONFIG['slide_id'], 
        slide_path, 
        os.path.join(CONFIG['directories']['temp_directory'], 'patches')
    )
    if patches_path is None:
        logger.error("Failed to create patches")
        return
    
    # Step 3: Extract patches
    extracted_dir = extract_patches_from_h5(
        CONFIG['slide_id'], 
        patches_path, 
        os.path.join(CONFIG['directories']['temp_directory'], 'extracted_patches')
    )
    if extracted_dir is None:
        logger.error("Failed to extract patches")
        return
    
    # Step 4: Extract features
    features_path = extract_features(
        CONFIG['slide_id'], 
        os.path.join(CONFIG['directories']['temp_directory'], 'extracted_patches'), 
        os.path.join(CONFIG['directories']['temp_directory'], 'features')
    )
    if features_path is None:
        logger.error("Failed to extract features")
        return
    
    # Step 5: Load model
    model = load_mil_model()
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Step 6: Load features and coordinates
    features = torch.load(features_path)
    
    with h5py.File(patches_path, 'r') as f:
        coords = f['coords'][()]
    
    # Step 7: Run inference and get prediction
    result = predict_slide(model, features)
    
    # Step 8: Generate attention map
    attention_scores = get_attention_map(model, features, coords)
    
    # Step 9: Create and save visualization
    viz_path, labeled_path = create_visualization(CONFIG['slide_id'], attention_scores, coords, result)
    
    logger.empty_line()
    logger.success(f"Evaluation complete for slide {CONFIG['slide_id']}")
    logger.success(f"Predicted class: {result['predicted_class_name']} with probability {result['probabilities'][result['predicted_class_name']]:.4f}")
    logger.success(f"Visualization saved to {viz_path}")
    logger.success(f"Labeled visualization saved to {labeled_path}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())