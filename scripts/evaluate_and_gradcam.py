import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import h5py
from datetime import datetime

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
from utils.file_utils import save_json

logger = Logger()

# Hardcoded configuration
CONFIG = {
    'backbone': Encoders.RESNET50.value,
    'mil_model': MILModels.TRANS_MIL.value,
    'n_classes': 6,
    'patch_size': 512,
    'in_dim': 1024,
    'drop_out': 0.25,
    'slide_id': '037c18f1a1ec42be86eed81b09867939',  # Hardcoded slide ID
    'checkpoint_path': None,
    'fold': 0,
    'slides_format': 'tiff',
    'attention_threshold': 0.5,
    'color_map': 'inferno',  # Using a high-contrast colormap
    'alpha': 0.7,  # Higher alpha for better visibility
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
    logger.text(f"> Attention Threshold: {CONFIG['attention_threshold']}")
    logger.text(f"> Color Map: {CONFIG['color_map']}")
    logger.text(f"> Alpha: {CONFIG['alpha']}")
    logger.empty_line()

class AttentionExtractor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attentions = None
        self.hooks = []
        self._register_hooks()
        
    def _hook_fn(self, module, input, output):
        if hasattr(module, 'attn'):
            # If using NystromAttention in TransMIL
            self.attentions = output.detach().cpu()
        
    def _register_hooks(self):
        # For TransMIL, look for TransLayer modules
        if hasattr(self.model, 'layer1'):
            self.hooks.append(self.model.layer1.register_forward_hook(self._hook_fn))
        if hasattr(self.model, 'layer2'):
            self.hooks.append(self.model.layer2.register_forward_hook(self._hook_fn))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_attention_scores(self, features):
        # Forward pass to trigger hooks
        with torch.no_grad():
            _ = self.model(features.to(self.device))
        
        # If attention wasn't captured by hooks, use a default method
        if self.attentions is None:
            # Use logits or feature importance as a proxy for attention
            outputs = self.model(features.to(self.device))
            logits = outputs['wsi_logits'] 
            # For multi-class, use the predicted class's logit
            pred_class = torch.argmax(logits, dim=1).item()
            
            # Simple approach: use patch-level feature norms as proxy for attention
            feature_norms = torch.norm(features.squeeze(0), dim=1).cpu().numpy()
            
            # Normalize to [0, 1]
            attention_scores = (feature_norms - feature_norms.min()) / (feature_norms.max() - feature_norms.min() + 1e-8)
            
            logger.warning("Could not extract attention maps directly, using feature magnitude as proxy.")
            return attention_scores, pred_class
        else:
            # Return the captured attention weights
            attention_scores = self.attentions.numpy()
            outputs = self.model(features.to(self.device))
            pred_class = torch.argmax(outputs['wsi_logits'], dim=1).item()
            return attention_scores, pred_class

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

def create_attention_heatmap(slide_path, coords, attention_scores, patch_size, output_path):
    """
    Create and save attention heatmap overlay on the slide
    Based on the reference implementation with improvements for visibility
    """
    try:
        # Validate inputs
        if coords is None or len(coords) == 0:
            logger.error("Coordinates are empty or None")
            return False
            
        if attention_scores is None or len(attention_scores) == 0:
            logger.error("Attention scores are empty or None")
            return False
            
        # Ensure coords and scores have compatible lengths
        if len(coords) != len(attention_scores):
            logger.warning(f"Mismatch between coords ({len(coords)}) and attention scores ({len(attention_scores)})")
            # Use the minimum length to avoid errors
            min_len = min(len(coords), len(attention_scores))
            coords = coords[:min_len]
            attention_scores = attention_scores[:min_len]
            
        # Load the WSI
        wsi_object = WholeSlideImage(slide_path, verbose=CONFIG['verbose'])
        
        # Ensure coords is a numpy array with the right shape
        if not isinstance(coords, np.ndarray):
            logger.warning("Converting coords to numpy array")
            coords = np.array(coords)
        
        # Reshape coords if needed (should be a 2D array with shape [n, 2])
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        
        # Normalize attention scores to 0-1 range
        norm_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
        
        # Log information for debugging
        logger.info(f"Coords shape: {coords.shape}")
        logger.info(f"Attention scores shape: {norm_scores.shape}")
        logger.info(f"Creating heatmap with {len(coords)} points")
        
        # Determine the best visualization level for the whole slide
        vis_level = wsi_object.wsi.get_best_level_for_downsample(32)
        logger.info(f"Using visualization level: {vis_level}")
        
        # Create a colormap function to map attention scores to colors
        cmap_func = plt.get_cmap(CONFIG['color_map'])
        
        # Determine the dimensions of the slide at the visualization level
        region_size = wsi_object.level_dim[vis_level]
        
        # Read the WSI image at the visualization level as background
        logger.info("Reading the whole slide image for background...")
        img = np.array(wsi_object.wsi.read_region((0,0), vis_level, region_size).convert("RGB"))
        
        # Scale coordinates to the visualization level
        downsample = wsi_object.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]
        scaled_coords = coords * np.array(scale)
        scaled_coords = scaled_coords.astype(np.int32)
        
        # Scale patch size to the visualization level
        scaled_patch_size = int(patch_size * scale[0])
        
        logger.info(f"Processing individual patches and applying heatmap...")
        
        # Create overlay for each patch
        highlighted_count = 0
        threshold = CONFIG['attention_threshold']
        
        for i, (coord, score) in enumerate(zip(scaled_coords, norm_scores)):
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(scaled_coords)} patches")
                
            # Get the patch coordinates
            x, y = coord
            
            # Skip if out of bounds
            if (x < 0 or y < 0 or 
                x + scaled_patch_size > region_size[0] or 
                y + scaled_patch_size > region_size[1]):
                continue
            
            # Skip patches below threshold
            if score < threshold:
                continue
                
            highlighted_count += 1
            
            # Get color for this attention score
            color = np.array(cmap_func(score)[:3]) * 255
            
            # Create overlay for this patch
            overlay = np.ones((scaled_patch_size, scaled_patch_size, 3), dtype=np.uint8) * color.astype(np.uint8)
            
            # Apply the overlay using alpha blending
            img[y:y+scaled_patch_size, x:x+scaled_patch_size] = cv2.addWeighted(
                img[y:y+scaled_patch_size, x:x+scaled_patch_size],
                1 - CONFIG['alpha'],
                overlay,
                CONFIG['alpha'],
                0
            )
            
            # Draw a white border around the patch for better visibility
            cv2.rectangle(
                img,
                (x, y),
                (x + scaled_patch_size, y + scaled_patch_size),
                (255, 255, 255),  # White border
                2  # Border thickness
            )
        
        logger.info(f"Highlighted {highlighted_count} patches above threshold {threshold}")
        
        # Save the original WSI at this visualization level for comparison
        orig_wsi_path = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.basename(output_path).split('_')[0]}_original.png"
        )
        orig_img = np.array(wsi_object.wsi.read_region((0,0), vis_level, region_size).convert("RGB"))
        Image.fromarray(orig_img).save(orig_wsi_path)
        logger.info(f"Saved original WSI thumbnail to: {orig_wsi_path}")
        
        # Save the heatmap
        logger.info("Saving the final heatmap...")
        heatmap = Image.fromarray(img)
        heatmap.save(output_path)
        logger.success(f"Attention heatmap saved to: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating attention heatmap: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

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
    
    # Step 8: Extract attention scores using the AttentionExtractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attention_extractor = AttentionExtractor(model, device)
    attention_scores, pred_class = attention_extractor.get_attention_scores(features)
    attention_extractor.remove_hooks()
    
    # Process attention scores if needed
    if isinstance(attention_scores, np.ndarray) and attention_scores.ndim > 1:
        # For simplicity, use mean across dimensions
        attention_scores = np.mean(attention_scores, axis=tuple(range(attention_scores.ndim - 1)))
    
    # Ensure attention_scores has same length as coords
    if len(attention_scores) != len(coords):
        logger.warning(f"Attention scores length ({len(attention_scores)}) doesn't match coords length ({len(coords)})")
        if len(attention_scores) > len(coords):
            attention_scores = attention_scores[:len(coords)]
        else:
            # If too few values, pad with mean
            mean_score = np.mean(attention_scores)
            attention_scores = np.pad(attention_scores, 
                                      (0, len(coords) - len(attention_scores)), 
                                      'constant', 
                                      constant_values=mean_score)
    
    # Step 9: Create attention heatmap using the reference implementation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    heatmap_path = os.path.join(
        CONFIG['directories']['output_directory'],
        f"{CONFIG['slide_id']}_{timestamp}_attention.png"
    )
    
    success = create_attention_heatmap(
        slide_path=slide_path,
        coords=coords,
        attention_scores=attention_scores,
        patch_size=CONFIG['patch_size'],
        output_path=heatmap_path
    )
    
    if success:
        logger.success(f"Attention heatmap saved to: {heatmap_path}")
    else:
        logger.error("Failed to create attention heatmap")
    
    logger.empty_line()
    logger.success(f"Evaluation complete for slide {CONFIG['slide_id']}")
    logger.success(f"Predicted class: {result['predicted_class_name']} with probability {result['probabilities'][result['predicted_class_name']]:.4f}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())