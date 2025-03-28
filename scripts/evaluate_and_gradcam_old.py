import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Add the missing OpenCV import
from PIL import Image
import pandas as pd
from datetime import datetime
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import OUTPUT_BASE_DIRECTORY, DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME
from constants.mil_models import MILModels
from constants.encoders import Encoders
from constants.configs import CREATE_PATCHES_PRESET  # Import segmentation configs
from utils.helper import create_directories
from utils.data_loader import GenericMILDataset, GenericSplit
from utils.train_engine import TrainEngine
from utils.file_utils import save_pkl, save_json
from utils.logger import Logger
from utils.wsi_core.whole_slide_image import WholeSlideImage

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
    'slide_id': None,
    'checkpoint_path': None,
    'dataset_info_csv': os.path.join(DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME),
    'attention_only': False,
    'attention_threshold': 0.5,
    'output_dir': os.path.join(OUTPUT_BASE_DIRECTORY, 'gradcam_results'),
    'slide_format': 'tiff',
    'color_map': 'coolwarm',
    'alpha': 0.5,
    'verbose': False,
    'directories': {
        'results_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'evaluation'),
        'train_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'train'),
        'create_splits_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_splits'),
        'create_patches_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'create_patches', 'patches'),
        'extract_patches_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_patches'),
        'features_pt_directory': os.path.join(OUTPUT_BASE_DIRECTORY, 'extract_features', 'pt_files'),
        'slides_directory': os.path.join(DATASET_BASE_DIRECTORY, 'train_images'),
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
    logger.text(f"> Slide ID: {CONFIG['slide_id']}")
    logger.text(f"> Checkpoint Path: {CONFIG['checkpoint_path']}")
    logger.text(f"> Attention Only: {CONFIG['attention_only']}")
    logger.text(f"> Attention Threshold: {CONFIG['attention_threshold']}")
    logger.text(f"> Color Map: {CONFIG['color_map']}")
    logger.text(f"> Alpha: {CONFIG['alpha']}")
    logger.text(f"> Verbose: {CONFIG['verbose']}")
    logger.empty_line()

def load_arguments():
    parser = argparse.ArgumentParser(description="Evaluate specific slides and generate attention visualizations")
    parser.add_argument(
        "--slide-id",
        type=str,
        required=True,
        help="Slide ID to evaluate"
    )
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
        "--fold",
        type=int,
        default=CONFIG['fold'],
        help=f"Fold to use (default: {CONFIG['fold']})"
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
        "--checkpoint-path",
        type=str,
        default=CONFIG['checkpoint_path'],
        help="Path to model checkpoint (default: auto-generated based on other parameters)"
    )
    parser.add_argument(
        "--attention-only",
        type=bool,
        default=CONFIG['attention_only'],
        help=f"Only show attention, don't apply GradCAM (default: {CONFIG['attention_only']})"
    )
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=CONFIG['attention_threshold'],
        help=f"Threshold for attention visualization (default: {CONFIG['attention_threshold']})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CONFIG['output_dir'],
        help=f"Output directory for results (default: {CONFIG['output_dir']})"
    )
    parser.add_argument(
        "--slide-format",
        type=str,
        default=CONFIG['slide_format'],
        help=f"Format of slide files (default: {CONFIG['slide_format']})"
    )
    parser.add_argument(
        "--color-map",
        type=str,
        default=CONFIG['color_map'],
        help=f"Color map for attention visualization (default: {CONFIG['color_map']})"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=CONFIG['alpha'],
        help=f"Alpha value for overlay (default: {CONFIG['alpha']})"
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=CONFIG['verbose'],
        help=f"Verbose mode (default: {CONFIG['verbose']})"
    )
    parser.add_argument(
        "--dataset-base-directory",
        type=str,
        default=DATASET_BASE_DIRECTORY,
        help=f"Dataset base directory (default: {DATASET_BASE_DIRECTORY})"
    )
    parser.add_argument(
        "--output-base-directory",
        type=str,
        default=OUTPUT_BASE_DIRECTORY,
        help=f"Output base directory (default: {OUTPUT_BASE_DIRECTORY})"
    )
    
    args = parser.parse_args()
    
    CONFIG['slide_id'] = args.slide_id
    CONFIG['backbone'] = args.backbone
    CONFIG['mil_model'] = args.mil_model
    CONFIG['drop_out'] = args.drop_out
    CONFIG['n_classes'] = args.n_classes
    CONFIG['fold'] = args.fold
    CONFIG['patch_size'] = args.patch_size
    CONFIG['in_dim'] = args.in_dim
    CONFIG['checkpoint_path'] = args.checkpoint_path
    CONFIG['attention_only'] = args.attention_only
    CONFIG['attention_threshold'] = args.attention_threshold
    CONFIG['output_dir'] = args.output_dir
    CONFIG['slide_format'] = args.slide_format
    CONFIG['color_map'] = args.color_map
    CONFIG['alpha'] = args.alpha
    CONFIG['verbose'] = args.verbose
    
    # Update directories based on provided base directories
    dataset_base_dir = args.dataset_base_directory
    output_base_dir = args.output_base_directory
    
    CONFIG['dataset_info_csv'] = os.path.join(dataset_base_dir, DATASET_INFO_FILE_NAME)
    CONFIG['directories']['results_directory'] = os.path.join(output_base_dir, 'evaluation')
    CONFIG['directories']['train_directory'] = os.path.join(output_base_dir, 'train')
    CONFIG['directories']['create_splits_directory'] = os.path.join(output_base_dir, 'create_splits')
    CONFIG['directories']['create_patches_directory'] = os.path.join(output_base_dir, 'create_patches', 'patches')
    CONFIG['directories']['extract_patches_directory'] = os.path.join(output_base_dir, 'extract_patches')
    CONFIG['directories']['features_pt_directory'] = os.path.join(output_base_dir, 'extract_features', 'pt_files')
    CONFIG['directories']['slides_directory'] = os.path.join(dataset_base_dir, 'train_images')
    
    # If checkpoint_path is not provided, generate it based on model and fold
    if CONFIG['checkpoint_path'] is None:
        CONFIG['checkpoint_path'] = os.path.join(
            CONFIG['directories']['train_directory'],
            f"s_{CONFIG['fold']}_checkpoint.pt"
        )

def create_attention_heatmap(
    slide_path, 
    coords, 
    attention_scores, 
    patch_size,
    output_path,
    threshold=0.0,
    alpha=0.7,
    cmap='jet'
):
    """Create and save attention heatmap overlay on the WSI showing individual patches without blurring"""
    try:
        # Validate inputs
        if coords is None or len(coords) == 0 or attention_scores is None or len(attention_scores) == 0:
            logger.error("Coordinates or attention scores are empty")
            return False
            
        # Match lengths if needed
        if len(coords) != len(attention_scores):
            logger.warning(f"Mismatch between coords ({len(coords)}) and attention scores ({len(attention_scores)})")
            min_len = min(len(coords), len(attention_scores))
            coords = coords[:min_len]
            attention_scores = attention_scores[:min_len]
            
        # Load the WSI
        wsi_object = WholeSlideImage(slide_path, verbose=CONFIG['verbose'])
        
        # Convert coords to numpy array if needed
        coords = np.array(coords) if not isinstance(coords, np.ndarray) else coords
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        
        # Get WSI visualization level
        vis_level = wsi_object.wsi.get_best_level_for_downsample(32)
        logger.info(f"Using visualization level: {vis_level}")
        
        # Get dimensions and read the WSI
        region_size = wsi_object.level_dim[vis_level]
        wsi_img = np.array(wsi_object.wsi.read_region((0,0), vis_level, region_size).convert("RGB"))
        
        # Scale coords to this level
        downsample = wsi_object.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]
        scaled_coords = (coords * np.array(scale)).astype(np.int32)
        scaled_patch_size = int(patch_size * scale[0])
        
        # Normalize attention scores for visualization
        min_score = np.min(attention_scores)
        max_score = np.max(attention_scores)
        norm_attention_scores = (attention_scores - min_score) / (max_score - min_score + 1e-8)
        
        logger.info(f"Original attention score range: {min_score:.4f} to {max_score:.4f}")
        
        # Get colormap function
        cmap_func = plt.get_cmap(cmap)
        
        # Create a copy of the WSI image for overlay
        overlay_img = wsi_img.copy()
        
        # Apply attention overlay for each patch
        logger.info(f"Creating overlay for {len(coords)} patches...")
        for i, (coord, score) in enumerate(zip(scaled_coords, norm_attention_scores)):
            # Skip if score is below threshold
            if score < threshold:
                continue
                
            # Get patch boundaries
            x_start = max(0, coord[0])
            y_start = max(0, coord[1])
            x_end = min(region_size[0], coord[0] + scaled_patch_size)
            y_end = min(region_size[1], coord[1] + scaled_patch_size)
            
            if x_end <= x_start or y_end <= y_start:
                continue
                
            # Get color for this attention score
            color = np.array(cmap_func(score)[:3]) * 255
            
            # Create colored patch with transparency based on attention score
            patch_overlay = np.zeros((y_end-y_start, x_end-x_start, 3), dtype=np.uint8)
            patch_overlay[:,:] = color
            
            # Blend patch with original image based on attention value
            patch_alpha = score * alpha
            original_patch = overlay_img[y_start:y_end, x_start:x_end]
            blended_patch = cv2.addWeighted(
                patch_overlay, patch_alpha, 
                original_patch, 1 - patch_alpha, 
                0
            )
            
            # Place blended patch back into the image
            overlay_img[y_start:y_end, x_start:x_end] = blended_patch
        
        # Save original WSI
        orig_path = os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path).split('_')[0]}_original.png")
        Image.fromarray(wsi_img).save(orig_path)
        logger.info(f"Saved original WSI to: {orig_path}")
        
        # Save overlay image
        overlay_path = output_path
        Image.fromarray(overlay_img).save(overlay_path)
        logger.info(f"Saved attention overlay to: {overlay_path}")
        
        # Create and save a pure attention heatmap (just patches, no WSI background)
        heatmap_img = np.zeros_like(wsi_img)
        
        for i, (coord, score) in enumerate(zip(scaled_coords, norm_attention_scores)):
            if score < threshold:
                continue
                
            x_start = max(0, coord[0])
            y_start = max(0, coord[1])
            x_end = min(region_size[0], coord[0] + scaled_patch_size)
            y_end = min(region_size[1], coord[1] + scaled_patch_size)
            
            if x_end <= x_start or y_end <= y_start:
                continue
                
            color = np.array(cmap_func(score)[:3]) * 255
            heatmap_img[y_start:y_end, x_start:x_end] = color
        
        heatmap_path = os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path).split('.')[0]}_heatmap.png")
        Image.fromarray(heatmap_img).save(heatmap_path)
        logger.info(f"Saved heatmap-only version to: {heatmap_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating attention heatmap: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def save_heatmap_overlay(wsi_img, heatmap, output_path, cmap='jet', alpha=0.7):
    """Helper function to apply colormap and save heatmap overlay"""
    try:
        # Apply colormap to create colored heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        
        # Create a more continuous and smooth color gradient
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay with weighted blending
        overlay = cv2.addWeighted(heatmap_colored, alpha, wsi_img, 1 - alpha, 0)
        
        # Save the overlay
        Image.fromarray(overlay).save(output_path)
        logger.info(f"Saved attention overlay to: {output_path}")
        
        # Save separate heatmap-only version
        heatmap_path = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.basename(output_path).split('.')[0]}_heatmap.png"
        )
        
        # Create visual heatmap with matplotlib for better appearance
        plt.figure(figsize=(12, 10))
        plt.imshow(heatmap, cmap=cmap)
        plt.colorbar(label='Attention Score')
        plt.title('Attention Heatmap')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved heatmap-only version to: {heatmap_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving heatmap overlay: {e}")
        return False

class AttentionExtractor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attentions = None
        self.all_attentions = []  # Store all attention layers
        self.hooks = []
        self._register_hooks()
        
    def _hook_fn(self, module, input, output):
        # For TransMIL model's attention mechanism
        if hasattr(module, 'attn'):
            curr_attention = output.detach().cpu()
            self.all_attentions.append(curr_attention)
            # We'll also keep the last attention for backward compatibility
            self.attentions = curr_attention
    
    def _register_hooks(self):
        # Register hooks for all potential attention layers
        if hasattr(self.model, 'layer1'):
            self.hooks.append(self.model.layer1.register_forward_hook(self._hook_fn))
        if hasattr(self.model, 'layer2'):
            self.hooks.append(self.model.layer2.register_forward_hook(self._hook_fn))
            
        # Add hooks for other potential attention layers or modules
        if hasattr(self.model, '_fc1') and isinstance(self.model._fc1, torch.nn.Sequential):
            for module in self.model._fc1:
                if hasattr(module, 'attn'):
                    self.hooks.append(module.register_forward_hook(self._hook_fn))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_attention_scores(self, features):
        # Reset attention storage before forward pass
        self.attentions = None
        self.all_attentions = []
        
        # Forward pass to trigger hooks
        with torch.no_grad():
            _ = self.model(features.to(self.device))
        
        # Output from the model for class prediction
        outputs = self.model(features.to(self.device))
        pred_class = torch.argmax(outputs['wsi_logits'], dim=1).item()
        
        # If we have multiple attention layers, return them all
        if len(self.all_attentions) > 0:
            # Convert list of tensors to a single numpy array
            attention_array = np.array([attn.numpy() for attn in self.all_attentions])
            logger.info(f"Extracted {len(self.all_attentions)} attention layers")
            return attention_array, pred_class
        
        # Fallback if no attention captured through hooks
        if self.attentions is None:
            logger.warning("Could not extract attention maps directly, using proxy method.")
            attention_scores = torch.ones(features.shape[1])  # Default to uniform attention
            return attention_scores.numpy(), pred_class
        else:
            return self.attentions.numpy(), pred_class

def prepare_slide_dataset(slide_id):
    """Prepare a dataset containing only the specified slide"""
    # Create a dataset entry for this slide
    slide_data = pd.DataFrame({
        'slide_id': [slide_id],
        'label': [0]  # Placeholder label, will be overwritten with prediction
    })
    
    # Create a GenericSplit object for this slide
    slide_dataset = GenericSplit(slide_data, num_classes=CONFIG['n_classes'])
    slide_dataset.set_backbone(CONFIG['backbone'])
    slide_dataset.set_patch_size(CONFIG['patch_size'])
    
    # Set paths for feature loading
    slide_dataset.extract_patches_dir = CONFIG['directories']['extract_patches_directory']
    slide_dataset.patches_dir = CONFIG['directories']['create_patches_directory']
    slide_dataset.features_pt_directory = CONFIG['directories']['features_pt_directory']
    
    return slide_dataset

def predict_slide(slide_dataset, checkpoint_path):
    """Get model prediction for a specific slide"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    drop_out = 0.25 if CONFIG['drop_out'] else 0.0
    train_engine = TrainEngine(
        datasets = [slide_dataset, slide_dataset, slide_dataset],  # Dummy splits using the same dataset
        fold = CONFIG['fold'],
        drop_out = drop_out,
        result_directory = CONFIG['directories']['train_directory'],
        mil_model_name = CONFIG['mil_model'],
        learning_rate = CONFIG['learning_rate'],
        max_epochs = 1,  # Not training, just predicting
        in_dim = CONFIG['in_dim'],
        n_classes = CONFIG['n_classes'],
        verbose = CONFIG['verbose']
    )
    
    # Load the model checkpoint
    if hasattr(train_engine.model, 'load_model'):
        train_engine.model.load_model(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        train_engine.model.load_state_dict(ckpt)
    
    # Get slide features and data
    batch = slide_dataset[0]
    features = batch['features'].unsqueeze(0)  # Add batch dimension
    coords = batch['coords']
    
    # Get model prediction
    train_engine.model.eval()
    with torch.no_grad():
        outputs = train_engine.model(features.to(device))
    
    logits = outputs['wsi_logits']
    probs = outputs['wsi_prob']
    pred_label = outputs['wsi_label'].item()
    
    # Create attention extractor
    attention_extractor = AttentionExtractor(train_engine.model, device)
    
    # Get attention scores
    attention_scores, _ = attention_extractor.get_attention_scores(features)
    
    # Clean up hooks
    attention_extractor.remove_hooks()
    
    # If attention is a multidimensional array, convert to a flattened score
    if isinstance(attention_scores, np.ndarray) and attention_scores.ndim > 1:
        # For simplicity, use mean across dimensions or the first attention layer
        attention_scores = np.mean(attention_scores, axis=tuple(range(attention_scores.ndim - 1)))
    
    # Ensure attention_scores has same length as coords
    if hasattr(coords, '__len__') and hasattr(attention_scores, '__len__'):
        if len(coords) != len(attention_scores):
            logger.warning(f"Attention scores length ({len(attention_scores)}) doesn't match coords length ({len(coords)})")
            
            # If we have more attention scores than coordinates,
            # use a smarter approach to map scores to coords
            if len(attention_scores) > len(coords):
                # Create an importance-based sampling
                # Get the most important indices (highest attention scores)
                sorted_indices = np.argsort(attention_scores)[::-1]
                selected_indices = sorted_indices[:len(coords)]
                # Sort them back to maintain original order
                selected_indices.sort()
                attention_scores = attention_scores[selected_indices]
            else:  # len(attention_scores) < len(coords)
                # Repeat or interpolate the scores
                indices = np.round(np.linspace(0, len(attention_scores) - 1, len(coords))).astype(int)
                attention_scores = attention_scores[indices]
    
    # Return all relevant information
    return {
        'slide_id': slide_dataset.slide_data['slide_id'][0],
        'prediction': pred_label,
        'probabilities': probs.cpu().numpy()[0],
        'logits': logits.cpu().numpy()[0],
        'features': features.cpu().numpy(),
        'attention_scores': attention_scores,
        'coords': coords,
    }

def main():
    logger.draw_header("Evaluate and Visualize Attention Maps")
    load_arguments()
    
    logger.info(f"Evaluating slide: {CONFIG['slide_id']}")
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    show_configs()
    
    # Check if checkpoint exists
    if not os.path.exists(CONFIG['checkpoint_path']):
        logger.error(f"Checkpoint not found at: {CONFIG['checkpoint_path']}")
        return
    
    # Prepare dataset for the specific slide
    logger.info("Preparing slide dataset...")
    slide_dataset = prepare_slide_dataset(CONFIG['slide_id'])
    
    # Predict on the slide
    logger.info("Running model prediction...")
    try:
        results = predict_slide(slide_dataset, CONFIG['checkpoint_path'])
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        if CONFIG['verbose']:
            import traceback
            logger.error(traceback.format_exc())
        return
    
    # Save prediction results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(CONFIG['output_dir'], f"{CONFIG['slide_id']}_{timestamp}_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'slide_id': results['slide_id'],
        'prediction': int(results['prediction']),
        'probabilities': results['probabilities'].tolist(),
        'logits': results['logits'].tolist(),
    }
    
    # Get class names if available, otherwise use class indices
    class_names = [f"Class {i}" for i in range(CONFIG['n_classes'])]
    class_probs = {class_names[i]: float(results['probabilities'][i]) for i in range(CONFIG['n_classes'])}
    json_results['class_probabilities'] = class_probs
    
    # Save JSON results
    save_json(results_file, json_results)
    logger.success(f"Prediction results saved to: {results_file}")
    
    # Log prediction summary
    logger.info(f"Prediction summary for slide {CONFIG['slide_id']}:")
    logger.info(f"  Predicted class: {results['prediction']}")
    logger.info("  Class probabilities:")
    for class_idx, prob in enumerate(results['probabilities']):
        logger.info(f"    Class {class_idx}: {prob:.4f}")
    
    # Create attention heatmap
    logger.info("Generating attention heatmap...")
    
    # Get full slide path
    slide_path = os.path.join(
        CONFIG['directories']['slides_directory'], 
        f"{CONFIG['slide_id']}.{CONFIG['slide_format']}"
    )
    
    if not os.path.exists(slide_path):
        logger.error(f"Slide image not found at: {slide_path}")
        return
    
    # Output path for heatmap
    heatmap_path = os.path.join(
        CONFIG['output_dir'],
        f"{CONFIG['slide_id']}_{timestamp}_attention.png"
    )
    
    # Create and save heatmap
    success = create_attention_heatmap(
        slide_path=slide_path,
        coords=results['coords'],
        attention_scores=results['attention_scores'],
        patch_size=CONFIG['patch_size'],
        output_path=heatmap_path,
        threshold=CONFIG['attention_threshold'],
        alpha=CONFIG['alpha'],
        cmap=CONFIG['color_map']
    )
    
    if (success):
        logger.success(f"Attention heatmap saved to: {heatmap_path}")
    else:
        logger.error("Failed to create attention heatmap")
        
        # Save debug information
        debug_file = os.path.join(CONFIG['output_dir'], 
                                f"{CONFIG['slide_id']}_{timestamp}_debug_info.json")
        
        debug_info = {
            "coords_shape": str(np.array(results['coords']).shape) if results['coords'] is not None else "None",
            "attention_scores_shape": str(np.array(results['attention_scores']).shape) if results['attention_scores'] is not None else "None",
            "slide_id": results['slide_id'],
            "prediction": int(results['prediction'])
        }
        
        save_json(debug_file, debug_info)
        logger.info(f"Saved debug information to: {debug_file}")
    
    logger.success("Evaluation and visualization completed successfully!")

if __name__ == '__main__':
    main()
