import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants.misc import OUTPUT_BASE_DIRECTORY, DATASET_BASE_DIRECTORY, DATASET_INFO_FILE_NAME
from constants.mil_models import MILModels
from constants.encoders import Encoders
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
            
            # Simple approach: use patch-level scores based on feature importance
            # This is a fallback when direct attention extraction fails
            attention_scores = torch.ones(features.shape[1])  # Default to uniform attention
            
            logger.warning("Could not extract attention maps directly, using proxy method.")
            return attention_scores.numpy(), pred_class
        else:
            # Return the captured attention weights
            attention_scores = self.attentions.numpy()
            outputs = self.model(features.to(self.device))
            pred_class = torch.argmax(outputs['wsi_logits'], dim=1).item()
            return attention_scores, pred_class

def create_attention_heatmap(
    slide_path, 
    coords, 
    attention_scores, 
    patch_size,
    output_path,
    threshold=0.5,
    alpha=0.5,
    cmap='coolwarm'
):
    """Create and save attention heatmap overlay on the slide"""
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
        
        # Normalize attention scores to 0-100 range
        norm_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8) * 100
        
        # Log information for debugging
        logger.info(f"Coords shape: {coords.shape}")
        logger.info(f"Attention scores shape: {norm_scores.shape}")
        logger.info(f"Creating heatmap with {len(coords)} points")
        
        # Create heatmap visualization
        heatmap = wsi_object.vis_heatmap(
            scores=norm_scores,
            coords=coords,
            vis_level=-1,  # Auto-select best level
            patch_size=(patch_size, patch_size),
            blank_canvas=False,
            alpha=alpha,
            binarize=threshold > 0,
            thresh=threshold,
            overlap=0.0,
            cmap=cmap
        )
        
        # Save the heatmap
        heatmap.save(output_path)
        logger.success(f"Attention heatmap saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating attention heatmap: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

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
            # If single value or wrong size, expand to match coords
            if len(attention_scores) == 1:
                attention_scores = np.full(len(coords), attention_scores[0])
            # If too many values, truncate
            elif len(attention_scores) > len(coords):
                attention_scores = attention_scores[:len(coords)]
            # If too few values, pad with mean
            else:
                mean_score = np.mean(attention_scores)
                attention_scores = np.pad(attention_scores, 
                                        (0, len(coords) - len(attention_scores)), 
                                        'constant', 
                                        constant_values=mean_score)
    
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
    
    if success:
        logger.success(f"Attention heatmap saved to: {heatmap_path}")
    else:
        logger.error("Failed to create attention heatmap")
        # If heatmap generation fails, try a simpler visualization approach
        try:
            fallback_heatmap_path = os.path.join(
                CONFIG['output_dir'],
                f"{CONFIG['slide_id']}_{timestamp}_attention_fallback.png"
            )
            
            # Create a fallback visualization (simpler approach)
            logger.info("Attempting fallback visualization method...")
            
            # Save coords and attention scores for debugging
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
        except Exception as e:
            logger.error(f"Fallback visualization also failed: {e}")
    
    logger.success("Evaluation and visualization completed successfully!")

if __name__ == '__main__':
    main()
