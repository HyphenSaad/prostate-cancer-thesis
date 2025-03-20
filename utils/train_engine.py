import torch
import os
import numpy as np
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, confusion_matrix, classification_report, accuracy_score
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from mil_models import find_mil_model
from utils.common_utils import EarlyStopping, AccuracyLogger
from utils.clam_utils import print_network, get_split_loader, calculate_error
from utils.logger import Logger

def save_splits(
  dataset_splits,
  column_keys,
  filename,
  boolean_style = False
):
  splits = [dataset_splits[i].slide_data['slide_id'] for i in range(len(dataset_splits))]
  if not boolean_style:
    df = pd.concat(splits, ignore_index = True, axis = 1)
    df.columns = column_keys
  else:
    df = pd.concat(splits, ignore_index = True, axis = 0)
    one_hot = np.eye(len(dataset_splits)).astype(bool)
    bool_array = np.repeat(one_hot, [len(dset) for dset in dataset_splits], axis = 0)
    df = pd.DataFrame(
      bool_array,
      index = df.values.tolist(), 
      columns = ['train', 'val', 'test']
    )

  df.to_csv(filename)

class TrainEngine:
  def __init__(
    self,
    datasets,
    fold,
    result_directory,
    mil_model_name,
    learning_rate,
    max_epochs,
    in_dim,
    n_classes,
    drop_out,
    weighted_sample = False,
    optimizer_name = 'adam',
    regularization = 1e-5,
    batch_size = 1,
    bag_loss = 'ce',
    verbose = False
  ):
    self.train_split, self.val_split, self.test_split = datasets
    self.fold = fold
    self.result_dir = result_directory
    self.mil_model_name = mil_model_name
    self.optimizer_name = optimizer_name
    self.learning_rate = learning_rate
    self.regularization = regularization
    self.weighted_sample = weighted_sample
    self.batch_size = batch_size
    self.max_epochs = max_epochs
    self.in_dim = in_dim
    self.n_classes = n_classes
    self.drop_out = drop_out    
    self.bag_loss = bag_loss
    self.verbose = verbose
    self.logger = Logger()

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_splits(
      datasets,
      ['train', 'val', 'test'],
      os.path.join(self.result_dir, 'splits_{}.csv'.format(fold))
    )

    self.init_logger()

    self.call_scheduler = self.get_learning_rate_scheduler()
    self.model = self.get_mil_model()
    self.loss_function = self.get_loss_function()
    self.optimizer = self.get_optimizer()
    
    self.early_stopping = EarlyStopping(
      patience = self.model.early_stopping_patience if hasattr(self.model, 'early_stopping_patience') else 20,
      stop_epoch = self.model.early_stopping_stop_epoch if hasattr(self.model, 'early_stopping_stop_epoch') else 50,
      verbose = self.verbose
    )

    self.train_loader, self.val_loader, self.test_loader = self.init_data_loaders()

    # core, if you implement your training framework, call setup to pass training hyperparameters.
    if hasattr(self.model, 'set_up'):
      extra_args = { 'total_iterations': max_epochs * len(self.train_loader) }
      self.model.set_up(
        lr = learning_rate,
        max_epochs = max_epochs,
        weight_decay = regularization,
        **extra_args
      )
      
  def init_logger(self):
    self.logger.info('Training Fold {}!'.format(self.fold), timestamp=True)
    
    writer_dir = os.path.join(self.result_dir, str(self.fold))
    if not os.path.isdir(writer_dir): os.mkdir(writer_dir)
    self.writer = SummaryWriter(writer_dir, flush_secs = 15)

    self.logger.info("Training on {} samples".format(len(self.train_split)))
    self.logger.info("Validating on {} samples".format(len(self.val_split)))
    self.logger.info("Testing on {} samples".format(len(self.test_split)))
    self.logger.info(f"Using device: {self.device}")

  def get_learning_rate_scheduler(self):
    return None

  def get_mil_model(self):
    model = find_mil_model(
      self.mil_model_name,
      self.in_dim,
      self.n_classes,
      self.drop_out
    )

    if hasattr(model, 'relocate'): model.relocate()
    else: model = model.to(self.device)

    if self.verbose:
      print_network(model)
    else:
      self.logger.info(f"Model: {self.mil_model_name} initialized")
    return model

  def get_loss_function(self):
    if hasattr(self.model, 'loss_function'):
      if self.verbose:
        self.logger.info('The loss function defined in the MIL model is adopted...')
      loss_function = self.model.loss_function
    else:
      if self.verbose:
        self.logger.info('Cross Entropy Loss is adopted as the loss function...')
      loss_function = torch.nn.CrossEntropyLoss()
    return loss_function

  def get_optimizer(self):
    if self.optimizer_name.lower() == "adam":
      optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr = self.learning_rate,
        weight_decay = self.regularization
      )
    elif self.optimizer_name.lower() == 'sgd':
      optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr = self.learning_rate,
        momentum = 0.9,
        weight_decay = self.regularization
      )
    else:
      raise NotImplementedError(f'Optimizer {self.optimizer_name} not implemented!')
    
    self.logger.info(f"Using {self.optimizer_name} optimizer with lr={self.learning_rate}")
    return optimizer

  def init_data_loaders(self):
    train_loader = get_split_loader(
      self.train_split,
      training = True,
      weighted = self.weighted_sample,
      batch_size = self.batch_size
    )
    val_loader = get_split_loader(self.val_split)
    test_loader = get_split_loader(self.test_split)
    return train_loader, val_loader, test_loader

  def train_model(self, fold):
    train_loop_func = self.train_loop_subtyping
    validate_func = self.validate_subtyping
    test_func = self.summary_subtyping

    self.logger.info("Starting training process", timestamp=True)
    for epoch in range(self.max_epochs):
      train_loop_func(epoch)
      stop = validate_func(epoch)
      if stop: 
        self.logger.warning("Early stopping triggered", timestamp=True)
        break
    
    checkpoint_path = os.path.join(self.result_dir, "s_{}_checkpoint.pt".format(fold))
    msg = self.model.load_state_dict(torch.load(checkpoint_path))
    
    self.logger.info('Loading best model checkpoint from: {}'.format(checkpoint_path), timestamp=True)
    if self.verbose:
      self.logger.debug(msg)
    
    # test_func on val loader
    self.logger.info("Evaluating on validation set...", timestamp=True)
    _, val_error, val_auc, _, _, val_f1, val_metrics = test_func(self.val_loader)
    
    # test on test loader
    self.logger.info("Evaluating on test set...", timestamp=True)
    results_dict, test_error, test_auc, acc_logger, df, test_f1, test_metrics = test_func(self.test_loader)
    
    self.logger.success("Final Results:", timestamp=True)
    self.logger.success("Test Error: {:.4f}, ROC AUC: {:.4f}, F1 Score: {:.4f}".format(test_error, test_auc, test_f1))
    self.logger.success("Val Error: {:.4f}, ROC AUC: {:.4f}, F1 Score: {:.4f}".format(val_error, val_auc, val_f1))

    for i in range(self.n_classes):
      acc, correct, count = acc_logger.get_summary(i)
      self.logger.info('Class {}: Accuracy {:.4f}, Correct {}/{}'.format(i, acc, correct, count))
      self.writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    self.writer.add_scalar('final/val_error', val_error, 0)
    self.writer.add_scalar('final/val_auc', val_auc, 0)
    self.writer.add_scalar('final/test_error', test_error, 0)
    self.writer.add_scalar('final/test_auc', test_auc, 0)
    self.writer.add_scalar('final/val_precision', val_metrics['precision_macro'], 0)
    self.writer.add_scalar('final/test_precision', test_metrics['precision_macro'], 0)
    self.writer.add_scalar('final/val_recall', val_metrics['recall_macro'], 0)
    self.writer.add_scalar('final/test_recall', test_metrics['recall_macro'], 0)
    self.writer.add_scalar('final/val_kappa', val_metrics['cohens_kappa'], 0)
    self.writer.add_scalar('final/test_kappa', test_metrics['cohens_kappa'], 0)
    self.writer.close()
    
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, test_f1, val_f1, test_metrics, val_metrics

  def train_loop_subtyping(self, epoch):   
    self.model.train()
    acc_logger = AccuracyLogger(n_classes = self.n_classes)
    train_loss = 0.0
    train_error = 0.0

    self.logger.empty_line()
    self.logger.info('Epoch: {}/{}'.format(epoch+1, self.max_epochs), timestamp=True)
    
    # Store debug messages to print after batch completion
    debug_messages = []
    
    progress_bar = tqdm(
        self.train_loader, 
        desc=f"Training Epoch {epoch+1}/{self.max_epochs}",
        disable=False, 
        ncols=100,
        leave=True  # Ensure progress bar is left on screen after completion
    )
    
    for batch_idx, batch in enumerate(progress_bar):
      iteration = epoch * len(self.train_loader) + batch_idx
      kwargs = {}
      data = batch['features']
      label = batch['label']
      kwargs['iteration'] = iteration
      kwargs['image_call'] = batch['image_call']
      
      # Core 1: if your model need specific pre-process of data and label
      if hasattr(self.model, 'process_data'):
        data, label = self.model.process_data(data, label, self.device)
      else:
        data = data.to(self.device)
        label = label.to(self.device)
        
      # Core 2: Model optimization step
      if hasattr(self.model, 'one_step'):
        outputs = self.model.one_step(data, label, **kwargs)
        loss = outputs['loss']
        if 'call_scheduler' in outputs.keys():
          self.call_scheduler = outputs['call_scheduler']
        logits, Y_prob, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
      else:
        # use universal code to update param
        kwargs['label'] = label
        outputs = self.model(data, **kwargs)
        logits, Y_prob, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']

        if hasattr(self.model, 'loss_function'): loss = self.loss_function(logits, label, **outputs)
        else: loss = self.loss_function(logits, label)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        outputs['loss'] = loss
      
      # to support batch size greater than 1
      if isinstance(label, torch.Tensor):
        acc_logger.log(Y_hat, label)
      else:
        for i in range(len(data)):
          acc_logger.log(Y_hat[i], label[i])

      loss_value = loss.item()
      if torch.isnan(loss):
        progress_bar.close()  # Close the progress bar before showing error
        self.logger.error('NaN loss detected!')
        if self.verbose:
          self.logger.debug('logits: {}'.format(logits))
          self.logger.debug('Y_prob: {}'.format(Y_prob))
          self.logger.debug('loss: {}'.format(loss))
        raise RuntimeError('Found Nan number')
      
      # Update progress bar
      error = calculate_error(Y_hat, label)
      progress_bar.set_postfix({
          'loss': f'{loss_value:.4f}',
          'error': f'{error:.4f}'
      })
      
      # Instead of printing debug messages directly, store them for later
      if (batch_idx + 1) % 20 == 0 and self.verbose:
        bag_size = data[0].shape[0] if isinstance(data, list) else data.shape[0]
        log_message = f'Batch {batch_idx+1}/{len(self.train_loader)}'
        for k, v in outputs.items():
          if 'loss' in k:
            log_message += f', {k}: {v.item():.4f}'
        log_message += f', label: {label.item()}, bag_size: {bag_size}'
        
        # Use tqdm.write to print without breaking the progress bar
        tqdm.write(f"[DEBUG] {log_message}")
              
      train_loss += loss_value
      train_error += error
    
    # Close progress bar properly
    progress_bar.close()

    # calculate loss and error for epoch
    train_loss /= len(self.train_loader)
    train_error /= len(self.train_loader)

    self.logger.success('Epoch: {}/{}, Train Loss: {:.4f}, Train Error: {:.4f}'.format(
        epoch+1, self.max_epochs, train_loss, train_error))
    
    if self.writer:
      self.writer.add_scalar('train/loss', train_loss, epoch)
      self.writer.add_scalar('train/error', train_error, epoch)
        
    if self.verbose:
      for i in range(self.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        self.logger.info('Class {}: Accuracy {:.4f}, Correct {}/{}'.format(i, acc, correct, count))
        if self.writer:
          self.writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if self.call_scheduler is not None:
      self.call_scheduler()

  def validate_subtyping(self, epoch):
    self.model.eval()
    acc_logger = AccuracyLogger(n_classes = self.n_classes)
    val_loss = 0.0
    val_error = 0.0

    prob = np.zeros((len(self.val_loader), self.n_classes))
    labels = np.zeros(len(self.val_loader))    
    Y_hats = np.zeros(len(self.val_loader))    
    
    self.logger.info("Validating epoch {}/{}".format(epoch+1, self.max_epochs), timestamp=True)
    
    progress_bar = tqdm(
        self.val_loader, 
        desc=f"Validating Epoch {epoch+1}/{self.max_epochs}", 
        disable=False,
        ncols=100,
        leave=True  # Ensure progress bar is left on screen
    )
    
    with torch.no_grad():
      for batch_idx, batch in enumerate(progress_bar):
        kwargs = {}
        data = batch['features']
        label = batch['label']
        kwargs['image_call'] = batch['image_call']
        
        if hasattr(self.model, 'process_data'):
          data, label = self.model.process_data(data, label, self.device)
        else:
          data = data.to(self.device)
          label = label.to(self.device)

        if hasattr(self.model, 'wsi_predict'):
          outputs = self.model.wsi_predict(data, **kwargs)
        else: # use universal code to update param
          outputs = self.model(data)

        logits, Y_prob, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
        acc_logger.log(Y_hat, label)
        try: loss = self.loss_function(logits, label, **outputs)
        except: loss = self.loss_function(logits, label)
            
        prob[batch_idx] = Y_prob.cpu().numpy()
        labels[batch_idx] = label.item()
        Y_hats[batch_idx] = Y_hat.item()
        
        loss_value = loss.item()
        error = calculate_error(Y_hat, label)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'error': f'{error:.4f}'
        })
        
        val_loss += loss_value
        val_error += error
    
    # Close progress bar properly
    progress_bar.close()

    val_error /= len(self.val_loader)
    val_loss /= len(self.val_loader)

    if self.n_classes == 2: auc = roc_auc_score(labels, prob[:, 1])
    else: auc = roc_auc_score(labels, prob, multi_class='ovr')

    # f1 score
    f1 = f1_score(labels, Y_hats, average='macro')

    if self.writer:
      self.writer.add_scalar('val/loss', val_loss, epoch)
      self.writer.add_scalar('val/auc', auc, epoch)
      self.writer.add_scalar('val/error', val_error, epoch)

    self.logger.success('Validation Results - Epoch: {}/{}, Loss: {:.4f}, Error: {:.4f}, AUC: {:.4f}, F1: {:.4f}'.format(
        epoch+1, self.max_epochs, val_loss, val_error, auc, f1))
    
    if self.verbose:
      for i in range(self.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        self.logger.info('Class {}: Accuracy {:.4f}, Correct {}/{}'.format(i, acc, correct, count))     

    # val_error is better than val_loss
    self.early_stopping(epoch, val_error, self.model, ckpt_name = os.path.join(self.result_dir, "s_{}_checkpoint.pt".format(self.fold)))
    
    if self.early_stopping.early_stop:
      self.logger.warning("Early stopping triggered at epoch {}/{}".format(epoch+1, self.max_epochs))
      return True
    else:
      return False

  def summary_subtyping(self, loader = None):
    if loader is None:
      loader = self.test_loader
    
    acc_logger = AccuracyLogger(n_classes = self.n_classes)
    self.model.eval()
    test_error = 0.0

    all_probs = np.zeros((len(loader), self.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    # create lists to store all predictions and labels
    all_Y_hat = []
    all_label = []
    
    self.logger.info("Running evaluation...", timestamp=True)
    
    progress_bar = tqdm(
        loader, 
        desc="Evaluating", 
        disable=False,
        ncols=100,
        leave=True  # Ensure progress bar is left on screen
    )
    
    with torch.no_grad():
      for batch_idx, batch in enumerate(progress_bar):
        data = batch['features']
        label = batch['label']
        
        if hasattr(self.model, 'process_data'):
          data, label = self.model.process_data(data, label, self.device)
        else:
          data = data.to(self.device)
          label = label.to(self.device)

        if hasattr(self.model, 'wsi_predict'):
          outputs = self.model.wsi_predict(data, **batch)
        else: # use universal code to update param
          outputs = self.model(data)

        logits, Y_prob, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
        slide_id = slide_ids.iloc[batch_idx]
        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error
        
        # Update progress bar
        progress_bar.set_postfix({
            'error': f'{error:.4f}'
        })
        
        # Append current predictions and labels to the lists
        all_Y_hat.append(Y_hat.cpu().numpy())
        all_label.append(label.cpu().numpy())
    
    # Close progress bar properly
    progress_bar.close()

    test_error /= len(loader)

    # convert the lists of all predictions and labels to numpy arrays
    all_Y_hat = np.concatenate(all_Y_hat)
    all_label = np.concatenate(all_label)
    
    # Calculate comprehensive metrics
    metrics = {}
    
    # Accuracy from error
    metrics['accuracy'] = 1 - test_error
    
    # ROC AUC
    if self.n_classes == 2:
      auc = roc_auc_score(all_labels, all_probs[:, 1])
      aucs = []
      metrics['auc'] = auc
    else:
      aucs = []
      binary_labels = label_binarize(all_labels, classes=[i for i in range(self.n_classes)])
      for class_idx in range(self.n_classes):
        if class_idx in all_labels:
          fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
          aucs.append(calc_auc(fpr, tpr))
        else:
          aucs.append(float('nan'))

      auc = np.nanmean(np.array(aucs))
      metrics['auc'] = auc
      metrics['class_auc'] = aucs
    
    # F1 Score (macro)
    f1 = f1_score(all_label, all_Y_hat, average='macro')
    metrics['f1_macro'] = f1
    
    # F1 Score (weighted)
    f1_weighted = f1_score(all_label, all_Y_hat, average='weighted')
    metrics['f1_weighted'] = f1_weighted
    
    # Class-wise F1 Score
    f1_per_class = f1_score(all_label, all_Y_hat, average=None)
    metrics['f1_per_class'] = f1_per_class
    
    # Precision (macro and weighted)
    precision_macro = precision_score(all_label, all_Y_hat, average='macro')
    precision_weighted = precision_score(all_label, all_Y_hat, average='weighted')
    metrics['precision_macro'] = precision_macro
    metrics['precision_weighted'] = precision_weighted
    
    # Class-wise precision
    precision_per_class = precision_score(all_label, all_Y_hat, average=None)
    metrics['precision_per_class'] = precision_per_class
    
    # Recall (macro and weighted)
    recall_macro = recall_score(all_label, all_Y_hat, average='macro')
    recall_weighted = recall_score(all_label, all_Y_hat, average='weighted')
    metrics['recall_macro'] = recall_macro
    metrics['recall_weighted'] = recall_weighted
    
    # Class-wise recall
    recall_per_class = recall_score(all_label, all_Y_hat, average=None)
    metrics['recall_per_class'] = recall_per_class
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(all_label, all_Y_hat)
    metrics['cohens_kappa'] = kappa
    
    # Confusion Matrix
    cm = confusion_matrix(all_label, all_Y_hat)
    metrics['confusion_matrix'] = cm
    
    # Classification Report (for logging purposes)
    cr = classification_report(all_label, all_Y_hat, output_dict=True)
    metrics['classification_report'] = cr

    # Log major metrics
    self.logger.success(f"Evaluation Results:", timestamp=True)
    self.logger.success(f"  Error: {test_error:.4f}")
    self.logger.success(f"  Accuracy: {metrics['accuracy']:.4f}")
    self.logger.success(f"  AUC: {auc:.4f}")
    self.logger.success(f"  F1 Score (macro): {f1:.4f}")
    self.logger.success(f"  Precision (macro): {precision_macro:.4f}")
    self.logger.success(f"  Recall (macro): {recall_macro:.4f}")
    self.logger.success(f"  Cohen's Kappa: {kappa:.4f}")
    
    # Log confusion matrix
    self.logger.info("Confusion Matrix:")
    self.logger.info(f"\n{cm}")
    
    if self.verbose:
      for i in range(self.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        self.logger.info('Class {}: Accuracy {:.4f}, Correct {}/{}'.format(i, acc, correct, count))
        self.logger.info(f'  Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}')

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(self.n_classes):
      results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    
    if self.verbose:
      self.logger.debug("Results dictionary created")
    
    df = pd.DataFrame(results_dict)

    return patient_results, test_error, auc, acc_logger, df, f1, metrics

  def eval_model(self, ckpt_path):
    self.logger.info(f"Loading model checkpoint from: {ckpt_path}", timestamp=True)
    
    if hasattr(self.model, 'load_model'):
      if self.verbose: 
        self.logger.info('Using built-in API to load the checkpoint...')
      self.model.load_model(ckpt_path)
    else:
      ckpt = torch.load(ckpt_path, map_location = 'cpu')
      msg = self.model.load_state_dict(ckpt)
      self.logger.info('Loading results: {}'.format(msg))
        
    self.logger.info("Running evaluation on test data...", timestamp=True)
    func = self.summary_subtyping
    patient_results, test_error, auc, acc_logger, df, f1, metrics = func(self.test_loader)
    
    self.logger.success(f"Evaluation complete - Error: {test_error:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}", timestamp=True)
    return patient_results, test_error, auc, acc_logger, df, f1, metrics