import torch
import os
import numpy as np
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from mil_models import find_mil_model
from utils.common_utils import EarlyStopping, AccuracyLogger
from utils.clam_utils import print_network, get_split_loader, calculate_error
from utils.logger import Logger

logger = Logger()

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
    logger.empty_line()
    logger.info("Training Fold {}!", self.fold)
    
    writer_dir = os.path.join(self.result_dir, str(self.fold))
    if not os.path.isdir(writer_dir): os.mkdir(writer_dir)
    self.writer = SummaryWriter(writer_dir, flush_secs = 15)

    logger.info("Training on {} samples", len(self.train_split))
    logger.info("Validating on {} samples", len(self.val_split))
    logger.info("Testing on {} samples", len(self.test_split))

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

    print_network(model)
    return model

  def get_loss_function(self):
    if hasattr(self.model, 'loss_function'):
      if self.verbose: logger.info('The loss function defined in the MIL model is adopted...')
      loss_function = self.model.loss_function
    else:
      if self.verbose: logger.info('Cross Entropy Loss is adopted as the loss function...')
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

    for epoch in range(self.max_epochs):
      train_loop_func(epoch)
      stop = validate_func(epoch)
      if stop: break
    
    msg = self.model.load_state_dict(torch.load(os.path.join(self.result_dir, "s_{}_checkpoint.pt".format(fold))))
    logger.info('Loading Trained Model...')
    if self.verbose: logger.text(msg)
    
    # test_func on val loader
    logger.info("Evaluating on validation set...")
    _, val_error, val_auc, _, _, val_f1 = test_func(self.val_loader)
    # test on test loader
    logger.info("Evaluating on test set...")
    results_dict, test_error, test_auc, acc_logger, _, test_f1 = test_func(self.test_loader)
    
    logger.info('Test Error: {:.4f}, ROC AUC: {:.4f}, F1 Score: {:.4f}', test_error, test_auc, test_f1)
    logger.info('Val Error: {:.4f}, ROC AUC: {:.4f}, F1 Score: {:.4f}', val_error, val_auc, val_f1)

    for i in range(self.n_classes):
      acc, correct, count = acc_logger.get_summary(i)
      logger.info('class {}: acc {}, correct {}/{}', i, acc, correct, count)
      self.writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    self.writer.add_scalar('final/val_error', val_error, 0)
    self.writer.add_scalar('final/val_auc', val_auc, 0)
    self.writer.add_scalar('final/test_error', test_error, 0)
    self.writer.add_scalar('final/test_auc', test_auc, 0)
    self.writer.close()
    
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, test_f1, val_f1

  def train_loop_subtyping(self, epoch):   
    self.model.train()
    acc_logger = AccuracyLogger(n_classes = self.n_classes)
    train_loss = 0.0
    train_error = 0.0

    logger.empty_line()
    logger.info('Epoch: {}', epoch)
    
    progress_bar = tqdm(
      self.train_loader,
      desc=f"Training Epoch {epoch}",
      unit="batch",
      disable=not self.verbose
    )
    
    for batch_idx, batch in enumerate(progress_bar):
      iteration = epoch * len(self.train_loader) + batch_idx
      kwargs = {}
      data = batch['features']
      label = batch['label']
      kwargs['iteration'] = iteration
      kwargs['image_call'] = batch['image_call']
      
      # Core 1: if your model need specific pre-process of data and label, please implement following function,
      # we will call to keep code clean
      if hasattr(self.model, 'process_data'):
        data, label = self.model.process_data(data, label, self.device)
      else:
        data = data.to(self.device)
        label = label.to(self.device)
        
      # Core 2: If your model has special optimizing strategy, e.g., using mutilpe optimizers, please
      # define your own update parameter code in one_step function. You may also need to define optimizes in you MIL model.
      if hasattr(self.model, 'one_step'):
        outputs = self.model.one_step(data, label, **kwargs)
        loss = outputs['loss']
        if 'call_scheduler' in outputs.keys():
          self.call_scheduler = outputs['call_scheduler']
        logits, Y_prob, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
      else:
        # use univer code to update param
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
        logger.error('logits: {}', logits)
        logger.error('Y_prob: {}', Y_prob)
        logger.error('loss: {}', loss)
        raise RuntimeError('Found Nan number')
      
      if (batch_idx + 1) % 20 == 0 and self.verbose:
        bag_size = data[0].shape[0] if isinstance(data, list) else data.shape[0]
        log_message = f'batch {batch_idx}'
        for k, v in outputs.items():
          if 'loss' in k:
            log_message += f', {k}:{v.item():.4f}'
        log_message += f', label: {label.item()}, bag_size: {bag_size}'
        logger.info(log_message)
        
      # Update progress bar
      progress_bar.set_postfix({
        'loss': f"{loss_value:.4f}"
      })
              
      error = calculate_error(Y_hat, label)
      train_loss += loss_value
      train_error += error

    # calculate loss and error for epoch
    train_loss /= len(self.train_loader)
    train_error /= len(self.train_loader)

    logger.info('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}', epoch, train_loss, train_error)
    if self.writer:
      self.writer.add_scalar('train/loss', train_loss, epoch)
      self.writer.add_scalar('train/error', train_error, epoch)
        
    for i in range(self.n_classes):
      acc, correct, count = acc_logger.get_summary(i)
      logger.info('class {}: acc {}, correct {}/{}', i, acc, correct, count)
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
    
    with torch.no_grad():
      progress_bar = tqdm(
        self.val_loader, 
        desc=f"Validating Epoch {epoch}", 
        unit="batch",
        disable=not self.verbose
      )
      
      for batch_idx, batch in enumerate(progress_bar):
        if self.verbose:
          logger.info('Evaluating: [{}/{}]', batch_idx + 1, len(self.val_loader))
          
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
        
        val_loss += loss.item()
        error = calculate_error(Y_hat, label)
        val_error += error
        
        # Update progress bar
        progress_bar.set_postfix({
          'loss': f"{loss.item():.4f}",
          'error': f"{error:.4f}"
        })

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

    logger.info('Val Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1 score: {:.4f}', 
                val_loss, val_error, auc, f1)
    
    for i in range(self.n_classes):
      acc, correct, count = acc_logger.get_summary(i)
      logger.info('class {}: acc {}, correct {}/{}', i, acc, correct, count)     

    # val_error is better than val_loss
    self.early_stopping(epoch, val_error, self.model, ckpt_name = os.path.join(self.result_dir, "s_{}_checkpoint.pt".format(self.fold)))
    
    if self.early_stopping.early_stop:
      logger.info("Early stopping")
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
    
    with torch.no_grad():
      progress_bar = tqdm(
        loader, 
        desc="Evaluating", 
        unit="batch",
        disable=not self.verbose
      )
      
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
        
        # Append current predictions and labels to the lists
        all_Y_hat.append(Y_hat.cpu().numpy())
        all_label.append(label.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
          'error': f"{error:.4f}"
        })

    test_error /= len(loader)

    if self.n_classes == 2:
      auc = roc_auc_score(all_labels, all_probs[:, 1])
      aucs = []
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
        
    # convert the lists of all predictions and labels to numpy arrays
    all_Y_hat = np.concatenate(all_Y_hat)
    all_label = np.concatenate(all_label)

    # calculate the F1 score
    f1 = f1_score(all_label, all_Y_hat, average='macro')

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(self.n_classes):
      results_dict.update({'p_{}'.format(c): all_probs[:,c]})
      
    if self.verbose:
      logger.info("Results: {}", results_dict)
      
    df = pd.DataFrame(results_dict)

    return patient_results, test_error, auc, acc_logger, df, f1

  def eval_model(self, ckpt_path):
    if hasattr(self.model, 'load_model'):
      logger.info('Using built-in API to load the checkpoint...')
      self.model.load_model(ckpt_path)
    else:
      ckpt = torch.load(ckpt_path, map_location = 'cpu')
      msg = self.model.load_state_dict(ckpt)
      logger.info('Loading results: {}', msg)
        
    func = self.summary_subtyping
    logger.info("Evaluating model...")
    patient_results, test_error, auc, _, df, f1 = func(self.test_loader)
    logger.info("Evaluation complete - Error: {:.4f}, AUC: {:.4f}, F1: {:.4f}", test_error, auc, f1)
    return patient_results, test_error, auc, df, f1