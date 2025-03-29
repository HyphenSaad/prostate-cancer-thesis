import numpy as np
import torch
import copy


class AccuracyLogger(object):
  def __init__(self, n_classes):
    super(AccuracyLogger, self).__init__()
    self.n_classes = n_classes
    self.initialize()

  def initialize(self):
    self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
  
  def log(self, Y_hat, Y):
    Y_hat = int(Y_hat)
    Y = int(Y)
    self.data[Y]["count"] += 1
    self.data[Y]["correct"] += (Y_hat == Y)
  
  def log_batch(self, Y_hat, Y):
    Y_hat = np.array(Y_hat).astype(int)
    Y = np.array(Y).astype(int)
    for label_class in np.unique(Y):
      cls_mask = Y == label_class
      self.data[label_class]["count"] += cls_mask.sum()
      self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
  
  def get_summary(self, c):
    count = self.data[c]["count"] 
    correct = self.data[c]["correct"]
    if count == 0: acc = None
    else: acc = float(correct) / count
    return acc, correct, count


class EarlyStopping:
  def __init__(self, patience = 20, stop_epoch = 50, verbose = False):
    self.patience = patience
    self.stop_epoch = stop_epoch
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.inf
    self.best_epoch = -1
    self.best_model_state = None

  def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
    score = -val_loss

    if self.best_score is None:
      self.best_score = score
      self.best_epoch = epoch
      self.save_checkpoint(val_loss, model, ckpt_name)
      # Store a copy of the best model state dict for recovery
      if not hasattr(model, 'save_model'):
        self.best_model_state = copy.deepcopy(model.state_dict())
    elif score < self.best_score:
      self.counter += 1
      if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience and epoch > self.stop_epoch:
        self.early_stop = True
    else:
      self.best_score = score
      self.best_epoch = epoch
      self.save_checkpoint(val_loss, model, ckpt_name)
      # Update best model state
      if not hasattr(model, 'save_model'):
        self.best_model_state = copy.deepcopy(model.state_dict())
      self.counter = 0

  def save_checkpoint(self, val_loss, model, ckpt_name):
    if self.verbose:
      print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

    if hasattr(model, 'save_model'):
      model.save_model(ckpt_name)
    else:
      if self.verbose: print('Warning: You model did not provide a `save_model` function, we use torch.save to save you model...')
      torch.save(model.state_dict(), ckpt_name)
    
    self.val_loss_min = val_loss

  def restore_best_weights(self, model):
    """Restore model to best weights if available"""
    if self.best_model_state is not None:
      if self.verbose:
        print(f'Restoring model to best epoch ({self.best_epoch+1}) with val_loss: {self.val_loss_min:.6f}')
      model.load_state_dict(self.best_model_state)
      return True
    return False