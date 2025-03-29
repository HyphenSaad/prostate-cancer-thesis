import pandas as pd
import numpy as np
import os 
import h5py
import torch
from scipy import stats
from torch.utils.data import Dataset
from PIL import Image
from multiprocessing.pool import ThreadPool

from utils.clam_utils import generate_split, nth

class GenericWSIClassificationDataset(Dataset):
    def __init__(self,
      csv_path,
      label_column,
      label_dict = {},
      patient_voting = 'max',
      verbose = False,
      **kwargs
    ):
      self.label_dict = label_dict
      self.num_classes = len(set(self.label_dict.values()))
      self.verbose = verbose
      self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
      self.label_column = label_column
      self.seed = 1

      slide_data = pd.read_csv(csv_path, dtype=str)
      slide_data.drop(['data_provider'], axis=1, inplace=True)
      slide_data.rename(columns={'image_id': 'slide_id'}, inplace=True)

      slide_data['case_id'] = slide_data['slide_id']
      
      slide_data = self.dataframe_preparation(slide_data, self.label_dict, self.label_column)
      self.slide_data = slide_data

      self.patient_data_preparation(patient_voting)
      self.class_ids_preparation()

      if verbose:
        self.summarize()

      self.data_cache = {}

    def class_ids_preparation(self):
      # store ids corresponding each class at the patient or case level
      self.patient_cls_ids = [[] for i in range(self.num_classes)]		
      for i in range(self.num_classes):
        self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

      # store ids corresponding each class at the slide level
      self.slide_cls_ids = [[] for i in range(self.num_classes)]
      for i in range(self.num_classes):
        self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_preparation(self, patient_voting='max'):
      patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
      patient_labels = []
      
      for p in patients:
        locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
        assert len(locations) > 0
        label = self.slide_data['label'][locations].values
        if patient_voting == 'max': label = label.max() # get patient label (MIL convention)
        elif patient_voting == 'maj': label = stats.mode(label)[0]
        else: raise NotImplementedError
        patient_labels.append(label)
      
      self.patient_data = { 'case_id':patients, 'label': np.array(patient_labels) }

    @staticmethod
    def dataframe_preparation(data, label_dict, label_col):
      if label_col != 'label':
        data['label'] = data[label_col].copy()

      data.reset_index(drop=True, inplace=True)
      
      for i in data.index:
        key = data.loc[i, 'label']
        data.at[i, 'label'] = label_dict[key]

      return data

    def __len__(self):
      return len(self.slide_data)

    def summarize(self):
      print("label column: {}".format(self.label_column))
      print("label dictionary: {}".format(self.label_dict))
      print("number of classes: {}".format(self.num_classes))
      print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
      for i in range(self.num_classes):
        print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
        print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def create_splits(
      self,
      k = 5,
      val_num = (25, 25),
      test_num = (40, 40),
    ):
      settings = {
        'n_splits' : k, 
        'val_num' : val_num, 
        'test_num': test_num,
        'seed': self.seed,
        'cls_ids': self.slide_cls_ids,
        'samples': len(self.slide_data)
      }

      self.split_gen = generate_split(**settings)

    def set_splits(self, start_from = None):
      if start_from: ids = nth(self.split_gen, start_from)
      else: ids = next(self.split_gen)
      self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, backbone, patch_size, all_splits, split_key = 'train'):
      split = all_splits[split_key]
      split = split.dropna().reset_index(drop = True)

      if len(split) > 0:
        mask = self.slide_data['slide_id'].isin(split.tolist())
        df_slice = self.slide_data[mask].reset_index(drop = True)
        split = GenericSplit(df_slice, num_classes = self.num_classes)
        split.set_backbone(backbone)
        split.set_patch_size(patch_size)
      else: split = None
      
      return split

    def get_merged_split_from_df(self, all_splits, split_keys = ['train']):
      merged_split = []
      for split_key in split_keys:
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop = True).tolist()
        merged_split.extend(split)

      if len(split) > 0:
        mask = self.slide_data['slide_id'].isin(merged_split)
        df_slice = self.slide_data[mask].reset_index(drop = True)
        split = GenericSplit(df_slice, num_classes = self.num_classes)
      else: split = None
      
      return split

    def return_splits(self, backbone = None, patch_size = '', from_id = True, csv_path = None):
      if from_id:
        if len(self.train_ids) > 0:
          train_data = self.slide_data.loc[self.train_ids].reset_index(drop = True)
          train_split = GenericSplit(train_data, num_classes = self.num_classes)
          train_split.set_backbone(backbone)
          train_split.set_patch_size(patch_size)
        else: train_split = None
        
        if len(self.val_ids) > 0:
          val_data = self.slide_data.loc[self.val_ids].reset_index(drop = True)
          val_split = GenericSplit(val_data, num_classes = self.num_classes)
          val_split.set_backbone(backbone)
          val_split.set_patch_size(patch_size)
        else: val_split = None
        
        if len(self.test_ids) > 0:
          test_data = self.slide_data.loc[self.test_ids].reset_index(drop = True)
          test_split = GenericSplit(test_data, num_classes = self.num_classes)
          test_split.set_backbone(backbone)
          test_split.set_patch_size(patch_size)
        else: test_split = None
      else:
        assert csv_path
        all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
        train_split = self.get_split_from_df(backbone, patch_size, all_splits, 'train')
        val_split = self.get_split_from_df(backbone, patch_size, all_splits, 'val')
        test_split = self.get_split_from_df(backbone, patch_size, all_splits, 'test')
      
      return train_split, val_split, test_split

    def get_list(self, ids):
      return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
      return self.slide_data['label'][ids]

    def __getitem__(self, idx):
      return None

    def test_split_gen(self, return_descriptor = False, verbose = False):
      if return_descriptor:
        index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
        columns = ['train', 'val', 'test']
        df = pd.DataFrame(
          np.full((len(index), len(columns)), 0, dtype = np.int32), 
          index = index,
          columns = columns
        )

      count = len(self.train_ids)
      if verbose: print('\nnumber of training samples: {}'.format(count))
      
      labels = self.getlabel(self.train_ids)
      unique, counts = np.unique(labels, return_counts = True)
      
      for u in range(len(unique)):
        if verbose: print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
        if return_descriptor: df.loc[index[u], 'train'] = counts[u]
      
      count = len(self.val_ids)
      if verbose: print('\nnumber of val samples: {}'.format(count))
      
      labels = self.getlabel(self.val_ids)
      unique, counts = np.unique(labels, return_counts = True)
      
      for u in range(len(unique)):
        if verbose: print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
        if return_descriptor: df.loc[index[u], 'val'] = counts[u]

      count = len(self.test_ids)
      if verbose: print('\nnumber of test samples: {}'.format(count))
      
      labels = self.getlabel(self.test_ids)
      unique, counts = np.unique(labels, return_counts = True)
      
      for u in range(len(unique)):
        if verbose: print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
        if return_descriptor: df.loc[index[u], 'test'] = counts[u]

      assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
      assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
      assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

      if return_descriptor: return df

    def save_split(self, filename):
      train_split = self.get_list(self.train_ids)
      val_split = self.get_list(self.val_ids)
      test_split = self.get_list(self.test_ids)
      df_tr = pd.DataFrame({'train': train_split})
      df_v = pd.DataFrame({'val': val_split})
      df_t = pd.DataFrame({'test': test_split})
      df = pd.concat([df_tr, df_v, df_t], axis=1) 
      df.to_csv(filename, index = False)


class GenericMILDataset(GenericWSIClassificationDataset):
  def __init__(
    self,
    extract_patches_dir = None, 
    patches_dir = None,
    features_pt_directory = None,
    **kwargs
  ):
    super(GenericMILDataset, self).__init__(**kwargs)

    self.extract_patches_dir = extract_patches_dir
    self.patches_dir = patches_dir
    self.features_pt_directory = features_pt_directory
    self.kaggle_feature_path = kwargs.get('kaggle_feature_path', None)

  def __getitem__(self, index):
    slide_id = self.slide_data['slide_id'][index]
    label = self.slide_data['label'][index]

    is_google_colab = os.getcwd().lower() == '/content'
    if not hasattr(self, 'extract_patches_dir'):
      if is_google_colab: self.extract_patches_dir = "/content/output/extract_patches"
      else: self.extract_patches_dir = "/kaggle/working/output/extract_patches"
    if not hasattr(self, 'patches_dir'):
      if is_google_colab: self.patches_dir = "/content/output/create_patches/patches"
      else: self.patches_dir = "/kaggle/working/output/create_patches/patches"
    if not hasattr(self, 'features_pt_directory'):
      if is_google_colab: self.features_pt_directory = "/content/output/extract_features/pt_files"
      else: self.features_pt_directory = "/kaggle/working/output/extract_features/pt_files"

    img_root = os.path.join(self.extract_patches_dir, slide_id)
    coords_path = os.path.join(self.patches_dir, '{}.h5'.format(slide_id))

    try:
      with h5py.File(coords_path, 'r') as f:
        coords = f['coords'][:]
    except:
      pass

    def image_caller(coor_index):
      img_list = []
      for c in coor_index:
        _x, _y = coords[c]
        path = os.path.join(img_root, f'{_x}_{_y}_{self.patch_size}_{self.patch_size}.jpg')
        img = Image.open(path).convert('RGB')
        img_list.append(img)
      return img_list

    if self.kaggle_feature_path:
      self.features_pt_directory = self.kaggle_feature_path
    else:
      raise NotImplementedError('kaggle_feature_path is not set. Please set it to the correct path.')

    
    # full_path = os.path.join(self.features_pt_directory, self.backbone, '{}.pt'.format(slide_id))
    # self.features_pt_directory = f'/kaggle/input/{self.backbone}-features-512x512-50-overlap-panda/{self.backbone}'
    full_path = os.path.join(self.features_pt_directory, '{}.pt'.format(slide_id))
    try:
      features = torch.load(full_path)
    except:
      raise RuntimeError('failed to load:{}'.format(full_path))

    label = torch.LongTensor([label])
    output = {
      'features': features, 'label': label,
      'image_call': image_caller,
      'coords': coords,
    }
    return output

  def set_backbone(self, backbone):
    self.backbone = backbone
  
  def set_patch_size(self, size):
    self.patch_size = size

class GenericSplit(GenericMILDataset):
    def __init__(self, slide_data, num_classes = 2):      
      self.use_h5 = False
      self.slide_data = slide_data
      self.num_classes = num_classes
      self.slide_cls_ids = [[] for i in range(self.num_classes)]
      self.data_cache = {}
      for i in range(self.num_classes):
        self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
      return len(self.slide_data)
        
    def set_backbone(self, backbone):
      self.backbone = backbone

    def set_patch_size(self, size):
      self.patch_size = size

    def pre_loading(self, thread = os.cpu_count()):
      self.cache_flag = True
      ids = list(range(len(self)))
      exe = ThreadPool(thread)
      exe.map(self.__getitem__, ids)