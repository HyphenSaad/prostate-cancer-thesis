import math
import os
import time
import cv2
import openslide
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import multiprocessing as mp
import matplotlib.pyplot as plt
from xml.dom import minidom
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200000000000

from utils.wsi_core.wsi_utils import save_patch_iter_bag_hdf5, \
  initialize_hdf5_bag, \
  save_hdf5, \
  screen_coords, \
  is_black_patch, \
  is_white_patch, \
  to_percentiles

from utils.wsi_core.wsi_util_classes import IsInContourV1, \
  IsInContourV2, \
  IsInContourV3Easy, \
  IsInContourV3Hard, \
  ContourCheckingFn

from utils.file_utils import load_pkl, save_pkl

class ImgReader:
  default_dims = [1.0, 2.0, 4.0, 8.0, 16.0]
  
  def __init__(
    self,
    filename,
    verbose = False
  ) -> None:
    self.verbose = verbose
    self.filename = filename
    dtype = filename.split('.')[-1]
    
    if dtype in ['tif', 'svs','ndpi', 'tiff']:
      self.openslide = True
      self.handle = openslide.OpenSlide(filename)
      self._shape = self.handle.level_dimensions[0]
    else:
      self.openslide = False
      img = cv2.imread(filename)[:, :, ::-1] # to RGB
      h, w, _ = img.shape
      # openslide (width, height)
      self.img = img
      self._shape = [w, h]

  def read_region(
    self,
    location,
    level,
    size
  ):
    # convert coors, the coors always on level 0
    x, y = location
    w, h = size
    _w = int(w*self.level_downsamples[level])
    _h = int(h*self.level_downsamples[level])

    if self.openslide:
      img = self.handle.read_region(location, 0, (_w, _h)).resize((w, h)).convert('RGB')
    else:
      img = self.img[y: y + _h, x: x + _w].copy()
      img = Image.fromarray(img).resize((w, h))
    return img
  
  def __read(
    self,
    location,
    level,
    size
  ):
    w, h = size
    _w = int(w*self.level_downsamples[level])
    _h = int(h*self.level_downsamples[level])
    r = 1/self.level_downsamples[level]

    if _w < 20000 or _h < 20000:
      img = self.handle.read_region(location, 0, (_w, _h)).resize((w, h))
    else:
      step = 10000
      img = []
      x, y = location
      ex, ey = _w + x, _h + y
      
      xx = list(range(x, ex, step))
      xx = xx if ex in xx else xx + [ex]
      yy = list(range(y, ey, step))
      yy = yy if ey in yy else yy + [ey]

      # top to down
      counter = 0
      for _yy in yy:
        temp = []
        for _xx in xx:
          t = np.array(self.handle.read_region((_xx, _yy), 0, (step, step)))
          t = cv2.resize(t, None, fx=r, fy=r)
          temp.append(t)
          counter += 1
          if self.verbose: print(counter, len(yy)*len(xx))
        temp = np.concatenate(temp, axis=1)
        img.append(temp)

      img = np.concatenate(img, axis=0)
      img = Image.fromarray(img)
    return img

  @property
  def dimensions(self):
    return self.level_dimensions[0]

  @property
  def level_count(self):
    return len(self.default_dims)
  
  @property
  def level_downsamples(self):
    shape = [self._shape[0]/r[0] for r in self.level_dimensions]
    return shape
  
  @property
  def level_dimensions(self):
    shape = [(int(self._shape[0]/r), int(self._shape[1]/r)) for r in self.default_dims]
    return shape
  
  def get_best_level_for_downsample(self, scale):
    preset = [i*i for i in self.level_downsamples]
    err = [abs(i-scale) for i in preset]
    level = err.index(min(err))
    return level

  def close(self):
    pass

class WholeSlideImage(object):
    def __init__(
      self,
      path,
      verbose = False
    ):
      self.verbose = verbose
      self.name = os.path.splitext(os.path.basename(path))[0]
      self.wsi = openslide.open_slide(path)

      level_dim = self.wsi.level_dimensions
      if self.verbose:
        print(self.wsi.level_dimensions)
        print(level_dim)
      
      if len(level_dim) == 1:
        if self.verbose: print('ImgReader is adopted to speed up ...')
        self.wsi = ImgReader(path, verbose = self.verbose)
  
      self.level_downsamples = self._assert_level_downsamples()
      self.level_dim = self.wsi.level_dimensions
  
      self.contours_tissue = None
      self.contours_tumor = None
      self.hdf5_file = None
      if self.verbose: print("WSI Reader is initialized ...")

    def get_open_slide(self):
      return self.wsi

    def init_XML(
      self,
      xml_path
    ):
      def _create_contour(coord_list):
        return np.array(
          [
            [
              [
                int(float(coord.attributes['X'].value)), 
                int(float(coord.attributes['Y'].value))
              ]
            ] for coord in coord_list
          ], 
          dtype = 'int32'
        )

      xmldoc = minidom.parse(xml_path)
      annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
      self.contours_tumor  = [_create_contour(coord_list) for coord_list in annotations]
      self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def init_txt(
      self,
      annot_path
    ):
      def _create_contours_from_dict(annot):
        all_cnts = []
        
        for idx, annot_group in enumerate(annot):
          contour_group = annot_group['coordinates']
          if annot_group['type'] == 'Polygon':
            for idx, contour in enumerate(contour_group):
              contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
              all_cnts.append(contour) 
          else:
            for idx, sgmt_group in enumerate(contour_group):
              contour = []
              for sgmt in sgmt_group: 
                contour.extend(sgmt)

              contour = np.array(contour).astype(np.int32).reshape(-1,1,2)    
              all_cnts.append(contour) 
              
        return all_cnts
      
      with open(annot_path, "r") as f:
        annot = f.read()
        annot = eval(annot)

      self.contours_tumor  = _create_contours_from_dict(annot)
      self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def init_segmentation(
      self,
      mask_file
    ):
      # load segmentation results from pickle file
      asset_dict = load_pkl(mask_file)
      self.holes_tissue = asset_dict['holes']
      self.contours_tissue = asset_dict['tissue']

    def save_segmentation(
      self,
      mask_file
    ):
      # save segmentation results using pickle
      asset_dict = {
        'holes': self.holes_tissue,
        'tissue': self.contours_tissue
      }
      save_pkl(mask_file, asset_dict)

    def segment_tissue(
      self, 
      seg_level = 0,
      sthresh = 20,
      sthresh_up = 255,
      mthresh = 7,
      close = 0,
      use_otsu = False,
      filter_params = { 'a_t': 100 },
      ref_patch_size = 512,
      exclude_ids = [],
      keep_ids = []
    ):
      """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
      """
      
      def _filter_contours(
        contours,
        hierarchy,
        filter_params
      ):
        """
          Filter contours by: area.
        """
        filtered = []

        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []
        
        # loop through foreground contour indices
        for cont_idx in hierarchy_1:
          # actual contour
          cont = contours[cont_idx]
          
          # indices of holes contained in this contour (children of parent contour)
          holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
          
          # take contour area (includes holes)
          a = cv2.contourArea(cont)
          
          # calculate the contour area of each hole
          hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
          
          # actual area of foreground contour region
          a = a - np.array(hole_areas).sum()
          
          if a == 0: continue
          if tuple((filter_params['a_t'],)) < tuple((a,)): 
            filtered.append(cont_idx)
            all_holes.append(holes)

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]
        hole_contours = []

        for hole_ids in all_holes:
          unfiltered_holes = [contours[idx] for idx in hole_ids ]
          unfilered_holes = sorted(
            unfiltered_holes,
            key = cv2.contourArea,
            reverse = True
          )
          
          # take max_n_holes largest holes by area
          unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
          filtered_holes = []
          
          # filter these holes
          for hole in unfilered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
              filtered_holes.append(hole)

          hole_contours.append(filtered_holes)
        return foreground_contours, hole_contours
      
      img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
      img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert to HSV space
      img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # apply median blurring
      
      # thresholding
      if use_otsu:
        _, img_otsu = cv2.threshold(
          img_med,
          0,
          sthresh_up,
          cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
      else:
        _, img_otsu = cv2.threshold(
          img_med,
          sthresh,
          sthresh_up,
          cv2.THRESH_BINARY
        )

      # ,orphological closing
      if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

      scale = self.level_downsamples[seg_level]
      scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
      filter_params = filter_params.copy()
      
      filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
      filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area
      
      # to find and filter contours
      contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # find contours 
      hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
      
      if filter_params: 
        # necessary for filtering out artifacts
        foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  

      self.contours_tissue = self.scale_contour_dim(foreground_contours, scale)
      self.holes_tissue = self.scale_holes_dim(hole_contours, scale)

      if len(keep_ids) > 0:
        contour_ids = set(keep_ids) - set(exclude_ids)
      else:
        contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

      self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
      self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]
      self.__vis_wsi_cache = img

    def vis_wsi(
      self,
      vis_level = 0,
      color = (0, 255, 0),
      hole_color = (0, 0, 255),
      annot_color = (255, 0, 0), 
      line_thickness = 250,
      max_size = None,
      top_left = None,
      bot_right = None,
      custom_downsample = 1,
      view_slide_only = False,
      number_contours = False,
      seg_display = True,
      annot_display = True
    ):
      downsample = self.level_downsamples[vis_level]
      scale = [1/downsample[0], 1/downsample[1]]
      
      if top_left is not None and bot_right is not None:
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)
        
        # read specific region
        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
      else:
        top_left = (0,0)
        region_size = self.level_dim[vis_level]
        
        # may produce wrong results, make sure the seg_level is consistent with the vis_level
        img = self.__vis_wsi_cache

      if not view_slide_only:
        offset = tuple(-(np.array(top_left) * scale).astype(int))
        line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
        
        if self.contours_tissue is not None and seg_display:
          if not number_contours:
            cv2.drawContours(
              img,
              self.scale_contour_dim(self.contours_tissue, scale), 
              -1,
              color,
              line_thickness,
              lineType = cv2.LINE_8,
              offset = offset
            )
          else: # just to add numbering to each contour
            for idx, cont in enumerate(self.contours_tissue):
              contour = np.array(self.scale_contour_dim(cont, scale))
              M = cv2.moments(contour)
              cX = int(M["m10"] / (M["m00"] + 1e-9))
              cY = int(M["m01"] / (M["m00"] + 1e-9))
              
              # draw the contour and put text next to center
              cv2.drawContours(
                img,
                [contour],
                -1,
                color,
                line_thickness,
                lineType = cv2.LINE_8,
                offset = offset
              )
              
              cv2.putText(
                img, 
                "{}".format(idx),
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                10
              )

          for holes in self.holes_tissue:
            cv2.drawContours(
              img,
              self.scale_contour_dim(holes, scale), 
              -1,
              hole_color,
              line_thickness,
              lineType = cv2.LINE_8
            )
        
        if self.contours_tumor is not None and annot_display:
          cv2.drawContours(
            img,
            self.scale_contour_dim(self.contours_tumor, scale), 
            -1,
            annot_color,
            line_thickness,
            lineType = cv2.LINE_8,
            offset = offset
          )
      
      img = Image.fromarray(img).convert('RGB')
  
      w, h = img.size
      if custom_downsample > 1:
        img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

      if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size/w if w > h else max_size/h
        img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
      
      return img

    def create_patches_bag_hdf5(
      self,
      save_path,
      patch_level = 0,
      patch_size = 256,
      step_size = 256,
      save_coord = True,
      **kwargs
    ):
      contours = self.contours_tissue
      contour_holes = self.holes_tissue

      if self.verbose: print("Creating patches for: ", self.name, "...",)
      elapsed = time.time()
      
      for idx, cont in enumerate(contours):
        patch_gen = self._get_patch_generator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)
        
        if self.hdf5_file is None:
          try: first_patch = next(patch_gen)
          except StopIteration: continue

          file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
          self.hdf5_file = file_path

        for patch in patch_gen:
          save_patch_iter_bag_hdf5(patch)

      return self.hdf5_file

    def _get_patch_generator(
      self,
      cont,
      cont_idx,
      patch_level,
      save_path,
      patch_size = 256,
      step_size = 256,
      custom_downsample = 1,
      white_black = True,
      white_thresh = 15,
      black_thresh = 50,
      contour_fn = 'four_pt',
      use_padding = True
    ):
      start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
      if self.verbose: print("Bounding Box:", start_x, start_y, w, h)
      if self.verbose: print("Contour Area:", cv2.contourArea(cont))
      
      if custom_downsample > 1:
        assert custom_downsample == 2 
        target_patch_size = patch_size
        patch_size = target_patch_size * 2
        step_size = step_size * 2
        if self.verbose: print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(
          custom_downsample, patch_size, patch_size, target_patch_size, target_patch_size))

      patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
      ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
      
      step_size_x = step_size * patch_downsample[0]
      step_size_y = step_size * patch_downsample[1]
      
      if isinstance(contour_fn, str):
        if contour_fn == 'four_pt':
          cont_check_fn = IsInContourV3Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
        elif contour_fn == 'four_pt_hard':
          cont_check_fn = IsInContourV3Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
        elif contour_fn == 'center':
          cont_check_fn = IsInContourV2(contour=cont, patch_size=ref_patch_size[0])
        elif contour_fn == 'basic':
          cont_check_fn = IsInContourV1(contour=cont)
        else:
            raise NotImplementedError
      else:
        assert isinstance(contour_fn, ContourCheckingFn)
        cont_check_fn = contour_fn

      img_w, img_h = self.level_dim[0]
      if use_padding:
        stop_y = start_y+h
        stop_x = start_x+w
      else:
        stop_y = min(start_y+h, img_h-ref_patch_size[1])
        stop_x = min(start_x+w, img_w-ref_patch_size[0])

      count = 0
      for y in range(start_y, stop_y, step_size_y):
        for x in range(start_x, stop_x, step_size_x):
          if not self.is_in_contours(cont_check_fn, (x,y), self.holes_tissue[cont_idx], ref_patch_size[0]): 
            # point not inside contour and its associated holes
            continue
          
          count += 1
          patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
          if custom_downsample > 1:
            patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))
          
          if white_black:
            if is_black_patch(np.array(patch_PIL), rgbThresh = black_thresh) or \
              is_white_patch(np.array(patch_PIL), satThresh = white_thresh): 
                continue

          patch_info = {
            'x': x // (patch_downsample[0] * custom_downsample), 
            'y': y // (patch_downsample[1] * custom_downsample),
            'cont_idx': cont_idx,
            'patch_level': patch_level, 
            'downsample': self.level_downsamples[patch_level],
            'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])//custom_downsample),
            'level_dim': self.level_dim[patch_level],
            'patch_PIL': patch_PIL,
            'name': self.name,
            'save_path': save_path
          }

          yield patch_info
      if self.verbose: print("patches extracted: {}".format(count))

    @staticmethod
    def is_in_holes(
      holes,
      pt,
      patch_size
    ):
      for hole in holes:
        if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
          return 1
      return 0

    @staticmethod
    def is_in_contours(
      cont_check_fn,
      pt,
      holes = None,
      patch_size = 256
    ):
      if cont_check_fn(pt):
        if holes is not None: return not WholeSlideImage.is_in_holes(holes, pt, patch_size)
        else: return 1
      return 0
    
    @staticmethod
    def scale_contour_dim(
      contours,
      scale
    ):
      return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scale_holes_dim(
      contours,
      scale
    ):
      return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

    def _assert_level_downsamples(self):
      level_downsamples = []
      dim_0 = self.wsi.level_dimensions[0]
      
      for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
        estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
      
      return level_downsamples

    def process_contours(
      self,
      save_path,
      patch_level = 0,
      patch_size = 256,
      step_size = 256,
      **kwargs
    ):
      save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
      
      if self.verbose: print("Creating patches for: ", self.name, "...",)
      elapsed = time.time()
      n_contours = len(self.contours_tissue)
      
      if self.verbose: print("Total number of contours to process: ", n_contours)
      fp_chunk_size = math.ceil(n_contours * 0.05)
      init = True
      
      for idx, cont in enumerate(self.contours_tissue):
        if (idx + 1) % fp_chunk_size == fp_chunk_size:
          if self.verbose: print('Processing contour {}/{}'.format(idx, n_contours))
        
        asset_dict, attr_dict = self.process_contour(
          cont,
          self.holes_tissue[idx],
          patch_level,
          save_path,
          patch_size,
          step_size,
          **kwargs
        )
        
        if len(asset_dict) > 0:
          if init:
            save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
            init = False
          else:
            save_hdf5(save_path_hdf5, asset_dict, mode='a')

      return self.hdf5_file

    def process_contour(
      self,
      cont,
      contour_holes,
      patch_level,
      save_path,
      patch_size = 256,
      step_size = 256,
      contour_fn = 'four_pt',
      use_padding = True,
      top_left = None,
      bot_right = None
    ):
      start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

      patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
      ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
      
      img_w, img_h = self.level_dim[0]
      if use_padding:
        stop_y = start_y + h
        stop_x = start_x + w
      else:
        stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
        stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)
      
      if self.verbose: print("Bounding Box:", start_x, start_y, w, h)
      if self.verbose: print("Contour Area:", cv2.contourArea(cont))

      if bot_right is not None:
        stop_y = min(bot_right[1], stop_y)
        stop_x = min(bot_right[0], stop_x)
      if top_left is not None:
        start_y = max(top_left[1], start_y)
        start_x = max(top_left[0], start_x)

      if bot_right is not None or top_left is not None:
        w, h = stop_x - start_x, stop_y - start_y
        if w <= 0 or h <= 0:
          if self.verbose: print("Contour is not in specified ROI, skip")
          return {}, {}
        else:
          if self.verbose: print("Adjusted Bounding Box:", start_x, start_y, w, h)
  
      if isinstance(contour_fn, str):
        if contour_fn == 'four_pt':
          cont_check_fn = IsInContourV3Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
        elif contour_fn == 'four_pt_hard':
          cont_check_fn = IsInContourV3Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
        elif contour_fn == 'center':
          cont_check_fn = IsInContourV2(contour=cont, patch_size=ref_patch_size[0])
        elif contour_fn == 'basic':
          cont_check_fn = IsInContourV1(contour=cont)
        else:
          raise NotImplementedError
      else:
        assert isinstance(contour_fn, ContourCheckingFn)
        cont_check_fn = contour_fn

      step_size_x = step_size * patch_downsample[0]
      step_size_y = step_size * patch_downsample[1]

      x_range = np.arange(start_x, stop_x, step=step_size_x)
      y_range = np.arange(start_y, stop_y, step=step_size_y)
      x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
      coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

      num_workers = mp.cpu_count()
      if num_workers > 4: num_workers = 4
      pool = mp.Pool(num_workers)

      iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
      results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
      pool.close()
      results = np.array([result for result in results if result is not None])
      
      if self.verbose: print('Extracted {} coordinates'.format(len(results)))

      if len(results) > 1:
        asset_dict = { 'coords': results }
        attr = {
          'patch_size': patch_size,
          'patch_level': patch_level,
          'downsample': self.level_downsamples[patch_level],
          'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])),
          'level_dim': self.level_dim[patch_level],
          'name': self.name,
          'save_path': save_path
        }

        attr_dict = { 'coords': attr }
        return asset_dict, attr_dict
      else:
        return {}, {}

    @staticmethod
    def process_coord_candidate(
      coord,
      contour_holes,
      ref_patch_size,
      cont_check_fn
    ):
      if WholeSlideImage.is_in_contours(cont_check_fn, coord, contour_holes, ref_patch_size):
        return coord
      else:
        return None

    def vis_heatmap(
      self,
      scores,
      coords,
      vis_level = -1, 
      top_left = None,
      bot_right = None,
      patch_size = (256, 256), 
      blank_canvas = False,
      canvas_color = (220, 20, 50),
      alpha = 0.4,
      blur = False,
      overlap = 0.0, 
      segment = True,
      use_holes = True,
      convert_to_percentiles = False,
      binarize = False,
      thresh = 0.5,
      max_size = None,
      custom_downsample = 1,
      cmap = 'coolwarm'
    ):
      """
      Args:
        scores (numpy array of float): Attention scores 
        coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
        vis_level (int): WSI pyramid level to visualize
        patch_size (tuple of int): Patch dimensions (relative to lvl 0)
        blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
        canvas_color (tuple of uint8): Canvas color
        alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
        blur (bool): apply gaussian blurring
        overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
        segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that 
                        self.contours_tissue and self.holes_tissue are not None
        use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
        convert_to_percentiles (bool): whether to convert attention scores to percentiles
        binarize (bool): only display patches > threshold
        threshold (float): binarization threshold
        max_size (int): Maximum canvas size (clip if goes over)
        custom_downsample (int): additionally downscale the heatmap by specified factor
        cmap (str): name of matplotlib colormap to use
      """

      if vis_level < 0:
        vis_level = self.wsi.get_best_level_for_downsample(32)

      downsample = self.level_downsamples[vis_level]
      scale = [1/downsample[0], 1/downsample[1]] # scaling from 0 to desired level

      if len(scores.shape) == 2:
        scores = scores.flatten()

      if binarize:
        if thresh < 0: threshold = 1.0 / len(scores)
        else: threshold =  thresh
      else:
        threshold = 0.0

      # calculate size of heatmap and filter coordinates/scores outside specified bbox region
      if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)
        coords = coords - top_left
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)
      else:
        region_size = self.level_dim[vis_level]
        top_left = (0,0)
        bot_right = self.level_dim[0]
        w, h = region_size

      patch_size  = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
      coords = np.ceil(coords * np.array(scale)).astype(int)
      
      if self.verbose: print('\ncreating heatmap for: ')
      if self.verbose: print('top_left: ', top_left, 'bot_right: ', bot_right)
      if self.verbose: print('w: {}, h: {}'.format(w, h))
      if self.verbose: print('scaled patch size: ', patch_size)

      # normalize filtered scores 
      if convert_to_percentiles:
        scores = to_percentiles(scores) 
      scores /= 100
      
      # calculate the heatmap of raw attention scores (before colormap) 
      # by accumulating scores over overlapped regions
      # heatmap overlay: tracks attention score over each pixel of heatmap
      # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
      overlay = np.full(np.flip(region_size), 0).astype(float)
      counter = np.full(np.flip(region_size), 0).astype(np.uint16)      
      
      count = 0
      for idx in range(len(coords)):
        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
          if binarize:
            score = 1.0
            count += 1
        else:
          score = 0.0
        
        # accumulate attention
        overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
        # accumulate counter
        counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1

      if binarize:
        if self.verbose: print('\nbinarized tiles based on cutoff of {}'.format(threshold))
        if self.verbose: print('identified {}/{} patches as positive'.format(count, len(coords)))
      
      # fetch attended region and average accumulated attention
      zero_mask = counter == 0

      if binarize:
        overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
      else:
        overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]

      del counter
      
      if blur:
        overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1-overlap)).astype(int) * 2 +1), 0)  

      if segment:
        tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
      
      if not blank_canvas: # downsample original image and use as canvas
        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
      else: # use blank canvas 
        img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255))) 

      if self.verbose: print('\ncomputing heatmap image')
      if self.verbose: print('total of {} patches'.format(len(coords)))
      twenty_percent_chunk = max(1, int(len(coords) * 0.2))

      if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
      
      for idx in range(len(coords)):
        if (idx + 1) % twenty_percent_chunk == 0:
          if self.verbose: print('progress: {}/{}'.format(idx, len(coords)))
        
        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
          # attention block
          raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
          
          # image block (either blank canvas or orig image)
          img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

          # color block (cmap applied to attention block)
          color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

          if segment:
            # tissue mask block
            mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] 
            
            # copy over only tissue masked portion of color block
            img_block[mask_block] = color_block[mask_block]
          else:
            # copy over entire color block
            img_block = color_block

          # rewrite image block
          img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()
      
      if self.verbose: print('Done')
      del overlay

      if blur:
        img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

      if alpha < 1.0:
        img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024)
      
      img = Image.fromarray(img)
      w, h = img.size

      if custom_downsample > 1:
        img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

      if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size/w if w > h else max_size/h
        img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
      
      return img
    
    def block_blending(
      self,
      img,
      vis_level,
      top_left,
      bot_right,
      alpha = 0.5,
      blank_canvas = False,
      block_size = 1024
    ):
      if self.verbose: print('\ncomputing blend')
      
      downsample = self.level_downsamples[vis_level]
      w = img.shape[1]
      h = img.shape[0]
      
      block_size_x = min(block_size, w)
      block_size_y = min(block_size, h)
      if self.verbose: print('using block size: {} x {}'.format(block_size_x, block_size_y))

      shift = top_left # amount shifted w.r.t. (0, 0)
      for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
        for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
          # 1. convert wsi coordinates to image coordinates via shift and scale
          x_start_img = int((x_start - shift[0]) / int(downsample[0]))
          y_start_img = int((y_start - shift[1]) / int(downsample[1]))
          
          # 2. compute end points of blend tile, careful not to go over the edge of the image
          y_end_img = min(h, y_start_img+block_size_y)
          x_end_img = min(w, x_start_img+block_size_x)

          if y_end_img == y_start_img or x_end_img == x_start_img: 
            continue
          
          # 3. fetch blend block and size
          blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img] 
          blend_block_size = (x_end_img-x_start_img, y_end_img-y_start_img)
          
          if not blank_canvas: # 4. read actual wsi block as canvas block
            pt = (x_start, y_start)
            canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))     
          else: # 4. OR create blank canvas block
            canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

          # 5. blend color block and canvas block
          img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
      return img

    def get_seg_mask(
      self,
      region_size,
      scale,
      use_holes = False,
      offset = (0, 0)
    ):
      if self.verbose: print('\ncomputing foreground tissue mask')
      tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
      contours_tissue = self.scale_contour_dim(self.contours_tissue, scale)
      offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

      contours_holes = self.scale_holes_dim(self.holes_tissue, scale)
      contours_tissue, contours_holes = zip(
        *sorted(
          zip(contours_tissue, contours_holes), 
          key = lambda x: cv2.contourArea(x[0]),
          reverse = True
        )
      )
      
      for idx in range(len(contours_tissue)):
        cv2.drawContours(
          image = tissue_mask,
          contours = contours_tissue,
          contourIdx = idx,
          color = (1),
          offset = offset,
          thickness = -1
        )
        
        if use_holes:
          cv2.drawContours(
            image = tissue_mask,
            contours = contours_holes[idx],
            contourIdx = -1,
            color = (0),
            offset = offset,
            thickness = -1
          )
              
      tissue_mask = tissue_mask.astype(bool)
      if self.verbose: print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
      
      return tissue_mask