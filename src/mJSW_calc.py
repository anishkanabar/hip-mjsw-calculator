import sys
sys.path.append('/research/projects/m303645_Anish/Current/')

import math
import copy 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
import imgviz
import slicerio
import nrrd
import base_osail_utils

from skimage.measure import regionprops, label, find_contours
from skimage.morphology import remove_small_objects, dilation, erosion
from skimage.transform import resize
from scipy.ndimage import rotate

from monai.data import NrrdReader

from hip_mjsw.segmentation_model import model

from circle_fit import taubinSVD

class mJSWCalculator():
    def __init__(self) -> None:
        self.segmentation_model = model.InferenceModel(
            #device=device
        )
        self.labels = {"femoral head": 0, "sourcil": 1}
        self.segmentation = None
        self.implant = False
    
    def reset(self):
        self.points = []
        self.lines = []
        self.results = {}
        self.spacing = None
        self.segmentation = None
        self.left_femhead_centroid = None
        self.right_femhead_centroid = None
        self.nsourcilsegments = None
        self.nfemheadsegments = None

    def calculate(self, image_path, plot=False):
        self.reset()
        self.image_path = image_path

        ## Get pixel spacing
        self.spacing = self._get_pixel_spacing(image_path=self.image_path)

        ## Get and resize segmentation
        self.segmentation = self._segment_images(image_path)
        self.segmentation = self._resize_annotations(self.segmentation, image_path)

        ls, rs = self._get_sourcil_countours(self.segmentation)

        lf, lfc, rf, rfc = self._get_femoral_head_countours(self.segmentation)
        # ls, rs, lf, rf = self._get_left_right_sides(s1, s2, f1, f2)
        output_dict = self._get_mJSW(ls, rs, lf, rf, self.spacing, image_path, plot) # left_mJSW, right_mJSW = 
        return output_dict
    
    def _get_pixel_spacing(self, image_path):
        """Checks DICOM header for information on pixel spacing"""
        try:
            dicom_data = pydicom.dcmread(image_path, force=True, stop_before_pixels=True)

            try:
                pixel_spacing = (dicom_data["PixelSpacing"].value[0],dicom_data["PixelSpacing"].value[1])
            except KeyError as e:
                pixel_spacing = (0,0)
            
            try:
                res = dicom_data["SpatialResolution"].value
            except KeyError as e:
                res = 0

        except FileNotFoundError as e:
            print("File not found")
            return 0
        
        if pixel_spacing[0] > 0:
            spacing = pixel_spacing[0]
        elif pixel_spacing[1] > 0:
            spacing = pixel_spacing[1]
        elif res > 0:
            spacing = res
        else:
            spacing = 0.2
        return spacing
    
    def _load_segmentations(self, path, labels):

        ## Load Data
        segmentation_info = slicerio.read_segmentation(path)
        voxels, _ = nrrd.read(path)
        
        ## Add channels dim if not present
        if len(voxels.shape) == 3:
            voxels = np.expand_dims(voxels, axis=0)

        ## Prep Empty Volume
        x = voxels.shape[1]
        y = voxels.shape[2]
        channels = len(segmentation_info["segments"])

        # Comment below print statement out when I am done debugging
        #print(f"Segmentation: {path} has {channels} channels with dims {voxels.shape}")
        output = np.zeros((x,y,channels))
        
        ## Loop through layers
        for i, segment in enumerate(segmentation_info["segments"]):
            
            ## Extract Metadata
            layer = segment["layer"]
            layer_name = segment["name"]
            labelValue = segment["labelValue"]
            
            ## Set up new layer based on voxel value from segmentation info
            layer_voxels = np.moveaxis(voxels, 0, -1)
            layer_voxels = np.squeeze(layer_voxels, axis=-2)
            indx = (layer_voxels[..., layer] == labelValue).nonzero()
            new_layer = np.zeros(layer_voxels[..., layer].shape)
            new_layer[indx] = labelValue
            
            ## Assign the new layer to the output based on defined channel order
            output[...,labels[str.lower(layer_name)]] = new_layer
        
        output = np.where(np.moveaxis(output, -1, 0) > 0.0, 1, 0)
        return output
        
    def _segment_images(self, images):
        """Runs segmentation model"""
        model_output = self.segmentation_model.predict(images)
        return model_output
    
    def _resize_annotations(self, annotation, image_path):
        '''
        Resizes segmentations back to the dimensions of the original image.
        Assumes that the images were resized for segmentation by padding to square and resizing.
        '''
        # Read NRRDs to get initial dimensions
        if 'nrrd' in image_path:
            reader = NrrdReader()
            image = reader.read(data=image_path).array
        else:
            image = pydicom.dcmread(image_path).pixel_array
        w, h = image.shape[:2]
        self.midpoint_x = w//2
        long_side = max(h, w)
        short_side = min(h, w)

        # Resize each channel in annotation
        #annotation_stack = [img[...] for img in annotation]
        output_stack = list()
        for channel in annotation[0]:
            channel = channel.cpu().numpy()
            resized_channel = cv2.resize(
                channel, 
                dsize=(long_side, long_side), 
                interpolation=cv2.INTER_NEAREST
            )

            if h == long_side:
                cropped_channel = resized_channel[(long_side-short_side)//2:(long_side+short_side)//2, :]
            else:
                cropped_channel = resized_channel[:, (long_side-short_side)//2:(long_side+short_side)//2] 
            
            output_stack.append(cropped_channel)

        return output_stack

    def _get_sourcil_countours(self, segmentations):
        segmentation = erosion(segmentations[1])
        # print(segmentations[1].shape)
        # x = segmentations[1].shape[0]
        # y = segmentations[1].shape[1]
        # resize(segmentations[1], (x*0.9,y*0.9))

        label_img = label(segmentation)
        regions = regionprops(label_img)

        perim_dict = {}
        self.nsourcilsegments = len(regions)

        for i,region in enumerate(regions):
            perim = region.perimeter
            if perim >=225:
                perim_dict[i] = perim #d_x

        #print(f'sourcil: {perim_dict}')
        
        idx_sorted = sorted(perim_dict, key=perim_dict.get, reverse=True)

        if len(perim_dict)>=2:
            #idx_sorted = sorted(perim_dict, key=perim_dict.get, reverse=True)
            region_1 = regions[idx_sorted[0]]
            contour_1 = find_contours(segmentation)[idx_sorted[0]]
            region_2 = regions[idx_sorted[1]]
            contour_2 = find_contours(segmentation)[idx_sorted[1]]

            # set left and right sides
            if region_1.centroid[1] < region_2.centroid[1]:
                left_sourcil = contour_1
                right_sourcil = contour_2
            else:
                left_sourcil = contour_2
                right_sourcil = contour_1
        else:
            region = regions[idx_sorted[0]]
            contour = find_contours(segmentation)[idx_sorted[0]]

            if region.centroid[1] < self.midpoint_x:
                self.sides = ['Left']
                left_sourcil = contour
                right_sourcil = None
            else:
                self.sides = ['Right']
                left_sourcil = None
                right_sourcil = contour

        return left_sourcil, right_sourcil
    
    def _get_femoral_head_countours(self, segmentations):
        segmentation = erosion(segmentations[0])
        label_img = label(segmentation)
        regions = regionprops(label_img)

        perim_dict = {}

        self.nfemheadsegments = len(regions)

        for i,region in enumerate(regions):
            perim = region.perimeter
            if perim >= 250:
                perim_dict[i] = perim #d_x

        for k in perim_dict.keys():
            if perim_dict[k] <= 100:
                perim_dict.pop(k)

        #print(f'femhead: {perim_dict}')

        idx_sorted = sorted(perim_dict, key=perim_dict.get, reverse=True)

        if len(perim_dict)>=2:
            #idx_sorted = sorted(perim_dict, key=perim_dict.get, reverse=True)
            region_1 = regions[idx_sorted[0]]
            contour_1 = find_contours(segmentation)[idx_sorted[0]]
            region_2 = regions[idx_sorted[1]]
            contour_2 = find_contours(segmentation)[idx_sorted[1]]

            # set left and right sides
            if region_1.centroid[1] < region_2.centroid[1]:
                left_femhead = contour_1
                #self.left_femhead_centroid = region_1.centroid
                self.left_femhead_centroid = self._adjust_centroid(contour_1, region_1.centroid, 'Left') 
                right_femhead = contour_2
                #self.right_femhead_centroid = region_2.centroid
                self.right_femhead_centroid = self._adjust_centroid(contour_2, region_2.centroid, 'Right')
            else:
                left_femhead = contour_2
                #self.left_femhead_centroid = region_2.centroid
                self.left_femhead_centroid = self._adjust_centroid(contour_2, region_2.centroid, 'Left')
                right_femhead = contour_1
                #self.right_femhead_centroid = region_1.centroid
                self.right_femhead_centroid = self._adjust_centroid(contour_1, region_1.centroid, 'Right')
            self.sides = ['Left', 'Right']
        else:
            region = regions[idx_sorted[0]]
            contour = find_contours(segmentation)[idx_sorted[0]]
            if region.centroid[1] < self.midpoint_x:
                self.sides = ['Left']
                left_femhead = contour
                right_femhead = None
                self.left_femhead_centroid = self._adjust_centroid(contour, region.centroid, 'Left')
                self.right_femhead_centroid = None
            else:
                self.sides = ['Right']
                right_femhead = contour
                left_femhead = None
                self.right_femhead_centroid = self._adjust_centroid(contour, region.centroid, 'Right')
                self.left_femhead_centroid = None

        return left_femhead, self.left_femhead_centroid, right_femhead, self.right_femhead_centroid

    def _adjust_centroid(self, contour, centroid, side):
        y_max = centroid[0]
        x_min = centroid[1]
        if side == 'Right':
            indices = np.where((contour[:,0]<=y_max) & (contour[:,1]<=x_min))
        else:
            indices = np.where((contour[:,0]<=y_max) & (contour[:,1]>=x_min))
        pts = contour[indices]
        xc, yc, r, sigma = taubinSVD(pts)
        return xc, yc 

    def _get_mJSW(self, left_sourcil, right_sourcil, left_femhead, right_femhead, spacing, image_path, plot):
        if len(self.sides) == 2:
            sourcils = [left_sourcil, right_sourcil]
            femheads = [left_femhead, right_femhead]
        elif self.sides == ['Left']:
            sourcils = [left_sourcil]
            femheads = [left_femhead]
        else:
            sourcils = [right_sourcil]
            femheads = [right_femhead]
        output_dict = {}
        #output_dict['spacing'] = self.spacing

        for side, sourcil, femhead in zip(self.sides, sourcils, femheads):
            if side == 'Left':
                medial_x_lim = self.left_femhead_centroid[1]
                if len(sourcil[sourcil[:,1]<medial_x_lim])>0:
                    lateral_sourcil = sourcil[sourcil[:,1]<medial_x_lim]
                else:
                    lateral_sourcil = min(sourcil, key=lambda point: point[1])
                    lateral_sourcil = np.reshape(lateral_sourcil, (1,2))
            else:
                medial_x_lim = self.right_femhead_centroid[1]
                if len(sourcil[sourcil[:,1]>medial_x_lim])>0:
                    lateral_sourcil = sourcil[sourcil[:,1]>medial_x_lim]
                else:
                    lateral_sourcil = max(sourcil, key=lambda point: point[1])
                    lateral_sourcil = np.reshape(lateral_sourcil, (1,2))

            mJSW_df = pd.DataFrame()
            lateral_sourcil_idx_list = []
            femhead_idx_list = []
            min_distance_list = []

            for i, node in enumerate(lateral_sourcil):
                fem_head_idx, min_distance = self._closest_point(node, femhead)
                lateral_sourcil_idx_list.append(i)
                femhead_idx_list.append(fem_head_idx)
                min_distance_list.append(min_distance)

            mJSW_df['lateral_sourcil_idx'] = lateral_sourcil_idx_list
            mJSW_df['femhead_idx'] = femhead_idx_list
            mJSW_df['min_distance'] = min_distance_list
            #mJSW = np.mean(mJSW_df.sort_values(by=['min_distance'])['min_distance'][:25])

            try:
                pixel_distance = math.sqrt(mJSW_df.sort_values(by=['min_distance'])['min_distance'].iloc[0])
                mJSW = pixel_distance*spacing
                #print(f'mJSW of {mJSW:.2f} mm') #pixel_distance, spacing
            except KeyError:
                pass

            output_dict[f'{side} mJSW'] = round(mJSW, 2)
            #output_dict['nsourcilsegments'] = self.nsourcilsegments
            #output_dict['nfemheadsegments'] = self.nfemheadsegments
            #print(side + ' mJSW ' + str(round(mJSW, 2)) + ' mm')
        
            if plot: 
                self._plot(lateral_sourcil, femhead, mJSW_df)
                if 'nrrd' in image_path:
                    reader = NrrdReader()
                    image = rotate(reader.read(data=image_path).array, angle=270)
                else:
                    image = pydicom.dcmread(image_path).pixel_array
                plt.axis('off')
                plt.imshow(image, cmap='bone')
                plt.imshow(erosion(self.segmentation[1]), alpha=0.3, cmap='bone')
                plt.imshow(erosion(self.segmentation[0]), alpha=0.3, cmap='bone')

        return output_dict
    
    def _closest_point(self, node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node)**2, axis=1)
        return np.argmin(dist_2), dist_2[np.argmin(dist_2)]
    
    def _plot(self, sourcil, femhead, mJSW_df):

        lateral_sourcil_idx = int(mJSW_df.sort_values(by=['min_distance'], axis=0).iloc[0]['lateral_sourcil_idx'])
        femhead_idx = int(mJSW_df.sort_values(by=['min_distance'], axis=0).iloc[0]['femhead_idx'])

        s_x = sourcil[lateral_sourcil_idx][1]
        s_y = sourcil[lateral_sourcil_idx][0]
        #print(f'sourcil coords: {s_x}, {s_y}')

        f_x = femhead[femhead_idx][1]
        f_y = femhead[femhead_idx][0]
        #print(f'femhead coords: {f_x}, {f_y}')

        #calculate distance
        #print(math.dist([s_x, s_y], [f_x, f_y]))

        if len(self.sides) == 2:
            plt.plot(self.left_femhead_centroid[1], self.left_femhead_centroid[0], marker="o", markersize=2, color='r')
            plt.plot(self.right_femhead_centroid[1], self.right_femhead_centroid[0], marker="o", markersize=2, color='r')
        elif self.sides == ['Left']:
            plt.plot(self.left_femhead_centroid[1], self.left_femhead_centroid[0], marker="o", markersize=2, color='r')
        else:
            plt.plot(self.right_femhead_centroid[1], self.right_femhead_centroid[0], marker="o", markersize=2, color='r')
        plt.plot(s_x,s_y,marker="o", markersize=2, color='b')
        plt.plot(f_x,f_y,marker="o", markersize=2, color='g')
        #plt.plot([s_x,f_x],[s_y,f_y], color='r')