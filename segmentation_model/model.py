################################################################################
# This files contains the InferenceModel for the hip and pelvis annotator.
################################################################################

import monai as mn
import os
from base_osail_utils import osail_utils
from tqdm import tqdm
import torch
from collections import OrderedDict
from .anatomical_structs import labels_dict

import matplotlib.pyplot as plt

################################################################################
# -C: InferenceModel

class InferenceModel():
    """
    A class to run inference on a single image or list of images.
    Args:
        device (str, optional): The device to load the model. Could be 'cpu'
            or 'cuda'. Defaults to 'cuda'.
    
    """
    def __init__(self, silent_mode=False, device='cuda'):
        self.silent = silent_mode
        weight_path = os.path.join(os.path.dirname(__file__), 'data','mgss2.ckpt')

        self.model = mn.networks.nets.FlexibleUNet(
            in_channels=1,
            out_channels=2,
            backbone='efficientnet-b6',
            dropout=0.1
        )
        checkpoint = torch.load(weight_path)
        model_weight = checkpoint['state_dict']
        # modify the key names
        state_dict = OrderedDict()
        for i, (key, value) in enumerate(model_weight.items()):
            splitted_key = key.split(".")
            
            new_key = ".".join(splitted_key[1:])  # get rid of the string part "model"
            state_dict[new_key] = value

        self.model.load_state_dict(state_dict, strict=False)
        self.device = device
        self.transforms = self._build_transforms()
        self.post_transforms = self._build_post_transforms()
        self.model.eval()
    
    ############################################################################
    # -M: predict_items
    
    def predict(
        self, 
        items, 
        batch_size=8, 
        num_workers=4, 
        dp=False, 
        **kwargs
    ):
        """_summary_: Predict the view of a list of items.

        Args:
            items (list): a single item or a list of items that could be passed to
            osail_utils.io.LoadImage to load an image.
            batch_size (int, optional): batch size for the dataloader. 
                Defaults to 8.
            num_workers (int, optional): Number of workers to be used by the
                data laoder. Defaults to 4.
            dp (bool, optional): Wheter or not to use torch.nn.DataParallel. 
                Defaults to False.
                return_logits (bool, optional): Whether or not to return the
                    logits of the model after softmax. Defaults to False.
        Returns:
            A 22x512x512 array where each channel is a different structure
            
        """
        self.model = self.model.to(self.device)
        if dp:
            assert torch.cuda.device_count() > 1, 'DP is not available'
            self.model = torch.nn.DataParallel(self.model)
        if not isinstance(items, list):
            items = [items]
        images = [{'image': item} for item in items]
        ds = mn.data.Dataset(data=images, transform=self.transforms)
        #plt.imsave('/research/projects/m303645_Anish/Current/hip_mjsw/segmentation_model/testimgdcm.png', torch.squeeze(ds[0]['image']))
        dl = mn.data.DataLoader(
            ds, 
            batch_size=batch_size, 
            num_workers=num_workers,
            **kwargs
        )
        results = []
        for batch in tqdm(dl, total=len(dl), disable=self.silent):
            images = batch['image'].to(self.device)
            output = self.post_transforms(self.model(images))
            for mask in output:
                results.append(mask.cpu())

        return results

    ############################################################################
    # -M: build_transforms
    
    def _build_transforms(self):
        """_summary_: Build the preprocessing transforms for the model.

        Returns:
            a monai.transforms.Compose object.
        """
        return mn.transforms.Compose([
            osail_utils.io.LoadImageD(
                keys=['image'], 
                pad=True, 
                target_shape=(1024, 1024),
                percentile_clip=2.5, 
                normalize=True, 
                standardize=True, 
                dtype=None,
                squeeze=True
            ),
            mn.transforms.EnsureChannelFirstD(keys=['image'], channel_dim='no_channel'),
            #mn.transforms.Rotate90D(keys=['image'], k=3, spatial_axes=(0, 1), lazy=False), # for NRRD
            mn.transforms.ScaleIntensityD(keys=['image']),
            mn.transforms.ToTensorD(keys=['image'])
        ])
    

    ############################################################################
    # -M: build_post_transforms
    
    def _build_post_transforms(self):
        """_summary_: Build the postprocessing transforms for the model.

        Returns:
            a monai.transforms.Compose object.
        """
        return mn.transforms.Compose(
            [
                mn.transforms.AsDiscrete(threshold=0.5, to_onehot=None)
            ]
        )
        
    ############################################################################
    # -M: help

    def help(self):
        print('Here is a list of output channels and their corresponding indices:')
        for struct in labels_dict:
            print(f"Name: {struct['name']:65}\tChannel: {struct['channel']:4d}")



        
