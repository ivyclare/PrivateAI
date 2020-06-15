"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        new_volume = np.moveaxis(volume, -1, 0)
        print("new volume", new_volume.shape)
        reshaped_volume = med_reshape(new_volume, new_shape=(new_volume.shape[0], self.patch_size, self.patch_size))
        pred = self.single_volume_inference(reshaped_volume.astype(np.float32))
        reshaped_pred = np.zeros_like(new_volume)
        print("hey", reshaped_volume.shape, pred.shape, reshaped_pred.shape)        
        resize_dim = [reshaped_pred.shape[2], reshaped_pred.shape[1]]
        print("lala", resize_dim)
        for i in range(reshaped_pred.shape[0]):
            reshaped_pred[i] = Image.fromarray(pred[i].astype(np.uint8)).resize(resize_dim)
            
        return np.moveaxis(reshaped_pred, 0, -1)

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        
        # slicing the x axis
        #test_tensor = torch.tensor(volume).unsqueeze(1)
        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
#         outputs = self.model(test_tensor.float().to(self.device)).cpu().detach()
#         _, slices = torch.max(outputs, 1)
#         slices = slices.numpy
        input_tensor = torch.tensor(volume).unsqueeze(1)
        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        outputs = self.model(input_tensor.float().to(self.device)).cpu().detach()
        _, slices = torch.max(outputs, 1)
    
        return slices.numpy()
        
