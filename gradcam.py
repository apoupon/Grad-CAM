import torch 
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM():
    def __init__(self, model, target_layer):
        self.model = model
        model.eval()
        self.activations = {}
        self.target_layer = target_layer

        # call hook functions during forward pass for activations maps and backward pass for gradients
        self.target_layer.register_forward_hook(self.save_output)
        self.target_layer.register_backward_hook(self.save_grad_output)

    # hook function to get the output of the target layer
    def save_output(self, model, input, output):
        self.activations["output"] = output.detach()

    # hook function to get the gradient output of the target layer
    def save_grad_output(self, model, grad_input, grad_output):
        self.activations["grad_output"] = grad_output[0].detach()

    # average gradients within each channel
    def average_gradients(self, grad_output):
        alpha_k = torch.mean(grad_output[0], dim=[1, 2])
        return alpha_k

    # aggregate hidden representation across channel dimension
    def aggregate_hidden_representation(self, alpha_k, output):
        out = torch.permute(output.squeeze(), (1, 2, 0))
        a_ij = torch.mul(alpha_k, out).sum(dim = [2])
        # ReLU function to suppress negative attributions
        relu = nn.ReLU()
        a_ij = relu(a_ij)
        return a_ij
    

    # display heatmap on the input image
    def heatmap(self, image, upsampled_mask):
        image = np.array(image.squeeze().permute(1, 2, 0))
        image = (image - image.min()) / (image.max() - image.min())  

        mask = np.array(upsampled_mask)
        mask = (mask - mask.min()) / (mask.max() - mask.min())  

        mask = np.uint8(255 * mask)  
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay the heatmap onto the original image with some transparency
        heatmap_img = 0.6 * image + 0.4 * heatmap 

        return heatmap_img


    # apply the procedure detailed in the GradCAM paper
    def __call__(self, image, category):
        self.model.zero_grad()
        
        y = self.model(image)[0]
        output = self.activations['output']

        y_grad = torch.tensor([0.0 for x in range(1000)], requires_grad=True)
        y_grad.data[category] = 1.0
        y.backward(gradient=y_grad)
        grad_output = self.activations['grad_output']

        alpha_k = self.average_gradients(grad_output)
        a_ij = self.aggregate_hidden_representation(alpha_k, output)
        upsampled = F.interpolate(a_ij.unsqueeze(0).unsqueeze(0), size = (224, 224), mode = 'bilinear').squeeze()
        heatmap = self.heatmap(image, upsampled)
        return heatmap

