import torch
import numpy as np
import segmentation_models_pytorch as smp

model = smp.Unet("resnet18",
                 encoder_weights=None,
                 decoder_channels=(64, 32, 16),
                 decoder_depth=3,
                 add_segmentation_head=False)

model = smp.UnetPlusPlus()

model.eval()


images = torch.rand(1,3,1024,1024)
print("Image shape: ",images.shape)
with torch.inference_mode():
    mask = model(images)

print("UNET keys: ",mask.keys())
print('Mask shape', mask['segmentation_masks'].shape)
print('Encoder shape', mask['encoder_output'].shape)
