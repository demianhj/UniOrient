import torch,timm,random
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as L
from torchvision import transforms


from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class DinoWrapper(L.LightningModule):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, is_train: bool = False):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.freeze(is_train)

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [0,1] scale and properly size
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model.forward_features(self.processor(image))
        feat_unflat = self.model.get_intermediate_layers(self.processor(image), n=1, reshape=True)[0]

        return outputs["x_norm_patchtokens"], outputs['x_norm_clstoken'], feat_unflat
    
    def freeze(self, is_train:bool = False):
        print(f"======== image encoder is_train: {is_train} ========")
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = is_train

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            # model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
            model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            mean = (0.485, 0.456, 0.406) if "dino" in model_name else (0.5, 0.5, 0.5)
            std = (0.229, 0.224, 0.225) if "dino" in model_name else (0.5, 0.5, 0.5)
            processor = transforms.Normalize(mean=mean, std=std)
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err
            

def resize_tensor(input_tensor, scale):
     # Add a batch dimension if it doesn't exist (assuming N=1 for now)
    if len(input_tensor.shape) == 3:  # [C, H, W] -> [1, C, H, W]
        input_tensor = input_tensor.unsqueeze(0)

    original_size = [input_tensor.shape[-2], input_tensor.shape[-1]]
    target_size = (int(scale * original_size[0]), int(scale * original_size[1]))

    return F.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=False).squeeze(0)

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

