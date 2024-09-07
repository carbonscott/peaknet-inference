import torch
import yaml
from peaknet.modeling.convnextv2_bifpn_net import PeakNet, PeakNetConfig, SegHeadConfig
from peaknet.modeling.bifpn_config import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
from contextlib import nullcontext

class PeakNetInference:
    def __init__(self, config_path, weights_path):
        self.device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config           = self._load_config(config_path)
        self.autocast_context = self._setup_autocast()
        self.model            = self._setup_model()
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

    def _setup_autocast(self):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        dist_dtype = self.config.get('dist').get('dtype')
        mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dist_dtype]
        autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = mixed_precision_dtype)
        return autocast_context

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_model(self):
        # Config
        model_params              = self.config.get("model")
        backbone_params           = model_params.get("backbone")
        hf_model_config           = backbone_params.get("hf_config")
        bifpn_params              = model_params.get("bifpn")
        bifpn_block_params        = bifpn_params.get("block")
        bifpn_block_bn_params     = bifpn_block_params.get("bn")
        bifpn_block_fusion_params = bifpn_block_params.get("fusion")
        seghead_params            = model_params.get("seg_head")

        # Backbone
        backbone_config = ConvNextV2Config(**hf_model_config)

        # BiFPN
        bifpn_block_params["bn"]     = BNConfig(**bifpn_block_bn_params)
        bifpn_block_params["fusion"] = FusionConfig(**bifpn_block_fusion_params)
        bifpn_params["block"]        = BiFPNBlockConfig(**bifpn_block_params)
        bifpn_config                 = BiFPNConfig(**bifpn_params)

        # Seg head
        seghead_config = SegHeadConfig(**seghead_params)

        # PeakNet
        peaknet_config = PeakNetConfig(
            backbone = backbone_config,
            bifpn    = bifpn_config,
            seg_head = seghead_config,
        )

        return PeakNet(peaknet_config)

    def _load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(weights)

    def predict(self, input_tensor):
        with torch.no_grad():
            with self.autocast_context:
                output_tensor = self.model(input_tensor)
        return output_tensor
