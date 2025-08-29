from .unet_base import BaseUnet


class ESD(BaseUnet):
    def __init__(
        self,
        unet_weights,
        name="ESD",
        **kwargs,
    ):
        super().__init__(name=name, unet_weights=unet_weights, **kwargs)

    def load_lora(self, lora_path, lora_scale):
        lora_dir = lora_path.parent
        weight_name = lora_path.name
        self.pipeline.load_lora_weights(lora_dir, weight_name=weight_name)
        self.pipeline._lora_scale = lora_scale