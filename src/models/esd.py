from .unet_base import BaseUnet


class ESD(BaseUnet):
    def __init__(
        self,
        unet_weights,
        name="ESD",
        **kwargs,
    ):
        super().__init__(name=name, unet_weights=unet_weights, **kwargs)