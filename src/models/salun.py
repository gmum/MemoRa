from .unet_base import BaseUnet


class SalUn(BaseUnet):
    def __init__(
        self,
        unet_weights,
        name="SalUn",
        **kwargs,
    ):
        super().__init__(name=name, unet_weights=unet_weights, **kwargs)
