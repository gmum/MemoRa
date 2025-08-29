from .unet_base import BaseUnet


class FMN(BaseUnet):
    def __init__(
        self,
        unet_weights,
        name="FMN",
        **kwargs,
    ):
        super().__init__(name=name, unet_weights=unet_weights, **kwargs)
