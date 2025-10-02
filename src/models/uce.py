from .unet_base import BaseUnet


class UCE(BaseUnet):
    def __init__(
        self,
        unet_weights,
        name="UCE",
        **kwargs,
    ):
        super().__init__(name=name, unet_weights=unet_weights, **kwargs)
