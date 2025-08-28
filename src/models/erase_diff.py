from .unet_base import BaseUnet


class EraseDiff(BaseUnet):
    def __init__(
        self,
        unet_weights,
        name="EraseDiff",
        **kwargs,
    ):
        super().__init__(name=name, unet_weights=unet_weights, **kwargs)
