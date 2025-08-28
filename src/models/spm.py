from .unet_base import BaseUnet


class SPM(BaseUnet):
    def __init__(
        self,
        unet_weights,
        name="SPM",
        **kwargs,
    ):
        super().__init__(name=name, unet_weights=unet_weights, **kwargs)
