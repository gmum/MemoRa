from .unet_base import BaseUnet


class ScissorHands(BaseUnet):
    def __init__(
        self,
        unet_weights,
        name="ScissorHands",
        **kwargs,
    ):
        super().__init__(name=name, unet_weights=unet_weights, **kwargs)
