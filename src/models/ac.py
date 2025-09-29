from .unet_base import BaseUnet


class AC(BaseUnet):
    def __init__(
        self,
        unet_weights,
        name="AC",
        **kwargs,
    ):
        super().__init__(name=name, unet_weights=unet_weights, **kwargs)