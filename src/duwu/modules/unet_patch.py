from omegaconf import OmegaConf, DictConfig

import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.resnet import (
    ResnetBlock2D,
)
from diffusers.models.attention import (
    BasicTransformerBlock,
)


class UNet2DFromScratch(UNet2DConditionModel):

    def init_weight(self):
        """
        We zero out (initialize with small std) all layers that are followed by
        residual connections:
        - out layer of attentions
        - out layer of feedforward
        - out layer of resblocks

        This is similar to `zero_module(...)` at
        - https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/diffusionmodules/openaimodel.py#L242  # noqa
        - https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/attention.py  # noqa

        Note that in the second link above out layers of attentions are not zeroed out

        Due to how UNet architecture is designed, this causes the model output to be
        almost only dependent on timestep which could have great influence on the value of
        the middle layer.
        """

        for module in self.modules():
            if isinstance(module, BasicTransformerBlock):
                nn.init.normal_(module.attn1.to_out[0].weight, 0.0, 1e-5)
                if module.attn2 is not None:
                    nn.init.normal_(module.attn2.to_out[0].weight, 0.0, 1e-5)
                if isinstance(module.ff.net[-2], nn.Linear):
                    nn.init.normal_(module.ff.net[-2].weight, 0.0, 1e-5)
                else:
                    nn.init.normal_(module.ff.net[-1].weight, 0.0, 1e-5)
            if isinstance(module, ResnetBlock2D):
                nn.init.normal_(module.conv2.weight, 0.0, 1e-5)
        nn.init.normal_(self.conv_out.weight, 0.0, 1e-5)

    @classmethod
    def from_config(
        cls, config: str | dict | DictConfig, **kwargs
    ) -> "UNet2DFromScratch":
        if isinstance(config, str):
            config = cls.load_config(config, **kwargs)
        elif isinstance(config, DictConfig):
            config = OmegaConf.to_container(config)
        model = super().from_config(config)
        model.init_weight()
        return model


if __name__ == "__main__":

    unet = UNet2DFromScratch.from_config(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet"
    )
    print(unet)
