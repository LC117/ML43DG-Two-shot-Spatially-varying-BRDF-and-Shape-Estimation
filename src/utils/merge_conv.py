import torch

from src.utils.common_layers import INReLU

"""
    def Fusion2DBlock(
    prevIn: Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]],
    filters: int,
    kernel_size: int,
    stride: int,
    downscale: bool = True,
    activation=INReLU,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    lmain = Conv2D("main_conv", prevIn[0], filters, kernel_size, activation=activation)
    laux = Conv2D("aux_conv", prevIn[1], filters, kernel_size, activation=activation)

    mixInput = [lmain, laux]
    prevMixOutput = prevIn[2]
    if prevMixOutput is not None:
        mixInput.append(prevMixOutput)

    mixIn = tf.concat(mixInput, -1, "mix_input")
    lmix = Conv2D("mix_conv", mixIn, filters, kernel_size, activation=activation)

    lmix = tf.add_n([laux, lmain, lmix], "mix_summation")

    if stride > 1:
        if downscale:
            lmain = MaxPooling("main_pool", lmain, 3, strides=stride, padding="SAME")
            laux = MaxPooling("aux_pool", laux, 3, strides=stride, padding="SAME")
            lmix = MaxPooling("mix_pool", lmix, 3, strides=stride, padding="SAME")
        else:
            lmain = upsample("main_upsample", lmain, factor=stride)
            laux = upsample("aux_upsample", laux, factor=stride)
            lmix = upsample("mix_upsample", lmix, factor=stride)

    return (lmain, laux, lmix)
    """


class MergeConv(torch.nn.Module):
    def __init__(self, in_channels: int,
    filters: int,
    kernel_size: int,
    stride: int,
    downscale: bool = True,
    activation=INReLU):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MergeConv, self).__init__()
        self.conv_main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, filters, kernel_size, padding='same'),
            activation(filters)
        )
        self.conv_flash = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, filters, kernel_size, padding='same'),
            activation(filters)
        )
        self.conv_concat = torch.nn.Sequential(
            torch.nn.Conv2d(filters * 2 + in_channels, filters, kernel_size, padding='same'),
            activation(filters)
        )

        self.conv_concat_reduced = torch.nn.Sequential(
            torch.nn.Conv2d(filters * 2, filters, kernel_size, padding='same'),
            activation(filters)
        )

        if stride > 1:
            if downscale:
                # apply downsampling using maxpooling and keep the resolution size
                self.pool_main = torch.nn.MaxPool2d(3, stride=stride, padding=1)
                self.pool_main = torch.nn.MaxPool2d(3, stride=stride, padding=1)
                self.pool_flash = torch.nn.MaxPool2d(3, stride=stride, padding=1)
                self.pool_mix = torch.nn.MaxPool2d(3, stride=stride, padding=1)
            else:
                self.pool_main = torch.nn.Upsample(scale_factor=stride, mode='nearest')
                self.pool_flash = torch.nn.Upsample(scale_factor=stride, mode='nearest')
                self.pool_mix = torch.nn.Upsample(scale_factor=stride, mode='nearest')
        else:
            self.pool_main = torch.nn.Identity()
            self.pool_flash = torch.nn.Identity()
            self.pool_mix = torch.nn.Identity()

    def forward(self, x):
        main, flash, mix = x
        main = self.conv_main(main)
        flash = self.conv_flash(flash)

        new_mix = [main, flash]
        if mix is not None:
            new_mix.append(mix)
            mix = self.conv_concat(torch.cat(new_mix, dim=1))
        else:
            mix = self.conv_concat_reduced(torch.cat(new_mix, dim=1))

        # sum up main, flash and mix on axis 1
        #mix = torch.sum([main, flash, mix], dim=1)
        #mix = torch.add(torch.add(main, flash, axis=1), mix, axis=1)
        mix = main + flash + mix

        main = self.pool_main(main)
        flash = self.pool_flash(flash)
        mix = self.pool_mix(mix)
        return main, flash, mix
