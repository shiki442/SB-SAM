import torch
from torch import nn
import functools
import math
from model.utils import register_model
from model import layers, layerspp
from model.nn import L_out

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
Combine = layerspp.Combine
conv3x3 = layers.ddpm_conv3x3
conv1x1 = layers.ddpm_conv1x1
get_act = layers.get_act

default_initializer = layers.default_init


@register_model(name='NCSNpp')
class NCSNpp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.act = act = get_act(cfg)
        self.nf = nf = cfg.model.nf
        self.cond_dim = cfg.model.cond_dim
        ch_mult = cfg.model.ch_mult
        dropout = cfg.model.dropout
        resamp_with_conv = cfg.model.resamp_with_conv
        self.num_res_blocks = num_res_blocks = cfg.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = cfg.model.attn_resolutions

        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            cfg.data.n // (2 ** i) for i in range(num_resolutions)]

        self.condition_on_t = condition_on_t = cfg.model.conditional  # noise-conditional
        fir = cfg.model.fir
        fir_kernel = cfg.model.fir_kernel
        # self.skip_rescale = skip_rescale = cfg.model.skip_rescale
        self.skip_rescale = skip_rescale = True

        self.resblock_type = resblock_type = cfg.model.resblock_type.lower()
        self.progressive = progressive = cfg.model.progressive.lower()
        self.progressive_input = progressive_input = cfg.model.progressive_input.lower()
        self.embedding_type = embedding_type = cfg.model.embedding_type.lower()
        combine_method = cfg.model.progressive_combine.lower()

        init_scale = cfg.model.init_scale

        modules = []
        out_padding = []


        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            assert cfg.training.continuous, "Fourier features are only used for continuous training."

            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=cfg.model.fourier_scale
            ))
            embed_dim = 2 * nf
        
        if condition_on_t:
            modules.append(layerspp.Conditioner(in_dim=embed_dim, embed_dim=nf * 4, act=act, cond_drop_prob=0.0))

        if self.cond_dim > 0:
            modules.append(layerspp.Conditioner(in_dim=self.cond_dim, embed_dim=nf * 4, act=act, batch_norm=True))

        Combiner = functools.partial(Combine, method=combine_method)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv,
                                     fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            # self.pyramid_upsample = layerspp.Upsample(fir=fir,
            #                                           fir_kernel=fir_kernel,
            #                                           with_conv=False)
            Pyramid_Upsample = functools.partial(layerspp.Upsample,
                                                 with_conv=False,
                                                 fir=fir, fir_kernel=fir_kernel)

        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv,
                                       fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir,
                                                          fir_kernel=fir_kernel,
                                                          with_conv=False)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4)

        channels = cfg.data.d
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        hs_f = [cfg.data.n]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                # 添加自注意力模块
                # if all_resolutions[i_level] in attn_resolutions:
                #     modules.append(AttnBlock(channels=in_ch))

                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    pass  # biggan结构
                    # modules.append(ResnetBlock(down=True, in_ch=in_ch))
                hs_f.append(L_out(modules[-1], hs_f[-1]))

                if progressive_input == 'input_skip':
                    modules.append(Combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    # if combine_method == 'cat':
                    #     in_ch *= 2

                elif progressive_input == 'residual':
                    pass
                #     modules.append(pyramid_downsample(
                #         in_ch=input_pyramid_ch, out_ch=in_ch))
                #     input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):

            check_L_out = 0 if L_out(
                layerspp.Upsample(), hs_f[i_level]) == hs_f[i_level-1] else 1
            out_padding.append(check_L_out)

            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            # 自注意力层
            # if all_resolutions[i_level] in attn_resolutions:
            #     modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(
                            conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        pass
                        # modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                        #                             num_channels=in_ch, eps=1e-6))
                        # modules.append(conv3x3(in_ch, in_ch, bias=True))
                        # pyramid_ch = in_ch

                else:
                    if progressive == 'output_skip':
                        modules.append(Pyramid_Upsample(in_ch=pyramid_ch,
                                                        out_padding=out_padding[-2]))
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        pass
                        # modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        # pyramid_ch = in_ch

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(
                        Upsample(in_ch=in_ch, out_padding=out_padding[-1]))
                else:
                    pass
                    # modules.append(ResnetBlock(in_ch=in_ch, up=True))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, t, cond=None):
        # Obtain the Gaussian random feature embedding for t
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        if self.condition_on_t:
            temb = modules[m_idx](temb)
            m_idx += 1
        else:
            temb = None

        if cond is not None and self.cond_dim > 0:
            temb = temb + modules[m_idx](cond)
            m_idx += 1
        elif self.cond_dim > 0:
            # If no condition is provided, we don't need to pass through the conditioner
            m_idx += 1

        hs = [modules[m_idx](x)]
        m_idx += 1
        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            # 残差层
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # if h.shape[-1] in self.attn_resolutions:
                #     h = modules[m_idx](h)
                #     m_idx += 1

                hs.append(h)

            # DownSample层
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    pass

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                else:
                    pass
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            # if h.shape[-1] in self.attn_resolutions:
            #     h = modules[m_idx](h)
            #     m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pass
                else:
                    if self.progressive == 'output_skip':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pass

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    pass

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid

        assert m_idx == len(modules)
        # if self.cfg.model.scale_by_sigma:
        #     used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
        #     h = h / used_sigmas

        return h
