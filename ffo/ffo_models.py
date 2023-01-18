import numpy as np
import torch.nn as nn

from ffo.ffo_module import FFOModule, BPModule


def create_fc_block(in_features, out_features, flatten=False, activation=nn.Tanh):
    block = [nn.Linear(in_features, out_features)]
    if activation is not None:
        block.append(activation())
    if flatten:
        block = [nn.Flatten(), *block]
    return nn.Sequential(*block)


def get_activation(block_id, num_blocks, regression):
    is_last_block = (block_id == num_blocks - 2)
    if is_last_block:
        return None if regression else nn.Sigmoid
    else:
        return nn.Tanh


def build_module(mode, blocks, block_output_sizes, connector_types=None, **kwargs):
    if mode == 'ffo':
        return FFOModule(blocks, block_output_sizes, connector_types, **kwargs)
    elif mode == 'bp':
        return BPModule(blocks)
    elif mode == 'llo':
        model = BPModule(blocks)
        for block in model.blocks[:-1]:  # freeze all but the last block
            for param in block.parameters():
                param.requires_grad = False
        return model
    else:
        raise ValueError(f'Invalid mode {mode}.')


def fc_model(depth, width, input_size, output_size=10, mode='ffo', regression=False, **kwargs):
    sizes = [np.prod(input_size).item()] + [width] * depth + [output_size]

    blocks = [create_fc_block(in_features, out_features, flatten=(i == 0),
                              activation=get_activation(i, len(sizes), regression))
              for i, (in_features, out_features) in enumerate(zip(sizes, sizes[1:]))]

    return build_module(mode, blocks, sizes[1:], **kwargs)


def fc1_500(input_size, output_size=10, mode='ffo', regression=False, **kwargs):
    return fc_model(1, 500, input_size, output_size, mode, regression, **kwargs)


def fc2_500(input_size, output_size=10, mode='ffo', regression=False, **kwargs):
    return fc_model(2, 500, input_size, output_size, mode, regression, **kwargs)


def fc1_1000(input_size, output_size=10, mode='ffo', regression=False, **kwargs):
    return fc_model(1, 1000, input_size, output_size, mode, regression, **kwargs)


def fc2_1000(input_size, output_size=10, mode='ffo', regression=False, **kwargs):
    return fc_model(2, 1000, input_size, output_size, mode, regression, **kwargs)
