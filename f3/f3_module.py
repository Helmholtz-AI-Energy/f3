import math
import numbers

import numpy as np
import torch
import torch.nn as nn


class F3Connection(torch.autograd.Function):
    """
    An autograd function implementing the F³ computation by detaching the input and replacing the gradient inbetween
    module blocks.
    """

    @staticmethod
    def forward(ctx, input, f3_gradient):
        """
        Receives an input and it's F³ gradient. Saves the gradient and returns the detached input (inplace).
        :param ctx: A context to store tensors and variables for the backward pass. Used to store the F³ gradient.
        :param input: The input to the function.
        :param f3_gradient: The precomputed F³ gradient, will be returned as gradient in backward.
        :return: The input but detached.
        """
        if f3_gradient is not None:
            ctx.save_for_backward(f3_gradient.view(input.shape))
        return input.detach()

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Returns the precomputed F³ gradient passed to forward as gradient of the input.
        :param ctx: A context to store tensors and variables for the backward pass. Used to load the F³ gradient.
        :param grad_output: The gradient of the output of this function, unused in F³.
        :return: The F³ gradient passed to the forward function.
        """
        f3_gradient, = ctx.saved_tensors
        return f3_gradient, None


class F3Connector(nn.Module):
    """
    A module to detach the input and replace the gradient inbetween module blocks.
    Holds the fixed feedback weights B as a buffer and computes the F³ gradient by multiplying the error information
    with these feedback weights. In training mode, the F³ gradient is computed in the forward pass and saved for the
    backward pass using the F3Connection autograd function. In eval mode, the connector has essentially no effect.
    """

    def __init__(self, model_output_size):
        """
        Create a connector module.
        :param model_output_size: Output size of the model, e.g. number of classes.
        """
        super(F3Connector, self).__init__()
        self.model_output_size = model_output_size

    @staticmethod
    def __initialize_fill_with_repeat(tensor, fill):
        assert fill.shape[0] == tensor.shape[0]
        repetitions = tensor.shape[1] // fill.shape[1]
        remainder = tensor.shape[1] % fill.shape[1]
        if remainder:
            tensor[:, :-remainder] = fill.repeat(1, repetitions)
            tensor[:, -remainder:] = fill[:, :remainder]
        else:
            tensor = fill.repeat(1, repetitions)
        return tensor

    @staticmethod
    def initialize_values(tensor, initialization_method='kaiming_uniform', scalar=None, discrete_values=None):
        """
        (Re-)Initialize the given tensor using the specified initialization method.
        Default uses kaiming uniform distribution (like nn.Linear).
        :param tensor: The tensor to initialize.
        :param initialization_method: The initialization method to be used.
        :param scalar: The scalar value to initialize the tensor with when using initialization method based on
        a constant initialization (potentially with some additional mutations) e.g. 'constant', 'alternate_negative',
        'chunked_negative', 'discrete_uniform'.
        :param discrete_values: The scaling factors to scale the constant value with. Sampled with uniform distribution
        for each value in the tensor.
        :param discrete_values: The scaling factors to scale the constant value with. Sampled with uniform distribution
        for each value in the tensor.
        :returns The initialized tensor.
        """
        const_based_methods = ['constant', 'alternate_negative', 'chunked_negative', 'discrete_uniform',
                               'cartesian_product']
        model_output_size = tensor.shape[0]
        scalar_values = {'ones': 1, 'zeros': 0, '1/C': 1 / model_output_size}

        if initialization_method in const_based_methods:
            assert scalar is not None
            assert isinstance(scalar, (numbers.Number, str))
            if isinstance(scalar, str):
                try:
                    scalar = scalar_values[scalar]
                except KeyError:
                    raise ValueError(f'Invalid scalar name {scalar}, valid names include: {scalar_values.keys()}')

            torch.nn.init.constant_(tensor, scalar)

            if initialization_method == 'alternate_negative':
                tensor[:, ::2] *= -1
            elif initialization_method == 'chunked_negative':
                tensor[:, tensor.shape[1] // 2:] *= -1
            elif initialization_method == 'discrete_uniform':
                assert discrete_values is not None
                discrete_values = torch.as_tensor(discrete_values)
                indices = torch.randint_like(tensor, 0, len(discrete_values), dtype=int)
                tensor *= discrete_values[indices]
            elif initialization_method == 'cartesian_product':
                assert discrete_values is not None
                if not tensor.shape[1] == len(discrete_values) ** model_output_size:
                    raise ValueError(f'Incorrect width {tensor.shape[1]} but should be '
                                     f'{len(discrete_values) ** model_output_size}. When using '
                                     f'{initialization_method=} the width of the hidden layers must be '
                                     f'len(discrete_values) ** model_output_size.')
                discrete_values = torch.as_tensor(discrete_values)
                permutations = torch.cartesian_prod(*([discrete_values] * model_output_size)).T
                tensor *= permutations

        elif initialization_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        elif initialization_method == 'kaiming_uniform_repeat_line':
            column = torch.empty((model_output_size, 1))
            nn.init.kaiming_uniform_(column, a=math.sqrt(5))
            tensor = F3Connector.__initialize_fill_with_repeat(tensor, column)
        elif initialization_method == 'identity_fill_zero':
            torch.nn.init.eye_(tensor)
        elif initialization_method == 'identity_repeat':
            eye = torch.eye(model_output_size)
            tensor = F3Connector.__initialize_fill_with_repeat(tensor, eye)
        elif initialization_method == 'identity_repeat_pm':
            pm_eye = torch.eye(model_output_size).repeat(1, 2)
            pm_eye[:, model_output_size:] *= -1
            tensor = F3Connector.__initialize_fill_with_repeat(tensor, pm_eye)
        else:
            raise ValueError(f'Invalid initialization method {initialization_method}.')
        return tensor

    def reset_feedback_weights(self, initialization_method='kaiming_uniform', scalar=None, discrete_values=None):
        raise NotImplementedError('Please use a sub-class of F3Connector which implements the gradient approximation.')

    def compute_f3_gradient(self, error_information, input):
        raise NotImplementedError('Please use a sub-class of F3Connector which implements the gradient approximation.')

    def forward(self, input, error_information):
        """
        Compute the F³ gradient based on the error information and passes both to the F3Connection function, which
        detaches the input and saves the F³ gradient to be returned by the backward function.
        :param input: The input to the layer (i.e. an intermediate result returned by the previous block and passed to
        the next block after the connector).
        :param error_information: The error information of the current batch.
        :return: The detached input.
        """
        f3_gradient = None
        if self.training:
            f3_gradient = self.compute_f3_gradient(error_information, input)
        return F3Connection.apply(input, f3_gradient)


class F3ConnectorFC(F3Connector):
    """
    A module to detach the input and replace the gradient inbetween module blocks.
    Holds the fixed feedback weights B as a buffer and computes the F³ gradient by multiplying the error information
    with these feedback weights. In training mode, the F³ gradient is computed in the forward pass and saved for the
    backward pass using the F3Connection autograd function. In eval mode, the connector has essentially no effect.
    """

    def __init__(self, input_size, model_output_size, device=None, dtype=None, initialization_method='kaiming_uniform',
                 scalar=None, discrete_values=None):
        """
        Create a connector module by initializing the feedback weights of size model_output_size x np.prod(input_size).
        :param input_size: Output size of the previous layer (will be flattened using np.prod).
        :param model_output_size: Output size of the model, e.g. number of classes.
        :param device: The device to create the feedback weights on.
        :param dtype: The dtype of the feedback weights.
        :param initialization_method: The initialization method for the feedback weights, default is kaiming_uniform,
        see reset_feedback_weights for other options.
        :param scalar: Additional parameters to the initialization method.
        :param discrete_values: Additional parameters to the initialization method.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(F3ConnectorFC, self).__init__(model_output_size)
        feedback_weight = torch.empty(torch.Size([model_output_size, np.prod(input_size)]), **factory_kwargs)
        self.register_buffer('feedback_weight', feedback_weight)
        self.reset_feedback_weights(initialization_method, scalar, discrete_values)

    def reset_feedback_weights(self, initialization_method='kaiming_uniform', scalar=None, discrete_values=None):
        self.feedback_weight = self.initialize_values(self.feedback_weight, initialization_method, scalar,
                                                      discrete_values)

    def compute_f3_gradient(self, error_information, input):
        """
        Compute the F³ gradient by multiplying the feedback weights with the error information.
        :param error_information: The error information of the current batch.
        :param input: Input to the connector, i.e. output of the previous layers forward pass.
        :return: The F³ gradient approximation.
        """
        return error_information.mm(self.feedback_weight)


class F3Module(nn.Module):
    """
    An F³ module taking a list of (sequential) submodules and building an F³ model by interspersing these blocks with
    F³ connectors, disconnecting the autograd graph and substituting the gradients with approximate gradients based on
    fixed random feedback weights and additional error information passed to the forward pass.
    The submodule blocks are treated as a black box and could be anything from single layers to complex modules.
    """

    def __init__(self, blocks, block_output_sizes, connector_types=None, **kwargs):
        """
        Create an F³ module, interspersing the given blocks with F³ connectors.
        :param blocks: The submodules composing this module (in sequential order).
        :param block_output_sizes: The output sizes of the blocks (int or iterable), used to determine the size of the
        feedback weights.
        :param connector_types: List of F3Connector subclass and corresponding kwargs to the init method. Contains n-1
        such tuples for n blocks.
        """
        super(F3Module, self).__init__()
        self.blocks = blocks

        # create connector modules inbetween all blocks
        model_output_size = block_output_sizes[-1]
        connector_types = [(F3ConnectorFC, {})] * (
                    len(block_output_sizes) - 1) if connector_types is None else connector_types
        connector_types = [(F3ConnectorFC, {}) if connector_type is None else connector_type
                           for connector_type in connector_types]
        self.connectors = [
            connector(in_features, model_output_size, **specific_kwargs, **kwargs)
            for (connector, specific_kwargs), in_features in zip(connector_types, block_output_sizes[:-1])]

        # create interspersed list of blocks and connectors
        # initialize list of correct size with dummy module
        self.interspersed_blocks = [nn.Module()] * (len(blocks) + len(self.connectors))
        self.interspersed_blocks[0::2] = blocks  # every second entry is a block, starting from index 0
        self.interspersed_blocks[1::2] = self.connectors  # every other entry is a connector, starting from index 1

        self.layers = nn.ModuleList(self.interspersed_blocks)

    def forward(self, x, error_information):
        """
        Pass the input x forward through the blocks, detaching the intermediate results inbetween all blocks and
        precomputing the F³ gradients (when in training mode) using the F³ connectors.
        :param x: The input x to pass through the network.
        :param error_information: The error information used to compute the F³ gradients, e.g. the delayed error from
        the previous epoch, the targets y,...
        :return: The output of model(x).
        """
        for i, layer in enumerate(self.layers):
            is_connector = (i % 2 == 1)  # the current layer is a connector if i is odd
            x = layer(x, error_information) if is_connector else layer(x)
        return x

    @staticmethod
    def get_connector_by_name(name):
        connectors = {'fc': F3ConnectorFC, }
        if name not in connectors:
            raise ValueError(f'Invalid connector {name}. Valid options are: {connectors.keys()}')
        return connectors[name]


class BPModule(nn.Module):
    """Same usage as the F³ module but uses standard BP for training."""

    def __init__(self, blocks, block_output_sizes=None):
        super(BPModule, self).__init__()
        self.blocks = blocks
        self.layers = nn.ModuleList(self.blocks)

    def forward(self, x, error_information=None):
        for layer in self.layers:
            x = layer(x)
        return x
