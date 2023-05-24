import sys

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

import torchvision

import einops
import einops.layers.torch

from f3 import f3_module, f3_models, datasets


class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., activation=nn.GELU):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.layers = nn.Sequential(
            self.linear1,
            activation(),
            nn.Dropout(dropout),
            self.linear2,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., norm=True, mlp_activation=nn.GELU):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim) if norm else nn.Identity()
        self.attention = AttentionBlock(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(dim) if norm else nn.Identity()
        self.mlp = MLPBlock(dim, mlp_dim, dropout=dropout, activation=mlp_activation)

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

    def __repr__(self):
        return f'Reshape({self.shape})'


class ToPatches(nn.Module):
    def __init__(self, patch_height, patch_width):
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.split_into_patches = einops.layers.torch.Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)

    def forward(self, x):
        x = self.split_into_patches(x)
        return x

    def __repr__(self):
        return f'ToPatches(b c (h p1) (w p2) -> b (h w) (p1 p2 c)) with p1={self.patch_height}, p2={self.patch_width}'


class F3VisionTransformer(nn.Module):
    def __init__(self, depth, image_size=28, channels=1, num_classes=10, hidden_dim=512, heads=1, dim_head=64,
                 mlp_dim=128, dropout=0., mode='f3', block_norm=True, activation=nn.Tanh, patch_size=7):
        super().__init__()

        self.sizes = [image_size ** 2 * channels] + [hidden_dim] * depth + [num_classes]

        num_tokens = int((image_size // patch_size) ** 2)
        token_dim = int(hidden_dim // num_tokens)
        transformer_block_kwargs = {'heads': heads, 'dim_head': dim_head, 'mlp_dim': mlp_dim, 'dropout': dropout,
                                    'norm': block_norm, 'mlp_activation': activation}

        input_embedding = nn.Sequential(ToPatches(patch_size, patch_size), nn.Flatten(),
                                        nn.Linear(self.sizes[0], hidden_dim))
        transformer_blocks = [nn.Sequential(Reshape((-1, int(hidden_dim // token_dim), token_dim)),
                                            TransformerBlock(token_dim, **transformer_block_kwargs),
                                            Reshape((-1, hidden_dim))) for _ in range(depth - 1)]
        mlp_head = [
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(p=dropout), activation()),
            nn.Sequential(nn.Linear(hidden_dim, num_classes))
        ]
        self.blocks = [nn.Sequential(input_embedding, transformer_blocks[0]), *transformer_blocks[1:], *mlp_head]

        connector_types = [(f3_module.F3ConnectorFC, {'initialization_method': 'discrete_uniform', 'scalar': 1,
                                                      'discrete_values': [-1, 0, 1]})] * len(self.blocks)

        self.model = f3_models.build_module(mode, self.blocks, self.sizes[1:], connector_types)

    def forward(self, x, error_information=None):
        return self.model(x, error_information)

    def __str__(self):
        return str(self.model)


def get_dataloaders(data_path='data', batch_size=100, batch_size_test=10000, download=True):
    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    num_classes = 10
    one_hot_transform = torchvision.transforms.Compose([lambda target: F.one_hot(torch.tensor(target),
                                                                                 num_classes).float()])
    train_set = torchvision.datasets.MNIST(data_path, train=True, download=download, transform=transform_mnist,
                                           target_transform=one_hot_transform)
    train_set = datasets.DatasetWithIndices(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.MNIST(data_path, train=False, download=download, transform=transform_mnist,
                                          target_transform=one_hot_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader


def evaluate(epoch, model, data_loader, device, loss_function):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = loss_function(output, target, reduction='sum')
            total_loss += loss.item()

            actual = target.argmax(dim=1, keepdim=True)
            prediction = torch.softmax(output, dim=1).argmax(dim=1, keepdim=True)
            correct_samples += prediction.eq(actual).sum()

    avg_loss = total_loss / total_samples
    print(f'Epoch {epoch}: Average test loss: {avg_loss:.4f}  Accuracy:{correct_samples:5}/{total_samples:5} '
          f'({100.0 * correct_samples / total_samples:4.2f}%)')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_error_information(model, index, data, target, delayed_error_info=None, error_info_type='delayed_error'):
    # get error information
    if error_info_type == 'one_hot_target':
        error_information = target
    elif error_info_type == 'current_error':
        # for DFA: run extra forward pass to get current error, this is less efficient than actual DFA since it leads to
        # an unnecessary second forward pass but results in the same weight updates
        model.eval()
        output = model(data, None)
        error_information = (target - F.softmax(output, dim=1)).detach_()
    elif error_info_type in ['delayed_loss', 'delayed_error', 'delayed_loss_softmax', 'delayed_error_softmax',
                             'delayed_error_sigmoid']:
        error_information = delayed_error_info[index]
    elif error_info_type in ['delayed_loss_one_hot', 'delayed_error_one_hot']:
        error_information = target * delayed_error_info[index]
    elif error_info_type == 'zeros':
        error_information = torch.zeros_like(target)
    else:
        raise ValueError(f"Invalid error info: {error_info_type}")
    return error_information


def update_error_information(index, target, output, optimizer, loss_function, delayed_error_info, error_info_type):
    if error_info_type == 'delayed_loss_softmax':
        optimizer.zero_grad()
        detached_output = output.detach()
        detached_output.requires_grad = True
        loss_after_softmax = loss_function(F.softmax(detached_output, dim=1), target)
        loss_after_softmax.backward()
        delayed_error_info[index] = -detached_output.grad.detach_()
    elif error_info_type == 'delayed_error_softmax':
        delayed_error_info[index] = (target - F.softmax(output, dim=1)).detach_()
    elif error_info_type == 'delayed_error_sigmoid':
        delayed_error_info[index] = (target - F.sigmoid(output)).detach_()
    elif error_info_type.startswith('delayed_loss'):
        delayed_error_info[index] = -output.grad.detach_()
    elif error_info_type.startswith('delayed_error'):
        delayed_error_info[index] = (target - output).detach_()


def initialize_error_information(train_loader, device):
    train_data = train_loader.dataset.dataset
    num_samples = len(train_data)
    error_information = torch.empty(torch.Size([num_samples, *train_data[0][1].shape]))
    for i in range(num_samples):
        error_information[i] = train_data[i][1]
    return error_information.to(device)


def train_batch(model, optimizer, index, data, target, loss_function, delayed_error_info=None,
                error_info_type='delayed_error'):
    error_information = get_error_information(model, index, data, target, delayed_error_info, error_info_type)

    model.train()
    optimizer.zero_grad()
    output = model(data, error_information)
    if error_info_type.startswith('delayed_loss'):  # retain output grad to update the delayed_error_info
        output.retain_grad()
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()

    update_error_information(index, target, output, optimizer, loss_function, delayed_error_info, error_info_type)


def train_epoch(model, optimizer, data_loader, device, loss_function, delayed_error_info=None,
                error_info_type='delayed_error'):
    for i, (index, data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        train_batch(model, optimizer, index, data, target, loss_function, delayed_error_info, error_info_type)


def train_mnist(data_path='data', epochs=25, batch_size=100, batch_size_test=10000, device=None, mode='f3',
                error_info_type='delayed_error', lr=0.003, seed=0, **model_kwargs):
    set_seed(seed)

    model_default_kwargs = {'image_size': 28, 'num_classes': 10, 'channels': 1, 'hidden_dim': 512, 'heads': 8,
                            'depth': 5, 'mode': mode}
    model = F3VisionTransformer(**{**model_default_kwargs, **model_kwargs})
    model.to(device)

    def loss_function(output, target, **kwargs):
        return F.nll_loss(F.log_softmax(output, dim=1), target.argmax(dim=1), **kwargs)

    train_loader, test_loader = get_dataloaders(data_path, batch_size, batch_size_test)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    error_information = initialize_error_information(train_loader,
                                                     device) if error_info_type.startswith('delayed_') else None

    evaluate(0, model, test_loader, device, loss_function)
    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, train_loader, device, loss_function, delayed_error_info=error_information,
                    error_info_type=error_info_type)
        evaluate(epoch, model, test_loader, device, loss_function)


if __name__ == '__main__':
    seed = sys.argv[1] if len(sys.argv) > 1 else 0

    shared_kwargs = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 100
    }

    kwargs_list = [
        ('bp', {'mode': 'bp'}),
        ('drtp', {'mode': 'f3', 'error_info_type': 'one_hot_target'}),
        ('dfa', {'mode': 'f3', 'error_info_type': 'current_error'}),
        ('f3', {'mode': 'f3', 'error_info_type': 'delayed_error_softmax'}),
    ]
    for label, kwargs in kwargs_list:
        print(label)
        train_mnist(seed=seed, **{**shared_kwargs, **kwargs})
