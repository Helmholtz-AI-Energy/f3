import torch
import torch.nn.functional as F
import torch.utils.benchmark
import torch.utils.data
import torchmetrics
from tqdm.autonotebook import tqdm

from ffo import datasets


def train_batch(model, optimizer, index, data, target, loss_function, delayed_error_info=None,
                error_info_type='one_hot_target'):
    # get error information
    if error_info_type == 'one_hot_target':
        error_information = target
    elif error_info_type == 'current_error':
        # for DFA: run extra forward pass to get current error, this is less efficient than actual DFA since it leads to
        # an unnecessary second forward pass but results in the same weight updates
        model.eval()
        output = model(data, None)
        error_information = (target - output).detach_()
    elif error_info_type in ['delayed_loss', 'delayed_error', 'delayed_loss_softmax', 'delayed_error_softmax']:
        error_information = delayed_error_info[index]
    elif error_info_type in ['delayed_loss_one_hot', 'delayed_error_one_hot']:
        error_information = target * delayed_error_info[index]
    elif error_info_type == 'zeros':
        error_information = torch.zeros_like(target)
    else:
        raise ValueError(f"Invalid error info: {error_info_type}")

    model.train()
    optimizer.zero_grad()

    output = model(data, error_information)
    if error_info_type.startswith('delayed_loss'):  # retain output grad to update the delayed_error_info
        output.retain_grad()
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()

    # update error information
    if error_info_type == 'delayed_loss_softmax':
        optimizer.zero_grad()
        detached_output = output.detach()
        detached_output.requires_grad = True
        loss_after_softmax = loss_function(F.softmax(detached_output, dim=1), target)
        loss_after_softmax.backward()
        delayed_error_info[index] = -detached_output.grad.detach_()
    elif error_info_type == 'delayed_error_softmax':
        delayed_error_info[index] = (target - F.softmax(output, dim=1)).detach_()
    elif error_info_type.startswith('delayed_loss'):
        delayed_error_info[index] = -output.grad.detach_()
    elif error_info_type.startswith('delayed_error'):
        delayed_error_info[index] = (target - output).detach_()


def log_results(**kwargs):
    print(', '.join([f'{key}: {value}' for key, value in kwargs.items()]))


def evaluate_dataset(model, device, data_loader, dataset_config, dataset_prefix='test'):
    model.eval()
    loss = 0
    correct = 0
    total = len(data_loader.dataset)
    loss_function = dataset_config['loss']
    secondary_metrics = {}

    if dataset_config['is_regression']:
        secondary_metrics['mape'] = (torchmetrics.MeanAbsolutePercentageError(), lambda x: x)
        secondary_metrics['mae'] = (torchmetrics.MeanAbsoluteError(), lambda x: x)
    else:
        num_classes = dataset_config['output_size']
        if dataset_config['classification_target'] == "one_hot":
            # noinspection PyTypedDict
            secondary_metrics['AUROC'] = (torchmetrics.classification.MultilabelAUROC(num_labels=num_classes),
                                          lambda x: x.int())
            secondary_metrics['MultilabelF1'] = (torchmetrics.classification.MultilabelF1Score(num_labels=num_classes),
                                                 lambda x: x.int())
        elif dataset_config['classification_target'] == "class":
            secondary_metrics['AUROC'] = (torchmetrics.AUROC(num_labels=num_classes), lambda x: x.int())
            secondary_metrics['F1'] = (torchmetrics.F1Score(num_classes=num_classes), lambda x: x.int())
        else:
            raise ValueError(f'Invalid classification_target: {dataset_config["classification_target"]}')
    for metric, _ in secondary_metrics.values():
        metric.to(device)

    with torch.no_grad():
        for batch in data_loader:
            data, target = batch[-2:]  # take last two elements to work for both the train and the test loader
            data, target = data.to(device), target.to(device)
            output = model(data, None)
            loss += loss_function(output, target, reduction='sum').item()  # sum up batch loss
            for metric, target_transform in secondary_metrics.values():
                metric.update(output, target_transform(target))
            if dataset_config['classification_target'] == "one_hot":
                actual = target.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(actual).sum().item()
            elif dataset_config['classification_target'] == "class":
                correct += output.eq(target).sum().item()
    loss /= total
    accuracy = 100. * correct / total

    eval_results = {f'{dataset_prefix}_loss': loss, 'loss_name': loss_function.__name__,
                    **{f'{dataset_prefix}_{name}': metric.compute().item()
                       for name, (metric, _) in secondary_metrics.items()}}
    if dataset_config['classification_target'] is not None:
        eval_results = {**eval_results, f'{dataset_prefix}_correct': correct, f'{dataset_prefix}_accuracy': accuracy}

    return eval_results


def evaluate(model, device, data_loader, epoch, processed_samples, dataset_config, dataset_prefix, print_results=False):
    evaluation_results = evaluate_dataset(model, device, data_loader, dataset_config, dataset_prefix)
    evaluation_results = {'epoch': epoch, 'processed_samples': processed_samples, **evaluation_results}
    if print_results:
        log_results(**evaluation_results)
    return evaluation_results


def training(model, dataset, device, epochs, batch_size, test_batch_size=1000, lr=1.0, dry_run=False, log_interval=0,
             print_eval=False, eval_every_nth_batch=0, error_info_type='one_hot_target', root='data'):
    device_kwargs = {'num_workers': 1, 'pin_memory': True} if 'cuda' in device else {}
    device = torch.device(device)

    train_loader, train_test_loader, test_loader = datasets.get_data_loaders(dataset, batch_size, test_batch_size,
                                                                             root=root, **device_kwargs)
    dataset_config = datasets.get_dataset_config(dataset)
    loss = dataset_config['loss']

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    error_information = None
    if error_info_type.startswith('delayed_loss') or error_info_type.startswith('delayed_error'):
        train_data = train_loader.dataset.dataset
        error_information = torch.empty(torch.Size([len(train_data), *train_data[0][1].shape]))
        for i in range(len(train_data)):
            error_information[i] = train_data[i][1]
        error_information = error_information.to(device)

    results = []
    processed_samples = 0
    results.append(evaluate(model, device, train_test_loader, 0, processed_samples, dataset_config, 'train',
                            print_eval))
    results.append(evaluate(model, device, test_loader, 0, processed_samples, dataset_config, 'test', print_eval))
    for epoch in tqdm(range(1, epochs + 1)):

        for batch_idx, (sample_index, data, target) in enumerate(train_loader):
            if eval_every_nth_batch and batch_idx % eval_every_nth_batch == 0 and not batch_idx == 0:
                print_results = log_interval and (batch_idx % log_interval == 0)
                results.append(evaluate(model, device, test_loader, epoch, processed_samples, dataset_config, 'test',
                                        print_eval))

            sample_index, data, target = sample_index.to(device), data.to(device), target.to(device)
            train_batch(model, optimizer, sample_index, data, target, loss, error_information, error_info_type)
            processed_samples += len(data)

            if dry_run:
                return

        results.append(evaluate(model, device, train_test_loader, epoch, processed_samples, dataset_config, 'train',
                                print_eval))
        results.append(evaluate(model, device, test_loader, epoch, processed_samples, dataset_config, 'test',
                                print_eval))

    return results
