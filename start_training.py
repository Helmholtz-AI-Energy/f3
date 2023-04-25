import argparse
import datetime
import pathlib
import uuid

import pandas
import torch

from f3 import training, f3_models, datasets


def create_model(depth, width, input_size, output_size, mode, initialization_method, scalar, discrete_values,
                 regression):
    return f3_models.fc_model(depth=depth, width=width, input_size=input_size, output_size=output_size, mode=mode,
                              regression=regression, initialization_method=initialization_method, scalar=scalar,
                              discrete_values=discrete_values)


def train_model(model, dataset, device, epochs, batch_size, test_batch_size, lr, dry_run, log_interval, print_eval,
                eval_every_nth_batch, error_info):
    return training.training(model=model, dataset=dataset, device=device, epochs=epochs, batch_size=batch_size,
                             test_batch_size=test_batch_size, lr=lr, dry_run=dry_run, log_interval=log_interval,
                             print_eval=print_eval, eval_every_nth_batch=eval_every_nth_batch,
                             error_info_type=error_info)


def init_method_to_string(initialization_method, scalar, discrete_values, **ignored):
    uses_scalar = ['constant', 'alternate_negative', 'chunked_negative', 'discrete_uniform', 'cartesian_product']
    uses_values = ['discrete_uniform', 'cartesian_product']
    tokens = [initialization_method]
    if initialization_method in uses_scalar:
        tokens.append(scalar)
    if initialization_method in uses_values:
        tokens.append(discrete_values)
    return '_'.join(str(token) for token in tokens)


def postprocess_results(result_list, configuration):
    dataframe = pandas.DataFrame(result_list)
    for key, value in vars(configuration).items():
        dataframe[key] = str(value)

    dataframe['init_method'] = init_method_to_string(**vars(configuration))
    return dataframe


def save_results(dataframe, output_name, output_path):
    today = datetime.datetime.today()
    path = pathlib.Path(output_path) / str(today.year) / f'{today.year}-{today.month}' / str(today.date())
    filename = f"{today.strftime('%Y-%m-%d--%H-%M-%S')}-{output_name}-{str(uuid.uuid4())[:8]}.csv"
    path.mkdir(parents=True, exist_ok=True)
    print(f'Saving results to {(path / filename).absolute()}')
    dataframe['result_filename'] = filename
    dataframe.to_csv(path / filename, index=False)


def parse_args(**kwargs):
    parser = argparse.ArgumentParser()

    # model arguments: depth, width, f3, initialization_method, scalar, discrete_values
    parser.add_argument("--depth", type=int, help="The model depth")
    parser.add_argument("--width", type=int, help="The model width = number of neurons in the FC layers")
    parser.add_argument("--model", type=str, help="The model name, alternative to specifying depth and width.",
                        choices=["fc1_500", "fc2_500", "fc1_1000", "fc2_1000"])
    parser.add_argument("--mode", type=str, default="f3", choices=["f3", "bp", "llo"], help="The training mode.")
    parser.add_argument("--initialization_method", type=str, default='kaiming_uniform',
                        choices=['constant', 'alternate_negative', 'chunked_negative', 'discrete_uniform',
                                 'cartesian_product', 'kaiming_uniform', 'kaiming_uniform_repeat_line',
                                 'identity_fill_zero', 'identity_repeat', 'identity_repeat_pm'],
                        help="The initialization method for the feedback weights")
    parser.add_argument("--scalar", type=float, default=1.0,
                        help="Additional parameter to the initialization method, only required for some methods")
    parser.add_argument("--discrete_values", type=float, default=[-1, 0, 1], nargs='+',
                        help="Additional parameter to the initialization method, only required for some methods")

    # training arguments: device, epochs, batch_size, test_batch_size, lr, dry_run, seed, log_interval, print_eval,
    #                     eval_every_nth_batch, error_info
    parser.add_argument("--device", type=str, default="cuda", help="The torch device to train on")
    parser.add_argument("--dataset", type=str, default="mnist", help="The dataset to train on",
                        choices=['mnist', 'cifar10', 'susy', 'kdd', 'sgemm', 'census_income'])
    parser.add_argument("--epochs", type=int, required=True, help="The number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=50, help="The batch size")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="The test batch size")
    parser.add_argument("--lr", type=float, default=1.5 * 1e-4, help="The learning rate")
    parser.add_argument("--dry_run", type=bool, default=False,
                        help="Set to True to perform a dry run and stop after the first batch")
    parser.add_argument("--seed", type=int, default=1, help="The random seed")
    parser.add_argument("--log_interval", type=int, default=0,
                        help="How often to log intermediate results, 0 for never.")
    parser.add_argument("--print_eval", action='store_true',
                        help="Whether the evaluation results should be printed after each epoch")
    parser.add_argument("--eval_every_nth_batch", type=int, default=0,
                        help="How often to evaluate during an epoch, 0 for never.")
    parser.add_argument("--error_info", type=str, default="one_hot_target",
                        choices=["one_hot_target", "zeros", "delayed_loss", "delayed_error", "delayed_loss_one_hot",
                                 "delayed_error_one_hot", "delayed_loss_softmax", "delayed_error_softmax",
                                 'current_error'],
                        help="Which error information to use for F³ training.")

    # output arguments: output_name, output_path
    parser.add_argument('--no_output', action='store_true')
    parser.add_argument("--output_path", type=pathlib.Path, default='results',
                        help="The path to output the results to. Results are written to"
                             "<output_path>/<year>/<month>/<date>/<timestamp>-<output_name>-<uuid>.csv")
    parser.add_argument("--output_name", type=str, default='',
                        help="Optional label to be incorporated into the output file name.")

    args = parser.parse_args(**kwargs)
    return args


if __name__ == '__main__':
    config = parse_args()
    print(config)

    # set seed
    torch.manual_seed(config.seed)

    dataset_config = datasets.get_dataset_config(config.dataset)
    model_kwargs = {
        'input_size': dataset_config['input_size'],
        'output_size': dataset_config['output_size'],
        'mode': config.mode,
        'regression': dataset_config['is_regression'],
        'initialization_method': config.initialization_method,
        'scalar': config.scalar,
        'discrete_values': config.discrete_values,
    }
    if config.model is None:
        assert config.depth is not None and config.width is not None
        model = create_model(config.depth, config.width, **model_kwargs)
    else:
        model_creator = getattr(f3_models, config.model)
        model = model_creator(**model_kwargs)
    print(model)
    results = train_model(model, config.dataset, config.device, config.epochs, config.batch_size,
                          config.test_batch_size, config.lr, config.dry_run, config.log_interval, config.print_eval,
                          config.eval_every_nth_batch, config.error_info)

    if not config.no_output:
        results = postprocess_results(results, config)
        save_results(results, config.output_name, config.output_path)
