import gzip
import io
import pathlib
import shutil
import urllib.request
import zipfile

import pandas
import sklearn.datasets
import torch.utils.data
import torchvision
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot, binary_cross_entropy, mse_loss


class DatasetWithIndices(torch.utils.data.Dataset):
    """Wrapper for a torch dataset to also return the index from __getitem__"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, *self.dataset[index]


class KDDCup99(torch.utils.data.Dataset):
    """Torch dataset for the kddcup99 dataset from sklearn"""
    PATH = 'kddcup99'
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test.csv'
    CAT_COLUMNS = ['protocol_type', 'service', 'flag', 'label']

    def __init__(self, root='data/', train=True, transform=None, target_transform=None, load_fraction=1.0):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        file_path = pathlib.Path(root) / self.PATH / (self.TRAIN_FILE if train else self.TEST_FILE)
        self.dataframe = pandas.read_csv(file_path)
        self.data = torch.tensor(self.dataframe.loc[:, ~self.dataframe.columns.isin(self.CAT_COLUMNS)].values)

        if load_fraction:
            indices = int(len(self.data) * load_fraction)
            self.data = self.data[:indices]

        self.x = self.data[:, :-1].float()
        self.y = self.data[:, -1].to(torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    @property
    def targets(self):
        return self.y

    @classmethod
    def prepare_train_test_split(cls, root='data/', test_size=0.2, random_state=1, **kwargs):
        dataframe, y = sklearn.datasets.fetch_kddcup99(subset=None, data_home=root, shuffle=False, random_state=0,
                                                       percent10=False, download_if_missing=True, return_X_y=True,
                                                       as_frame=True)
        dataframe['label'] = y
        for column in ['protocol_type', 'service', 'flag', 'label']:
            dataframe[column] = dataframe[column].str.decode("utf-8")
            dataframe[f"{column}_enc"] = dataframe[column].astype('category').cat.codes
        dataframe = dataframe.convert_dtypes()

        path = pathlib.Path(root) / cls.PATH
        path.mkdir(parents=True, exist_ok=True)
        train_data, test_data = train_test_split(dataframe, test_size=test_size, random_state=random_state,
                                                 stratify=dataframe['label_enc'], **kwargs)
        train_data.to_csv(path / cls.TRAIN_FILE, index=False)
        test_data.to_csv(path / cls.TEST_FILE, index=False)


class CensusIncome(torch.utils.data.Dataset):
    """
    Torch dataset for the census income (also known as adult) dataset from UCI
    http://archive.ics.uci.edu/ml/datasets/Adult
    """
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
    DATA_SUBDIR = 'census_income'
    IN_FILES_DEFAULT_NAME = ['adult.data', 'adult.test']
    TRAIN_FILE = 'census_income/train.csv'
    TEST_FILE = 'census_income/test.csv'

    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "label"]
    CAT_COLUMNS = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                   "native_country", "label"]

    def __init__(self, root='data/', train=True, transform=None, target_transform=None, load_fraction=1.0):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        file_path = pathlib.Path(root) / (self.TRAIN_FILE if train else self.TEST_FILE)
        self.dataframe = pandas.read_csv(file_path)
        numeric_data = self.dataframe.loc[:, ~self.dataframe.columns.isin(self.CAT_COLUMNS)]
        self.data = torch.tensor(numeric_data.values)

        if load_fraction:
            indices = int(len(self.data) * load_fraction)
            self.data = self.data[:indices]

        self.x = self.data[:, :-1].float()
        self.y = self.data[:, -1].to(torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    @property
    def targets(self):
        return self.y

    @classmethod
    def prepare_data(cls, in_file_train=None, in_file_test=None, root='data', download=True):
        root = pathlib.Path(root)
        (root / cls.DATA_SUBDIR).mkdir(parents=True, exist_ok=True)

        default_train_file, default_test_file = cls.IN_FILES_DEFAULT_NAME
        in_file_train = default_train_file if in_file_train is None else in_file_train
        in_file_test = default_test_file if in_file_test is None else in_file_test
        in_file_train = root / cls.DATA_SUBDIR / in_file_train
        in_file_test = root / cls.DATA_SUBDIR / in_file_test

        if not (in_file_train.exists() and in_file_test.exists()):
            if download:
                in_file_train, in_file_test = cls.download(root)
            else:
                raise FileNotFoundError(f'In files {in_file_train}, {in_file_test} not found. Pass download=True to '
                                        'download them automatically from the UCI directory.')

        for in_file, out_file in zip([in_file_train, in_file_test], [cls.TRAIN_FILE, cls.TEST_FILE]):
            dataframe = pandas.read_csv(in_file, names=cls.COLUMNS, comment='|')
            for column in cls.CAT_COLUMNS:
                dataframe[f"{column}_enc"] = dataframe[column].astype('category').cat.codes
            dataframe = dataframe.convert_dtypes()
            dataframe.to_csv(root / out_file, index=False)

    @classmethod
    def download(cls, root='data'):
        # download data from URL and extract to root/DATA_SUBDIR
        root = pathlib.Path(root)
        path = root / cls.DATA_SUBDIR
        path.mkdir(parents=True, exist_ok=True)

        for file_name in cls.IN_FILES_DEFAULT_NAME:
            http_response = urllib.request.urlopen(cls.URL + '/' + file_name)
            with open(path / file_name, 'wb') as file:
                file.write(http_response.read())

        return [path / file_name for file_name in cls.IN_FILES_DEFAULT_NAME]


class UCIDataset(torch.utils.data.Dataset):
    """Torch dataset for UCI datasets stored in a csv"""
    COLUMNS = None
    TRAIN_FILE = None
    TEST_FILE = None

    def __init__(self, root='data/', train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        file_path = pathlib.Path(root) / (self.TRAIN_FILE if train else self.TEST_FILE)
        self._raw_data = torch.tensor(pandas.read_csv(file_path).values)

        self.x, self.y = None, None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    @property
    def targets(self):
        return self.y

    @classmethod
    def prepare_train_test_split(cls, in_file, root='data', test_size=0.2, random_state=1, download=True, **kwargs):
        root = pathlib.Path(root)
        in_file = root / in_file

        if not in_file.exists():
            if download:
                in_file = cls.download(root)
            else:
                raise FileNotFoundError(f'In file {in_file} not found. Pass download=True to download it automatically '
                                        'from the UCI directory.')

        dataframe = pandas.read_csv(in_file, names=cls.COLUMNS)
        train_data, test_data = train_test_split(dataframe, test_size=test_size, random_state=random_state, **kwargs)
        train_data.to_csv(root / cls.TRAIN_FILE, index=False)
        test_data.to_csv(root / cls.TEST_FILE, index=False)

    @classmethod
    def download(cls, root='data'):
        raise NotImplementedError(f'Please implement a specific download method in the UCIDataset subclass')


class SGEMMPerformance(UCIDataset):
    """
    Torch dataset for the SGEMM dataset from UCI
    https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance
    """
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00440/sgemm_product_dataset.zip'
    COLUMNS = None
    DATA_SUBDIR = 'sgemm_product_dataset'
    IN_FILE_DEFAULT_NAME = 'sgemm_product.csv'
    TRAIN_FILE = 'sgemm_product_dataset/train.csv'
    TEST_FILE = 'sgemm_product_dataset/test.csv'

    FEATURES = ['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'MDIMA', 'NDIMB',
                'KWI', 'VWM', 'VWN', 'STRM', 'STRN', 'SA', 'SB']

    # mean and std over the training dataset (features and targets)
    MEAN = torch.tensor([80.45099338, 80.33145695, 25.52218543, 13.93956954, 13.93000828, 17.37127483, 17.38451987,
                         4.9973303, 2.45177463, 2.45005174, 0.49980339, 0.50042943, 0.49984478, 0.50050186,
                         217.8314705220405])
    STD = torch.tensor([42.47476047, 42.45861397, 7.85386979, 7.88287486, 7.86731171, 9.3888511, 9.39787133, 3.00000657,
                        1.95446444, 1.9563342, 0.50000125, 0.50000111, 0.50000127, 0.50000104, 369.30339822069976])

    def __init__(self, root='data/', train=True, transform=None, target_transform=None):
        super().__init__(root, train, transform, target_transform)
        data = (self._raw_data - self.MEAN[None, :]) / self.STD[None, :]
        self.x = data[:, :-1].float()
        self.y = data[:, -1:].float()

    @classmethod
    def prepare_train_test_split(cls, in_file=None, root='data', test_size=0.2, random_state=1, download=True,
                                 **kwargs):
        root = pathlib.Path(root)
        in_file_name = f'{cls.DATA_SUBDIR}/{cls.IN_FILE_DEFAULT_NAME}' if in_file is None else in_file
        in_file = root / in_file_name

        if not in_file.exists():
            if download:
                in_file = cls.download(root)
            else:
                raise FileNotFoundError(f'In file {in_file} not found. Pass download=True to download it automatically '
                                        'from the UCI directory.')

        dataframe = pandas.read_csv(in_file, names=cls.COLUMNS)
        train_data, test_data = train_test_split(dataframe, test_size=test_size, random_state=random_state, **kwargs)
        train_data = train_data.melt(id_vars=cls.FEATURES).drop(columns=['variable'])
        train_data.to_csv(root / cls.TRAIN_FILE, index=False)
        test_data = test_data.melt(id_vars=cls.FEATURES).drop(columns=['variable'])
        test_data.to_csv(root / cls.TEST_FILE, index=False)

    @classmethod
    def download(cls, root='data'):
        # download data from URL and extract to root/DATA_SUBDIR
        root = pathlib.Path(root)
        path = root / cls.DATA_SUBDIR
        http_response = urllib.request.urlopen(cls.URL)
        zip_file = zipfile.ZipFile(io.BytesIO(http_response.read()))
        zip_file.extractall(path=path)

        return path / cls.IN_FILE_DEFAULT_NAME


class SUSY(UCIDataset):
    """
    Torch dataset for the SUSY dataset from UCI
    https://archive.ics.uci.edu/ml/datasets/SUSY
    """
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz'
    COLUMNS = ['is_noise', 'lepton_1_pT', 'lepton_1_eta', 'lepton_1_phi', 'lepton_2_pT', 'lepton_2_eta', 'lepton_2_phi',
               'missing_energy_magnitude', 'missing_energy_phi', 'MET_rel', 'axial_MET', 'M_R', 'M_TR_2', 'R', 'MT2',
               'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']

    # mean and std over the train features
    MEAN = torch.tensor([1.00030744e+00, -2.37989387e-04, 3.51096376e-05, 9.99471085e-01, -3.64020384e-04,
                         3.56027145e-04, 9.99904375e-01, -1.43986532e-04, 1.00143192e+00, 2.40201365e-04,
                         1.00032161e+00, 9.99864998e-01, 9.99830453e-01, 1.00023247e+00, 1.00006217e+00,
                         1.00017825e+00, 9.99421870e-01, 2.24871813e-01])
    STD = torch.tensor([0.68751029, 1.00306866, 1.0017512, 0.65468612, 1.00264245, 1.00156785, 0.87232394,
                        1.0016187, 0.88986497, 1.00120734, 0.62870944, 0.58410324, 0.47077775, 0.8591677,
                        0.62059026, 0.62380901, 0.43608498, 0.19698323])
    DATA_SUBDIR = 'SUSY'
    IN_FILE_DEFAULT_NAME = 'SUSY.csv'
    TRAIN_FILE = 'SUSY/train.csv'
    TEST_FILE = 'SUSY/test.csv'

    def __init__(self, root='data/', train=True, transform=None, target_transform=None):
        super().__init__(root, train, transform, target_transform)
        self.x = self._raw_data[:, 1:].float()
        self.x = (self.x - self.MEAN[None, :]) / self.STD[None, :]
        self.y = self._raw_data[:, 0].to(torch.int64)

    @classmethod
    def prepare_train_test_split(cls, in_file=None, root='data', test_size=0.2, random_state=1, download=True,
                                 **kwargs):
        in_file = f'{cls.DATA_SUBDIR}/{cls.IN_FILE_DEFAULT_NAME}' if in_file is None else in_file
        super().prepare_train_test_split(in_file, root, test_size, random_state, download, **kwargs)

    @classmethod
    def download(cls, root='data'):
        # download data from URL and extract to root/DATA_SUBDIR
        root = pathlib.Path(root)
        (root / cls.DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
        out_path = root / cls.DATA_SUBDIR / cls.IN_FILE_DEFAULT_NAME

        http_response = urllib.request.urlopen(cls.URL)
        gz_file_path = root / cls.DATA_SUBDIR / cls.URL.split('/')[-1]
        with open(gz_file_path, 'wb') as gz_file:
            gz_file.write(http_response.read())

        with gzip.open(gz_file_path, 'rb') as gz_file:
            with open(out_path, 'wb') as out_file:
                shutil.copyfileobj(gz_file, out_file)

        return out_path


def data_loaders(train_set, test_set, batch_size, test_batch_size, **data_loader_kwargs):
    train_loader = torch.utils.data.DataLoader(DatasetWithIndices(train_set), shuffle=True, batch_size=batch_size,
                                               **data_loader_kwargs)
    train_test_loader = torch.utils.data.DataLoader(train_set, shuffle=False, batch_size=test_batch_size,
                                                    **data_loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, **data_loader_kwargs)

    return train_loader, train_test_loader, test_loader


def mnist_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    num_classes = 10
    target_transform = torchvision.transforms.Compose([lambda target: one_hot(torch.tensor(target),
                                                                              num_classes).float()])

    train_data = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform,
                                            target_transform=target_transform)
    test_data = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform,
                                           target_transform=target_transform)
    return data_loaders(train_data, test_data, batch_size, test_batch_size, **data_loader_kwargs)


def cifar10_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs):
    normalize = torchvision.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    num_classes = 10
    target_transform = torchvision.transforms.Compose([lambda target: one_hot(torch.tensor(target),
                                                                              num_classes).float()])

    train_data = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform,
                                              target_transform=target_transform)
    test_data = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform,
                                             target_transform=target_transform)
    return data_loaders(train_data, test_data, batch_size, test_batch_size, **data_loader_kwargs)


def susy_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs):
    num_classes = 2
    target_transform = torchvision.transforms.Compose([lambda target: one_hot(target, num_classes).float()])

    train_data = SUSY(root, train=True, transform=None, target_transform=target_transform)
    test_data = SUSY(root, train=False, transform=None, target_transform=target_transform)
    return data_loaders(train_data, test_data, batch_size, test_batch_size, **data_loader_kwargs)


def kdd_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs):
    num_classes = 23
    target_transform = torchvision.transforms.Compose([lambda target: one_hot(target, num_classes).float()])

    train_data = KDDCup99(root, train=True, transform=None, target_transform=target_transform, load_fraction=1.0)
    test_data = KDDCup99(root, train=False, transform=None, target_transform=target_transform)
    return data_loaders(train_data, test_data, batch_size, test_batch_size, **data_loader_kwargs)


def census_income_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs):
    num_classes = 2
    target_transform = torchvision.transforms.Compose([lambda target: one_hot(target, num_classes).float()])

    train_data = CensusIncome(root, train=True, transform=None, target_transform=target_transform)
    test_data = CensusIncome(root, train=False, transform=None, target_transform=target_transform)
    return data_loaders(train_data, test_data, batch_size, test_batch_size, **data_loader_kwargs)


def sgemm_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs):
    train_data = SGEMMPerformance(root, train=True, transform=None, target_transform=None)
    test_data = SGEMMPerformance(root, train=False, transform=None, target_transform=None)
    return data_loaders(train_data, test_data, batch_size, test_batch_size, **data_loader_kwargs)


def get_dataset_config(dataset):
    configs = {
        'mnist': {'input_size': [1, 28, 28], 'output_size': 10, 'loss': binary_cross_entropy,
                  'classification_target': 'one_hot', 'is_regression': False},
        'cifar10': {'input_size': [3, 32, 32], 'output_size': 10, 'loss': binary_cross_entropy,
                    'classification_target': 'one_hot', 'is_regression': False},
        'susy': {'input_size': 18, 'output_size': 2, 'loss': binary_cross_entropy,
                 'classification_target': 'one_hot', 'is_regression': False},
        'kdd': {'input_size': 41, 'output_size': 23, 'loss': binary_cross_entropy,
                'classification_target': 'one_hot', 'is_regression': False},
        'census_income': {'input_size': 14, 'output_size': 2, 'loss': binary_cross_entropy,
                          'classification_target': 'one_hot', 'is_regression': False},
        'sgemm': {'input_size': 14, 'output_size': 1, 'loss': mse_loss,
                  'classification_target': None, 'is_regression': True},
    }
    if dataset not in configs:
        raise ValueError(f'Invalid dataset {dataset}. Valid datasets are: {configs.keys()}')
    return configs[dataset]


def get_data_loaders(dataset, batch_size, test_batch_size, root='data', **data_loader_kwargs):
    if dataset == 'mnist':
        return mnist_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs)
    if dataset == 'cifar10':
        return cifar10_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs)
    elif dataset == 'susy':
        return susy_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs)
    elif dataset == 'kdd':
        return kdd_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs)
    elif dataset == 'census_income':
        return census_income_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs)
    elif dataset == 'sgemm':
        return sgemm_data_loaders(batch_size, test_batch_size, root, **data_loader_kwargs)
    else:
        raise ValueError(f'Invalid dataset {dataset}.')
