import argparse
import ffo.datasets as datasets

if __name__ == '__main__':
    # arguments to select which datasets are prepared
    parser = argparse.ArgumentParser()
    parser.add_argument('--census_income', action='store_true', help='Download and prepare the census income dataset.')
    parser.add_argument('--sgemm', action='store_true', help='Download and prepare the SGEMM dataset.')
    parser.add_argument('--susy', action='store_true', help='Download and prepare the SUSY dataset.')
    parser.add_argument('--kdd', action='store_true', help='Download and prepare the KDDCup99 dataset.')
    args = parser.parse_args()

    # download and prepare the selected datasets for training
    if args.census_income:
        print('Preparing census income data')
        datasets.CensusIncome.prepare_data()
    if args.sgemm:
        print('Preparing SGEMM data')
        datasets.SGEMMPerformance.prepare_train_test_split()
    if args.susy:
        print('Preparing SUSY data')
        datasets.SUSY.prepare_train_test_split()
    if args.kdd:
        print('Preparing KDDCup99 data')
        datasets.KDDCup99.prepare_train_test_split()
