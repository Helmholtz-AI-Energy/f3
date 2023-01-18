import setuptools
import pathlib


readme = pathlib.Path(__file__).parent / "README.md"

setuptools.setup(
      name="ffo",
      version="1.0.0",
      description='Feed-Forward-Only (FFO) is a novel, biologically-inspired algorithm to train neural networks without'
                  ' backpropagation.',
      long_description=readme.read_text(encoding='utf-8'),
      long_description_content_type="text/markdown",
      packages=['ffo'],
      install_requires=['numpy', 'pandas', 'scikit-learn', 'torch', 'torchmetrics', 'torchvision', 'tqdm'])
