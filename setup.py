from setuptools import setup, find_packages

setup(name='psrl',
      version='0.1',
      packages=find_packages(),
      python_requires=">=3.8",
      install_requires=[
            'ipykernel==6.28.0',
            'numpy==1.23.5',
            'matplotlib==3.7.0',
            'scipy==1.10.1',
            'pandas==2.2.2',
            'gymnasium==1.1.1',
            'pygame==2.6.1'
      ],
      extras_require={
            "dev": [
            "pytest",
            "jupyter",
            # other dev tools
        ]
    },)