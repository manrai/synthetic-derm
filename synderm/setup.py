from setuptools import setup, find_packages

setup(
    name='synderm',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas'
    ],
    author='Thomas Buckley',
    author_email='',
    description='A package for generating synthetic data to augment image classifiers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/manrai/synderm',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license='MIT',
)
