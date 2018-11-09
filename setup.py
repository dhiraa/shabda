
import setuptools


long_description = '''
Shabda is an open-source toolkit based on Tensorflow,
aiming to support a broad set of machine learning in and around audio,
such as speech commands classification, genre classification, audio clustering, and so on.
Shabda is designed for both researchers and practitioners for fast prototyping and experimentation.
'''

setuptools.setup(
    name="shabda",
    version="0.1",
    url="https://github.com/dhiraa/shabda",

    description="Toolkit for DL on Audio and Beyond",
    long_description=long_description,
    license='Apache License Version 2.0',

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=[
        'numpy',
        'pyyaml',
        'requests',
    ],
    extras_require={
        'tensorflow-cpu': ['tensorflow>=1.7.0', 'tensorflow-probability >= 0.3.0'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.7.0', 'tensorflow-probability-gpu >= 0.3.0']
    },
    package_data={
        "shabda": [
            "../bin/utils/multi-bleu.perl",
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
    ],
)
