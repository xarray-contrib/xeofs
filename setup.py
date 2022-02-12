from setuptools import setup, find_packages

with open("./requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read()


setup(
    name='xeofs',
    version='0.0.0',
    license='MIT',
    description='Empirical orthogonal functions (EOF) analysis and variants used in climate science for numpy, pandas and xarray',
    packages=find_packages(),
    long_description_content_type='text/x-rst',
    long_description='README.rst',
    author='Niclas Rieger',
    author_email='niclasrieger@gmail.com',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    include_package_data=True,
    # url='https://github.com/nicrie/xmca',
    python_requires='>=3.8',
    install_requires=REQUIRED_PACKAGES,
    test_suite='test',
)
