import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mup",
    use_scm_version={"local_scheme": "no-local-version"},
    author="Edward J Hu, Greg Yang, Jianbin Chang",
    author_email="edwardjhu@edwardjhu.com, gregyang@microsoft.com, shjwudp@gmail.com",
    description="Maximal Update Parametrization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shjwudp/mup",
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'seaborn',
        'tqdm',
        'pyyaml'
      ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
