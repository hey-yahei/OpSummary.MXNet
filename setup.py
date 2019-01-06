import setuptools

VERSION = "0.1.1"
REQUIREMENTS = [
    "mxnet"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mxop",
    version=VERSION,
    author="YaHei",
    author_email="hey-yahei@qq.com",
    description="A tool to count OPs and paramters of MXNet model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hey-yahei/OpSummary.MXNet",
    packages=setuptools.find_packages(),
    license='MIT',

    zip_safe=True,
    install_requires=REQUIREMENTS,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)