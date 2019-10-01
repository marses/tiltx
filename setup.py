import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiltx",
    version="0.0.1",
    author="Marko Seslija",
    author_email="marko.seslija@gmail.com",
    description="Feature extraction from mobile phone motion sensor data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marses/tiltx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis"
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
