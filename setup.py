import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="peaknet-inference",
    version="24.09.06",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="A neural network model finding Bragg peaks in SFX experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/peaknet-inference",
    keywords = ['SFX', 'X-ray', 'Neural network model'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
