from setuptools import setup, find_packages

setup(
    name="segbook",
    version="0.0.4",
    author="Jin Ye",
    author_email="jin.ye@monash.edu", 
    description="Medical image segmentation model zoo",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uni-medical/MedSegModelZoo",
    packages=find_packages(include=["segbook", "segbook.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    python_requires=">=3.9",
    license="Apache-2.0",
    install_requires=[
        "numpy<2",
        "tqdm>=4.50.0",
        "requests>=2.24.0",
        "nnunet==1.7.1",
        "torch>=2.0.0",
        "dicom2nifti",
        "scikit-image>=0.14",
        "medpy",
        "scipy",
        "batchgenerators>=0.23",
        "scikit-learn",
        "SimpleITK",
        "pandas",
        "nibabel", 
        "tifffile", 
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "segbook=segbook.segbook:main",
        ],
    },
)
