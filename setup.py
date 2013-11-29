from setuptools import setup, find_packages
setup(
    name = "LearningNorth",
    version = "0.1",
    packages = find_packages(),
    install_requires= ["numpy",
                       "pil",
                       "scipy",
                       "scikit-learn",
                       "pandas",
                       "cython",
                       "scikit-image"]
)
