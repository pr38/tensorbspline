from setuptools import setup

setup(

    name='tensorbspline', 
    version='0.0.1', 
    description='Socratic bump search meta estimator, scikit-learn compatible',
    url='https://github.com/pr38/tensorbspline', 
    install_requires=["scikit-learn>=0.20.1"],
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3',
    ],
    packages=["tensorbspline"],
    python_requires='>=3.5',
)
