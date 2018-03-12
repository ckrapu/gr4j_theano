from setuptools import setup

setup(
    name='gr4j_theano',
    version='0.1',
    description='An implementation of the hydrology model GR4J in Theano',
    author='Christopher Krapu',
    author_email='ckrapu@gmail.com',
    py_modules=["gr4j_theano"],
    install_requires=['theano','numpy','pymc3']
)
