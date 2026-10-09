from setuptools import setup

setup(name='rocketturbines',
      version='1.0.0',
      description='Python package for sizing and meanline analysis of axial rocket turbines',
      url='https://github.com/janstruzinski/RocketTurbines.git',
      author='Jan Struzinski',
      license='GPL-3.0-only',
      py_modules=['Turbine1D', 'TraupelLossModel', 'IdealGas'],
      package_dir={'': 'RocketTurbines'},
      install_requires=['CoolProp>=6.7.0', 'numpy>=1.26', 'scipy>=1.11', 'thermoprop>=2.1.2'],
      python_requires='>=3.11',
      keywords='rocket turbine axial meanline supersonic impulse partial admission Traupel turbopump',
      zip_safe=False,
      )
