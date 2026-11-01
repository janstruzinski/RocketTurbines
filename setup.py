from setuptools import find_packages, setup

setup(name='rocketturbines',
      version='1.1.0',
      description='Python package for sizing and meanline analysis of axial rocket turbines',
      url='https://github.com/janstruzinski/RocketTurbines.git',
      author='Jan Struzinski',
      license='GPL-3.0-only',
      packages=find_packages(include=['RocketTurbines', 'RocketTurbines.*']),
      install_requires=['CoolProp>=6.7.0', 'numpy>=1.26', 'scipy>=1.11', 'thermoprop>=2.1.2', 'tabulate>=0.9.0',
                        'matplotlib>=3.7', 'plotly>=6.0', 'mapbox-earcut>=1.0.3', 'pymoo>=0.6.2,<0.7',
                        'supersonic-turbine-blading @ git+https://github.com/janstruzinski/SupersonicTurbineBlading.git'],
      python_requires='>=3.11',
      keywords='rocket turbine axial meanline supersonic impulse partial admission Traupel turbopump',
      zip_safe=False,
      )
