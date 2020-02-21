import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from distutils.version import LooseVersion


class GetPybindInclude:
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def search_pybind11_headers(self):
        import pybind11

        def recommended():
            return pybind11.get_include(self.user)

        def setuptools_temp_egg():
            # If users of setuptools drag in pybind11 only as a setup_require(ment), the pkg will be placed
            # temporarily into .eggs, but we can not use the headers directly. So we have to
            # link non-installed header files to correct subdirectory, so they can be used during compilation
            found = False
            for p in pybind11.__path__:
                if '.egg' in p:
                    found = True
            if not found:
                return ''

            header_src = os.path.abspath(os.path.join(pybind11.__path__[0], '..'))
            hdrs = []

            for _, _, filenames in os.walk(header_src):
                hdrs += [f for f in filenames if f.endswith('.h')]
            for h in sorted(hdrs):
                if 'detail' in h:
                    sub = 'detail'
                else:
                    sub = ''
                dest = os.path.join(pybind11.__path__[0], sub, os.path.basename(h))
                try:
                    os.link(h, dest)
                except OSError:
                    pass
            return header_src

        methods = (recommended(),
                   setuptools_temp_egg(),
                   )
        for m in methods:
            if os.path.exists(os.path.join(m, 'pybind11', 'pybind11.h')):
                return m
        return ''

    def __str__(self):
        result = self.search_pybind11_headers()
        if not result:
            raise RuntimeError()
        return result


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            try:
                GPP_COMPILER = subprocess.check_output("which g++", shell=True).decode().strip('\n\t\r')
                GCC_COMPILER = subprocess.check_output("which gcc", shell=True).decode().strip('\n\t\r')
            except subprocess.CalledProcessError as e:
                GPP_COMPILER, GCC_COMPILER = None, None
            else:
                cmake_args += ['-DCMAKE_CXX_COMPILER=' + GPP_COMPILER,
                               '-DCMAKE_C_COMPILER=' + GCC_COMPILER]
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


print(GetPybindInclude())

setup(
    name='gacf',
    version='0.3',
    author='Joshua Briegal',
    author_email='jtb34@cam.ac.uk',
    description='A generalisation of the autocorrelation function, for non-uniformly sampled timeseries data',
    long_description='',
    packages=['gacf'],
    ext_modules=[CMakeExtension('gacf.correlator', 'gacf'), CMakeExtension('gacf.datastructure', 'gacf')],
    cmdclass=dict(build_ext=CMakeBuild,
                  test=PyTest),
    tests_require=['pytest'],
    zip_safe=False,
    setup_requires=['pybind11']
)
