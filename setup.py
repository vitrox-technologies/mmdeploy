import os

from pkg_resources import parse_version
from setuptools import find_namespace_packages, setup

EXT_TYPE = ''
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension
    cmd_class = {'build_ext': BuildExtension}
    EXT_TYPE = 'torch'
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')
pwd = os.path.dirname(__file__)
version_file = 'mmdeploy/version.py'


def readme():
    with open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    with open(os.path.join(pwd, version_file), 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a file but strips specific
    versioning information.

    Args:
        fname (str): path to the file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def get_extensions():
    extensions = []
    ext_name = 'mmdeploy.backend.torchscript.ts_optimizer'

    if EXT_TYPE == 'torch':
        import glob
        import platform

        from torch.utils.cpp_extension import CppExtension

        try:
            import psutil
            num_cpu = len(psutil.Process().cpu_affinity())
            cpu_use = max(4, num_cpu - 1)
        except (ModuleNotFoundError, AttributeError):
            cpu_use = 4

        os.environ.setdefault('MAX_JOBS', str(cpu_use))
        define_macros = []

        # Before PyTorch1.8.0, when compiling CUDA code, `cxx` is a
        # required key passed to PyTorch. Even if there is no flag passed
        # to cxx, users also need to pass an empty list to PyTorch.
        # Since PyTorch1.8.0, it has a default value so users do not need
        # to pass an empty list anymore.
        # More details at https://github.com/pytorch/pytorch/pull/45956
        extra_compile_args = {'cxx': []}

        # c++14 is required.
        # However, in the windows environment, some standard libraries
        # will depend on c++17 or higher. In fact, for the windows
        # environment, the compiler will choose the appropriate compiler
        # to compile those cpp files, so there is no need to add the
        # argument
        if platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['cxx'] = ['-std=c++14']
            else:
                extra_compile_args['cxx'] = ['-std=c++17']

        include_dirs = []

        op_files = glob.glob(
            './csrc/mmdeploy/backend_ops/torchscript/optimizer/*.cpp'
        ) + glob.glob(
            './csrc/mmdeploy/backend_ops/torchscript/optimizer/ir/*.cpp'
        ) + glob.glob(
            './csrc/mmdeploy/backend_ops/torchscript/optimizer/passes'
            '/onnx/*.cpp')
        extension = CppExtension

        # c++14 is required.
        # However, in the windows environment, some standard libraries
        # will depend on c++17 or higher. In fact, for the windows
        # environment, the compiler will choose the appropriate compiler
        # to compile those cpp files, so there is no need to add the
        # argument
        if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['nvcc'] += ['-std=c++14']
            else:
                extra_compile_args['nvcc'] += ['-std=c++17']

        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
        extensions.append(ext_ops)

    return extensions


if __name__ == '__main__':
    # PACKAGING NOTES FOR THIS FORK. Read before changing anything below.
    #
    # This branch builds the `mmdeploy_vitrox` distribution that VTE depends on.
    # Until 2026-08-27 the distribution name and the package list were NOT in
    # this file: whoever published mmdeploy_vitrox 1.3.1 to PyPI in March 2025
    # edited setup.py locally and did not commit it. The published wheel was
    # therefore not reproducible from this repository, and when that engineer
    # left, nobody could rebuild it. Both edits are committed now so the wheel
    # can be rebuilt by anyone. Do not move them back out of version control.
    #
    # Two more things this build depends on, neither of which is expressible
    # here, so they are written down instead:
    #
    #   1. torch MUST NOT be installed in the build environment. Above, this
    #      file sets EXT_TYPE='torch' when `import torch` succeeds and then
    #      compiles ts_optimizer, which yields a platform-specific wheel. VTE
    #      needs the pure-Python `py3-none-any` wheel, because it stopped using
    #      mmdeploy's compiled ops in favour of onnxruntime_extensions. Absence
    #      of torch is the only switch; there is no flag.
    #   2. Build with setuptools==75.1.0. Newer setuptools dropped
    #      pkg_resources, which this file imports at module level, so the build
    #      fails outright. That version also matches the published wheel.
    #
    # Verified recipe:
    #   uv venv --python 3.10 bldvenv
    #   VIRTUAL_ENV=bldvenv uv pip install "setuptools==75.1.0" wheel build \
    #       cython numpy packaging
    #   bldvenv/bin/python -m build --wheel --no-isolation --outdir dist .
    setup(
        name='mmdeploy_vitrox',
        version=get_version(),
        description='OpenMMLab Model Deployment',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='OpenMMLab',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, model deployment',
        url='https://github.com/open-mmlab/mmdeploy',
        # find_namespace_packages, not find_packages: mmdeploy/codebase/mmrazor/
        # has no __init__.py, so plain find_packages cannot reach
        # mmdeploy.codebase.mmrazor.deploy underneath it. The published 1.3.1
        # wheel ships those files, so its build had this change too.
        packages=find_namespace_packages(include=['mmdeploy', 'mmdeploy.*']),
        include_package_data=True,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=get_extensions(),
        cmdclass=cmd_class,
        zip_safe=False)
