import os
from setuptools import setup, Extension, find_packages

_VERSION = "1.0"

sparseSEM_lib = Extension(name='sparseSEM.lassoSML',
                          sources=[],
                          include_dirs=['sparseSEM/src',
                                        '/opt/intel/oneapi/mkl/2023.1.0/include'],
                          library_dirs=['/opt/intel/oneapi/mkl/2023.1.0/lib'],
                          libraries=['mkl_rt'],
                          extra_compile_args=['-std=c11', '-O2'])

sparseSEM_lib.name = 'sparseSEM'

if __name__ == "__main__":
    setup(
        name="sparseSEM",
        version=_VERSION,
        description="Python wrapper for sparseSEM",
        long_description_content_type="text/markdown",
        long_description=open('README.md').read(),
        author="Anhui Huang",
        author_email="anhuihuang@gmail.com",
        url="https://scholar.google.com/citations?user=WhDMZEIAAAAJ&hl=en",
        install_requires=[
            "numpy>=1.9.2",
            "networkx",
            "matplotlib"
        ],
        python_requires=">=3.6",
        setup_requires=["setuptools"],
        ext_modules=[sparseSEM_lib],
        packages=find_packages(),
        package_data={'sparseSEM': ['src/*.c','src/*.h', 'data/*', 'doc/*']},
        include_package_data=True,
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3 :: Only',
            'Operating System :: OS Independent',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
            'Topic :: Scientific/Engineering'
        ]
    )
