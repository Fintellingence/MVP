try:
    from Cython.Distutils import build_ext
except ImportError:
    raise ImportError("Cython is not installed")
import numpy as np
from setuptools import Extension


def build(setup_kwargs):
    ext = [
        Extension(
            name="mvp.bootstrap",
            sources=["mvp/bootstrap.pyx", "mvp/_bootstrap.cpp"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"],
            language="c++",
        )
    ]

    setup_kwargs.update(
        {"ext_modules": ext, "cmdclass": {"build_ext": build_ext}}
    )
