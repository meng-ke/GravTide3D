"""Split-dimensional exact FFT implementation notes.

The operational v0.1 out-of-core executor is adapted from the production
engine in ``gravtide3d._internal.production_splitfft``.  This module is the
stable refactoring target: the algorithm is

    rho(z,y,x)
      -> rFFT2(y,x) for each z plane
      -> FFT(z) for spectral (ky,kx) tiles
      -> Poisson / tidal multipliers
      -> inverse FFT(z)
      -> inverse rFFT2(y,x).

Spatial subdomains are never treated as independent gravity problems.
"""
