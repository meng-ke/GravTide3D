# GravTide3D examples

[`example.ipynb`](example.ipynb) provides an OMC-1 observational tutorial:

1. Read the H₂ column-density FITS map `omc1_nh2.fits`.
2. Convert it to mass surface density and reconstruct a 3-D density cube
   using [Volume Density Mapper (VDM)](https://github.com/gxli/volume-density-mapper).
3. Compute and visualize the gravitational potential on CPU.
4. Compute and visualize `lambda_min` of the DC-corrected tidal field on CPU.
5. **Optionally** repeat the tidal calculation with a single CUDA GPU and
   transfer the result back to the host for Matplotlib plotting.

Install GravTide3D from the repository root and provide these tutorial-only
packages in the Jupyter environment:

```bash
python -m pip install -e .
python -m pip install jupyter ipykernel astropy matplotlib
```

Follow VDM's repository for its installation and any additional dependencies.
For CUDA use, install exactly one appropriate CuPy CUDA wheel as documented
in the [main README](../README.md). You can execute all CPU sections without
CUDA; the last section is optional.

**Scientific assumptions:** the column-density-to-mass conversion uses a mean
molecular weight `mu_H2 = 2.8` in *hydrogen-atom mass units per H₂ molecule*.
The OMC-1 distance used for this illustration is 414 pc. GravTide3D assumes
a periodic FFT grid without zero-padding, which may introduce edge artifacts
for a finite observed region.

**Data provenance and citation:** The dataset provider identifies
`omc1_nh2.fits` as an OMC-1 H₂ column-density cutout derived from the
**Herschel Gould Belt Survey (HGBS)**. The FITS header records OMC-1,
`BUNIT=H2/cm^2`, and nominal 36-arcsec resolution. Per the
[IRSA HGBS archive](https://irsa.ipac.caltech.edu/data/Herschel/HGBS/overview.html),
please cite **André et al. (2010)** and the survey dataset DOI
[10.26131/IRSA72](https://doi.org/10.26131/IRSA72).

The local FITS header does **not** identify the exact original map, its
processing/publication reference, or the permission to redistribute this
particular cutout. The maintainer should document its download/derivation
history and confirm redistribution rights **before publishing the FITS file
in a public repository**. HGBS scientific attribution is not, on its own,
a license to redistribute the cutout.
