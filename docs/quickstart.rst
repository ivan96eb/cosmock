Quickstart
==========

The repository includes example arrays in ``data/``. This script fits a
``G3`` transformation and generates one mock kappa-map cube.

.. code-block:: python

   import numpy as np

   from cosmock import fit_parameters, generate_mocks

   path_to_map = "data/Kappa_Gower_St_ID_44_DESy3_tomography_Nside_256.npy"
   path_to_cl = "data/UNBIASED_3point75nsideminus1_Cls_NG_Gower_St_ID_44.npy"
   path_to_pixwin = "data/pixwin_256.npy"

   kappa_map = np.load(path_to_map)
   cl_ng = np.load(path_to_cl)
   pixwin = np.load(path_to_pixwin)

   order = 3
   n_mocks = 1

   params = fit_parameters(kappa_map, cl_ng, order)
   mocks = generate_mocks(params, n_mocks, pixwin=pixwin)

   print(mocks.shape)

Expected progress messages:

.. code-block:: text

   Done fitting the G3 parameters
   Done finding the power spectrum of the underlying gaussian random field

Expected output shape:

.. code-block:: text

   (1, N_bins, N_pix)

For the included data, ``N_pix`` is the number of HEALPix pixels at
``nside=256``. Each generated mock is a dimensionless kappa-map cube with
one map per tomographic bin.

Standard workflow
-----------------

The main user workflow has two steps:

1. ``fit_parameters`` learns the transformation parameters and converts the
   target non-Gaussian spectra to latent Gaussian spectra.
2. ``generate_mocks`` samples latent Gaussian maps and transforms them into
   mock kappa maps.

The current implementation supports transformation orders ``N=2`` and
``N=3``.
