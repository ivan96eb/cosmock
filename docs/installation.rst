Installation
============

Runtime install
---------------

Run these commands from the repository root:

.. code-block:: bash

   python -m pip install -e .

This installs ``cosmock`` and the runtime packages needed for fitting,
HEALPix map synthesis, and spectrum conversion.

Expected inputs
---------------

The standard workflow expects:

* a kappa-map array with shape ``(N_bins, N_pix)``;
* a non-Gaussian angular power spectrum array with shape
  ``(N_bins, N_bins, N_ell)``;
* an optional HEALPix pixel window array.

Kappa is dimensionless lensing convergence. The ``C_ell`` arrays are
angular power spectra of those dimensionless fields.
