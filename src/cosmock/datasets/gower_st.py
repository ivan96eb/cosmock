"""Project-specific Gower Street map loader."""

from __future__ import annotations

import os

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .._optional import import_optional


class GowerStConvergenceIA:
    """Load Gower Street convergence and intrinsic-alignment maps."""

    def __init__(
        self,
        sim_serial_number,
        *,
        cosmology_path="/spiff/ssarnabo/GowerStSims/IvanData/GowerStCosmoPars.npy",
        simdir_root="/spiff/ssarnabo/GowerStSims/sim",
        nz_path="/spiff/ivanespinoza/weak_lensing_data_emulator_data/Making_cls_match_pred",
    ):
        pyccl = import_optional("pyccl", extra="ccl", purpose="Loading Gower Street maps")
        self._pyccl = pyccl

        GowerStcosmo_pars = np.load(cosmology_path)
        self.cosmo_pars = GowerStcosmo_pars[sim_serial_number - 1]
        self.z_bins = np.linspace(0.0, 3.0, 1000)
        simdir = simdir_root + f"{sim_serial_number:05}"
        self.z_slab_file = simdir + "/z_values.txt"
        self.z_slab_data = np.genfromtxt(self.z_slab_file, skip_header=1, delimiter=",")
        self.slab_files = self.get_slab_files(simdir)
        self.z_boundaries, self.z_mid = self.get_z_slabs(self.z_slab_data, self.slab_files)
        self.cosmo = self.get_cosmo(self.cosmo_pars)
        self.chi = pyccl.comoving_radial_distance(self.cosmo, 1.0 / (1.0 + self.z_bins))
        self.chi_boundaries = pyccl.comoving_radial_distance(
            self.cosmo, 1.0 / (1.0 + self.z_boundaries)
        )
        self.N_slabs = self.chi_boundaries.shape[0] - 1
        self.N_SRC_BINS = 4
        self.nz_list = [np.load(nz_path + "/source_bin%d.npy" % (i + 1)) for i in range(self.N_SRC_BINS)]
        self.DZ_SRC = np.zeros(self.N_SRC_BINS)
        self.wl_tracers = [
            pyccl.WeakLensingTracer(
                self.cosmo, dndz=(self.nz_list[i][0] + self.DZ_SRC[i], self.nz_list[i][1])
            )
            for i in range(self.N_SRC_BINS)
        ]
        self.lensing_kernels = [self.wl_tracers[i].get_kernel(self.chi)[0] for i in range(self.N_SRC_BINS)]

        self.ia_tracers = [
            pyccl.NumberCountsTracer(
                self.cosmo,
                dndz=(self.nz_list[i][0] + self.DZ_SRC[i], self.nz_list[i][1]),
                bias=(self.nz_list[i][0] + self.DZ_SRC[i], np.ones_like(self.nz_list[i][0])),
                has_rsd=False,
            )
            for i in range(self.N_SRC_BINS)
        ]
        self.ia_kernels = [self.ia_tracers[i].get_kernel(self.chi)[0] for i in range(self.N_SRC_BINS)]

        self.lensing_weights = [self.get_kernel_weight(kernel) for kernel in self.lensing_kernels]
        self.ia_weights = [self.get_kernel_weight(kernel) for kernel in self.ia_kernels]
        dens_list = []
        for file in self.slab_files[:-1]:
            dens = np.load(simdir + "/" + file)
            delta = dens / dens.mean() - 1.0
            dens_list.append(delta)
        self.dens_arr = np.array(dens_list)

        self.C_cr = 0.013877
        self.z0 = 0.62
        self.C_ia = self.A2C(1, 0.0, self.cosmo_pars[0])

    def get_slab_files(self, simdir):
        file_list = os.listdir(simdir)
        slab_files = []
        for file in file_list:
            if file[:3] == "run" and file[-3:] == "npy":
                if "incomplete" not in file.split("."):
                    slab_files.append(file)
        slab_files.sort(reverse=True)
        return slab_files

    def get_highest_z(self, z_near, slab_files):
        last_file = slab_files[-1]
        return z_near[int(last_file.split(".")[1])]

    def get_z_indices(self, slab_files):
        indices = []
        for file in slab_files:
            ind = int(file.split(".")[1]) - 1
            indices.append(ind)
        return indices

    def get_z_slabs(self, z_slab_data, slab_files):
        z_near = z_slab_data[:, 1]
        z_far = z_slab_data[:, 2]
        z_hi = self.get_highest_z(z_near, slab_files)
        z_indices = self.get_z_indices(slab_files)
        z_boundaries = z_far[z_indices]
        z_boundaries = z_boundaries[z_boundaries <= z_hi]
        z_slabs = 0.5 * (z_boundaries[1:] + z_boundaries[:-1])
        return z_boundaries, z_slabs

    def get_cosmo(self, cosmo_pars):
        Omega_m, sigma8, w, omegab, h, ns, mnu = cosmo_pars
        Omega_b = omegab / h / h
        Omega_c = Omega_m - Omega_b
        return self._pyccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            n_s=ns,
            sigma8=sigma8,
            w0=w,
            m_nu=mnu,
            transfer_function="boltzmann_camb",
        )

    def get_kernel_weight(self, kernel):
        """Get kernel weights to multiply with density slabs."""

        kernel_interp = interp1d(self.chi, kernel)
        weights = []
        for i in range(self.N_slabs):
            weight = quad(kernel_interp, self.chi_boundaries[i], self.chi_boundaries[i + 1])[0]
            weights.append(weight)
        return np.array(weights)

    def A2C(self, A1, eta=0.0, OmegaM=0.3):
        """Compute the intrinsic-alignment coefficient."""

        return -A1 * self.C_cr * OmegaM

    def create_field(self, weights, dens_arr):
        """Create an integrated field from density slabs and slab weights."""

        return np.sum(weights[:, np.newaxis] * dens_arr, axis=0)

    def get_kappa_map(self):
        """Return convergence maps in the configured tomographic bins."""

        return np.array([self.create_field(weights, self.dens_arr) for weights in self.lensing_weights])

    def get_IA_map(self):
        """Return intrinsic-alignment maps in the configured tomographic bins."""

        return self.C_ia * np.array([self.create_field(weights, self.dens_arr) for weights in self.ia_weights])

