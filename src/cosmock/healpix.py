"""Optional HEALPix and torch autograd helpers."""

from __future__ import annotations

import math

import numpy as np

from .util.optional import OptionalDependencyError

try:
    import healpy as hp
    import torch
except ImportError:
    hp = None
    torch = None


def _require_torch_healpix():
    if hp is None or torch is None:
        raise OptionalDependencyError(
            "HEALPix torch transforms require optional dependencies 'torch' and 'healpy'. "
            "Install them with `pip install cosmock[torch]`."
        )


class _UnavailableAutogradFunction:
    def __init__(self, name):
        self.name = name

    def apply(self, *args, **kwargs):
        _require_torch_healpix()


if torch is not None and hp is not None:

    class Alm2Map(torch.autograd.Function):
        @staticmethod
        def forward(ctx, alms, nside, lmax):
            ctx.alms = alms
            ctx.nside = nside
            ctx.lmax = lmax
            return torch.tensor(hp.alm2map(alms.numpy(), nside, lmax=lmax))

        @staticmethod
        def backward(ctx, grad_output):
            nside = ctx.nside
            lmax = ctx.lmax
            _, emm = hp.Alm.getlm(lmax)
            a = torch.ones(len(emm), dtype=torch.double)
            a[emm > 0] = 2
            grad_out_alm = Map2Alm.apply(grad_output, lmax)
            grad_alm = a * hp.nside2npix(nside) / (4 * math.pi) * grad_out_alm
            return grad_alm, None, None


    class Map2Alm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, m, lmax):
            ctx.m = m
            ctx.nside = hp.npix2nside(len(m))
            ctx.lmax = lmax
            return torch.tensor(hp.map2alm(m.numpy(), lmax=lmax, use_pixel_weights=True))

        @staticmethod
        def backward(ctx, grad_output):
            nside = ctx.nside
            lmax = ctx.lmax
            _, emm = hp.Alm.getlm(lmax)
            a = torch.ones(len(emm), dtype=torch.double)
            a[emm > 0] = 0.5
            grad_out_m = Alm2Map.apply(a * grad_output, nside, lmax)
            grad_m = 4 * math.pi / hp.nside2npix(nside) * grad_out_m
            return grad_m, None


    class Alm2MapSpin(torch.autograd.Function):
        @staticmethod
        def forward(ctx, elm, blm, nside, lmax):
            ctx.nside = nside
            ctx.lmax = lmax
            inputs = [np.zeros_like(elm.numpy()), elm.numpy(), blm.numpy()]
            _, q, u = hp.alm2map(inputs, nside, lmax=lmax)
            return torch.tensor(q), torch.tensor(u)

        @staticmethod
        def backward(ctx, q_grad, u_grad):
            nside = ctx.nside
            lmax = ctx.lmax
            _, emm = hp.Alm.getlm(lmax)
            a = torch.ones(len(emm), dtype=torch.double)
            a[emm > 0] = 2
            elm_grad, blm_grad = Map2AlmSpin.apply(q_grad, u_grad, lmax)
            elm_grad = a * hp.nside2npix(nside) / (4 * math.pi) * elm_grad
            blm_grad = a * hp.nside2npix(nside) / (4 * math.pi) * blm_grad
            return elm_grad, blm_grad, None, None


    class Map2AlmSpin(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, u, lmax):
            ctx.nside = hp.npix2nside(len(q))
            ctx.lmax = lmax
            inputs = [np.zeros_like(q.numpy()), q.numpy(), u.numpy()]
            _, elm, blm = hp.map2alm(inputs, lmax=lmax, use_pixel_weights=True)
            return torch.tensor(elm), torch.tensor(blm)

        @staticmethod
        def backward(ctx, elm_grad, blm_grad):
            nside = ctx.nside
            lmax = ctx.lmax
            _, emm = hp.Alm.getlm(lmax)
            a = torch.ones(len(emm), dtype=torch.double)
            a[emm > 0] = 0.5
            q_grad, u_grad = Alm2MapSpin.apply(a * elm_grad, a * blm_grad, nside, lmax)
            q_grad = 4 * math.pi / hp.nside2npix(nside) * q_grad
            u_grad = 4 * math.pi / hp.nside2npix(nside) * u_grad
            return q_grad, u_grad, None


    class UDGrade(torch.autograd.Function):
        @staticmethod
        def forward(ctx, m, out_nside):
            ctx.in_nside = hp.npix2nside(len(m))
            ctx.out_nside = out_nside
            return torch.tensor(hp.ud_grade(m.numpy(), out_nside))

        @staticmethod
        def backward(ctx, grad_out):
            if ctx.out_nside > ctx.in_nside:
                fac = (ctx.out_nside / ctx.in_nside) ** 2.0
            else:
                fac = (ctx.in_nside / ctx.out_nside) ** (-2.0)
            grad = UDGrade.apply(grad_out, ctx.in_nside) * fac
            return grad, None

else:
    Alm2Map = _UnavailableAutogradFunction("Alm2Map")
    Map2Alm = _UnavailableAutogradFunction("Map2Alm")
    Alm2MapSpin = _UnavailableAutogradFunction("Alm2MapSpin")
    Map2AlmSpin = _UnavailableAutogradFunction("Map2AlmSpin")
    UDGrade = _UnavailableAutogradFunction("UDGrade")


def _ell_indexed_window_values(pixwin, ell):
    """Return one pixel-window factor per alm mode from an ell-indexed window."""

    ell = np.asarray(ell, dtype=int)
    arr = np.asarray(pixwin, dtype=float)
    if arr.ndim != 1:
        raise ValueError("pixwin must be a one-dimensional ell-indexed array.")
    required = int(ell.max()) + 1 if ell.size else 0
    if arr.size < required:
        raise ValueError(f"pixwin has length {arr.size}, but ell={required - 1} is required.")
    return arr[ell]


def shear2conv(g1, g2, lmax=None):
    """Convert shear maps to convergence using HEALPix spin transforms."""

    _require_torch_healpix()
    nside = hp.npix2nside(len(g1))
    gelm, _ = Map2AlmSpin.apply(g1, g2, lmax)
    lmax = hp.Alm.getlmax(len(gelm))
    ell, _ = hp.Alm.getlm(lmax)
    ell = torch.tensor(ell, dtype=torch.double)
    good_ls = ell > 1
    fac = torch.zeros_like(ell)
    good_ell = ell[good_ls]
    fac[good_ls] = -torch.sqrt(good_ell * (good_ell + 1) / ((good_ell + 2) * (good_ell - 1)))
    kelm = fac * gelm
    return Alm2Map.apply(kelm, nside, lmax)


def conv2shear(k, lmax=None, pixwin=None):
    """Convert convergence to shear maps using HEALPix spin transforms.

    ``pixwin`` is interpreted as an ell-indexed window with length at least
    ``lmax + 1`` and is expanded to alm mode order internally.
    """

    _require_torch_healpix()
    nside = hp.npix2nside(len(k))
    kelm = Map2Alm.apply(k, lmax)
    lmax = hp.Alm.getlmax(len(kelm))
    ell_np, _ = hp.Alm.getlm(lmax)
    if pixwin is not None:
        pixwinatell = _ell_indexed_window_values(pixwin, ell_np)
        kelm = kelm * torch.as_tensor(pixwinatell, dtype=torch.double, device=kelm.device)
    ell = torch.tensor(ell_np, dtype=torch.double)
    good_ls = ell > 0
    fac = torch.zeros_like(ell)
    good_ell = ell[good_ls]
    fac[good_ls] = -torch.sqrt((good_ell + 2) * (good_ell - 1) / (good_ell * (good_ell + 1)))
    gelm = fac * kelm
    gblm = torch.zeros_like(kelm)
    return Alm2MapSpin.apply(gelm, gblm, nside, lmax)


def degrade_karmmalike(k):
    """Legacy KARMMA-like downgrade helper."""

    _require_torch_healpix()
    nside = 256
    kelm = Map2Alm.apply(k, 3 * 256 - 1)
    lmax = hp.Alm.getlmax(len(kelm))
    kappa = Alm2Map.apply(kelm, nside, lmax)
    return kappa.numpy()
