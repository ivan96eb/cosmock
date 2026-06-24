# cosmock

Mock-field generation tools for cosmological kappa maps, implementing the
formalism and algorithm presented in 2411.04759.

## Installation

Install the base package for fitting, transforms, and spectrum conversion:

```bash
python -m pip install -e .
```

Install the HEALPix extra when using map-generation helpers in `cosmock.mocker`:

```bash
python -m pip install -e ".[healpix]"
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```
