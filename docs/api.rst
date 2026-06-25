API reference
=============

The public API is intentionally small. Most users should import only these
two functions:

.. code-block:: python

   from cosmock import fit_parameters, generate_mocks

Public workflow
---------------

.. automodule:: cosmock.wrappers
   :members: fit_parameters, generate_mocks

Returned parameter object
-------------------------

.. automodule:: cosmock.structs
   :members: NonlinParameters

Transformation model
--------------------

.. automodule:: cosmock.Gn
   :members:

Fitting helpers
---------------

.. automodule:: cosmock.fitter
   :members:

Spectrum conversion helpers
---------------------------

.. automodule:: cosmock.Cls
   :members:

Mock-generation helpers
-----------------------

.. automodule:: cosmock.mocker
   :members:
