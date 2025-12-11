Setting up RM-Gen
====================

For now, this package is not yet deployed on PyPi, and must be setup as local package. This means you need to
download and setup the code on https://github.com/artistrea/rm-gen to use the library.

==================================
Installing required packages
==================================

The project uses `uv`_ for managing its packages.
To setup virtual environment and install all packages run:

.. _uv: https://trimesh.org/trimesh.creation.html#trimesh.creation.triangulate_polygon

.. code-block:: bash

	uv sync

In case you want to try running the examples, you may also want to install the optional dependencies.
To do so, run:

.. code-block:: bash

	uv sync --all-groups

After that, you can activate the virtual environment with:

.. code-block:: bash

	. .venv/bin/activate  # for Windows it's .venv\Scripts\activate

And you are ready to go!

You can now try running the example notebooks in the `examples` folder.