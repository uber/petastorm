.. inclusion-marker-start-do-not-remove

====================
Developing petastorm
====================

Please follow the following instructions to develop Petastorm:

For tensorflow without GPU:

.. code-block:: bash

    virtualenv env
    source env/bin/activate
    pip install -U pip
    pip install -e .[opencv,tf,test,torch]

For tensorflow with GPU:

.. code-block:: bash

    virtualenv env
    source env/bin/activate
    pip install -U pip
    pip install -e .[opencv,tf_gpu,test,torch]

To run tests, please run the following:

.. code-block:: bash

    pytest -v


Petastorm auto-generated documentation
======================================

The petastorm project uses sphinx autodoc capabilities, along with free
documentation hosting by ReadTheDocs.org (RTD), to serve up auto-generated API
docs on http://petastorm.rtfd.io .

The `RTD site`_ is configured via webhooks to trigger sphinx doc builds from
changes in the petastorm github repo.  Documents are configured to build the
same locally or on RTD.

All the source files needed to generate the autodocs reside under ``docs/autodoc/``.

To make documents locally:

.. code-block:: bash

    pip install -e .[docs]
    cd docs/autodoc

    # To nuke all generated HTMLs
    make clean

    # Each run incrementally updates HTML based on file changes
    make html

Once the HTML build process completes successfully, naviate your browser to
``file:///<petastorm-repo-path>/docs/autodoc/_build/html/index.html``.

Some changes may require build and deployment to see, including:

* Changes to ``readthedocs.yml``
* Changes to ``docs/autodoc/conf.py``
* A change that makes RTD build different from a local build

To see the above documentation changes:

1. One needs to create a petastorm branch and push it
2. Then configure RTD to activate a version_ for that branch
3. A project maintainer will need to effect such version activation
4. The status of a `built version`_, as well as the resulting docs, can then be viewed

.. _RTD site: https://readthedocs.org/projects/petastorm/
.. _version: https://readthedocs.org/dashboard/petastorm/versions/
.. _built version: https://readthedocs.org/projects/petastorm/versions/

Release versions
----------------

By default, RTD defines the ``latest`` version, which can be pointed at master
or another branch.  Additionally, each release may have an associated RTD build
version, which must be explicitly activated in the
`Versions settings page <https://readthedocs.org/dashboard/petastorm/versions/>`_.

As with any source file, once a release is tagged, it is essentially immutable,
so be sure that all the desired documentation changes are in place before
tagging a release.

Note that ``conf.py`` defines a ``release`` and ``version`` property.  For ease
of maintenance, we've set that to be the same version string as defined in
``petastorm/__init__.py``.

Known doc-build caveats and issues
----------------------------------

* Due to RTD's `build resource limitations`_, we are unable to pip install any
  of the petastorm extra-required library packages.
* Since Sphinx must be able to load a python module to read its docstrings,
  the doc page for any module that imports ``cv2``, ``tensorflow``, or
  ``torch`` will, unfortunately, fail to build.
* The `alabaster Sphinx theme`_ defaults to using ``travis-ci.org`` for the
  Travis CI build badge, whereas the petastorm project is served on ``.com``,
  so we don't currently have a working Travis CI build status.

.. _build resource limitations: https://docs.readthedocs.io/en/latest/builds.html
.. _alabaster Sphinx theme: https://alabaster.readthedocs.io/

Future: auto-generate with ``sphinx-apidoc``
--------------------------------------------

Sphinx has the ability to auto-generate the entire API, either via the
autosummary_ extension, or the ``sphinx-apidoc`` tool.

The following ``sphinx-apidoc`` invocation will autogenerate an ``api/``
subdirectory of rST files for each of the petastorm modules.  Those files can
then be glob'd into a TOC tree.

.. code-block:: bash

  cd docs/autodocs
  sphinx-apidoc -fTo api ../.. ../../setup.py

The ``apidoc_experiment`` branch_ and `RTD output`_ demonstrates the outcome of
vanilla usage.  Actually leveraging this approach to produce uncluttered
auto-generated API doc will require:

1. Code package reorganization
2. Experimentation with sphinx settings, if available, to shorten link names
3. Configuration change to auto-run ``sphinx-apidoc`` in RTD build, as opposed
   to committing the ``api/*.rst`` files

.. _autosummary: http://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
.. _branch: https://github.com/uber/petastorm/compare/apidoc_experiment
.. _RTD output: https://petastorm.readthedocs.io/en/apidoc_experiment


.. inclusion-marker-end-do-not-remove

Need to upgrade from petastorm pre-0.3.0?
=========================================

.. toctree::

   migrating-0.3.0
