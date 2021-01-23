.. inclusion-marker-start-do-not-remove

Developer guide
===============

Setting up development environment (recommendation)
---------------------------------------------------
This section shows a way to configure a development environment that allows you to run tests and build documentation.

.. code-block:: bash

    virtualenv env
    source env/bin/activate
    pip install -U pip setuptools
    pip install -e .[opencv,tf,test,torch]

Additionally, you can use the Dockerized Linux workspace via the Makefile provided at `docker/Makefile`. The following
will build the Docker image, start a running container with petastorm source mounted into it from the host,
and open a BASH shell into it (you must have GNU Make and Docker installed beforehand):

.. code-block:: bash

    make build run shell

Within the Dockerized workspace, you can find the Python virtual environments at `/petastorm_venv2.7` and `/petastorm_venv3.6`,
and the local `petastorm/` mounted at `/petastorm`. Remember to set python for pyspark correctly after load virtual env, for example:

.. code-block:: bash

    export PYSPARK_PYTHON=`which python3`


Also, if you are seeing "ImportError: libGL.so.1"from "import cv2", update with running
"apt-get update; apt-get install ffmpeg libsm6 libxext6  -y"
(reference: https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-share).

Unit tests
----------
To run unit tests:

.. code-block:: bash

    pytest -v petastorm

NOTE: you need to have Java 1.8 to be installed for the test to pass (it's a dependency of Spark)

``pytest`` has mulitple useful plugins. Consider installing the following plugins:

.. code-block:: bash

    pip install pytest-xdist pytest-repeat pytest-pycharm

which enable you to run tests in parallel (``-n`` switch) and repeat tests multiple times (``--count`` switch)

Caching test datasets
^^^^^^^^^^^^^^^^^^^^^
Some unit tests rely on mock data. Generating these datasets is not very fast as it spins up local Spark isntance.
Use ``-Y`` switch to cache these datasets. Be careful, as the dataset generation exercises Petastorm code, hence
in some cases you would need to invalidate the cache for the test to take all code changes into account.
Use ``--cache-clear`` switch to do so.

Documentation
-------------

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
``file:///tmp/autodocs/_build/html/index.html``.

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
^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Release procedure
-----------------
1. Make sure you are on the latest mater in your local workspace (``git checkout master && git pull``).
2. Update ``__version__`` in ``petastorm/__init__.py`` and commit.
3. Update ``docs/release-notes.rst``.

   a. Delete ``(unreleased)`` from the release we are about to release.
   b. Add any additional information if needed.
   c. Add kudos message to any new contributors who contributed to the release.
   d. Create a future release entry and tag it with ``(unreleased))`` string.

4. Commit the changes.
5. Tag as ``vX.X.Xrc0`` (``git tag vX.X.Xrc0``) and push both master and the tag (``git push origin master vX.X.Xrc0``). This will trigger build and pypi release.
6. Provide an opportunity for users to test the new release (slack channel/tweater). Create new release candidates as needed.
7. Tag as ``vX.X.X`` (``git tag vX.X.X``) and push both master and the tag (``git push origin master vX.X.X``). This will trigger build and pypi release
8. Once the build finishes, a new python wheel will be pushed to public pypi server.
9. Navigate to https://readthedocs.org/ --> "My Projects" --> "Builds" --> Trigger build of the 'latest' documentation (not clear when RTD picks up new tags from github, so you may see only outdated release versions there).

.. inclusion-marker-end-do-not-remove


Setting up pyspark for working with S3 locally
----------------------------------------------
Checked these instructions for pyspark 3.0.1
1. Download the following files into some local directory:

    a. https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.7.4/aws-java-sdk-1.7.4.jar
    b. https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/2.7.4/hadoop-aws-2.7.4.jar
    c. https://repo1.maven.org/maven2/net/java/dev/jets3t/jets3t/0.9.4/jets3t-0.9.4.jar (was not able to confirm s3 protocol due to authentication issues)

2. Add/set ``CLASSPATH`` environment variable to point to the directory containing these jars.