# Developing Petastorm

Please follow the following instructions to develop Petastorm:

For tensorflow without GPU:
```virtualenv env
source env/bin/activate
pip install -U pip
pip install -e .[opencv,tf,test,torch]
```

For tensorflow with GPU:
```virtualenv env
source env/bin/activate
pip install -U pip
pip install -e .[opencv,tf_gpu,test,torch]
```

To run tests, please run the following: ```pytest```


# Petastorm Auto-generated Documentation

The petastorm project uses sphinx autodoc capabilities, along with free
documentation hosting by ReadTheDocs.org (RTD), to serve up auto-generated API
docs on http://petastorm.rtfd.io .

The project site is [here](https://readthedocs.org/projects/petastorm/), and
is configured via webhooks to trigger sphinx doc builds from changes in the 
petastorm github repo.  Documents are configured to build the same locally or
on RTD.

All the source files needed to generate the autodocs reside under ``docs/``.

To make documents locally:

```bash
pip install -e .[docs]
cd docs

# To nuke all generated HTMLs
make clean

# Each run incrementally updates HTML based on file changes
make html
```

Once the HTML build process completes successfully, naviate your browser to
`file:///<petastorm-repo-path>/docs/_build/html/index.html`.

Some changes may require build and deployment to see, including:
* Changes to `readthedocs.yml`
* Changes to `docs/conf.py`
* A change that makes RTD build different from a local build

To see the above documentation changes:
1. One needs to create a petastorm branch and push it
2. Then configure RTD to activate a [version](https://readthedocs.org/dashboard/petastorm/versions/) for that branch
3. A project maintainer will need to effect such version activation
4. The status of a [built version](https://readthedocs.org/projects/petastorm/versions/), as well as the resulting docs, can then be viewed

## Release Versions

By default, RTD defines the `latest` version, which can be pointed at master
or another branch.  Additionally, each release may have an associated RTD build
version, which must be explicitly activated in the
[Versions settings page](https://readthedocs.org/dashboard/petastorm/versions/).

As with any source file, once a release is tagged, it is essentially immutable,
so be sure that all the desired documentation changes are in place before
tagging a release.

Note that `conf.py` defines a `release` and `version` property.  For ease of
maintenance, we've set that to be the same version string as defined in
`petastorm/__init__.py`.

## Known Doc Build Caveats and Issues

* Due to RTD's [build resource limitations](https://docs.readthedocs.io/en/latest/builds.html),
we are unable to pip install any of the petastorm extra-required library packages.
* Since Sphinx must be able to load a python module to read its docstrings,
the doc page for any module that imports `cv2`, `tensorflow`, or `torch` will,
unfortunately, fail to build.
* The `alabaster` Sphinx theme defaults to using `travis-ci.org` for the
Travis CI build badge, whereas the petastorm project is served on `.com`, so
we don't currently have a working Travis CI build status.
