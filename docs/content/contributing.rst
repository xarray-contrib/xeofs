Contributing
============

Introduction
------------
Looking to fix a bug, suggest an improvement, or even add a new feature? You're in the right place! We welcome all contributions, regardless of size or complexity.

This guide will walk you through the process step-by-step. Here's what we'll cover:

1. Creating a local copy of the repository
2. Setting up your development environment
3. Updating your local copy
4. Creating a new branch
5. Making your changes
6. Testing your changes
7. Submitting a pull request


.. note:: 
    
    Steps 1 and 2 only need to be done once, but starting from Step 3, you should repeat them for every new contribution.

For this guide, we assume you have a basic understanding of version control, Git, and GitHub. For a more in-depth look, consider the `xarray Contributing Guide`_ and its references.



1. Create a local copy of the repository
----------------------------------------

To work on the code, start by creating your own fork. Visit the xeofs project page and click the 'Fork' button at the top. This action copies the code to your GitHub account.

Next, clone your fork to your computer:

.. code-block:: bash

    git clone https://github.com/your-user-name/xeofs.git
    cd xeofs
    git remote add upstream https://github.com/xarray-contrib/xeofs.git

This command sequence creates a local directory named *xeofs* and links your version to the primary xeofs project.

2. Set up your development environment
--------------------------------------

Install `Pixi <https://pixi.sh/latest/installation/>`__, then prepare your
environment from the repository root:

.. code-block:: bash

    pixi install --locked

The default environment installs Python 3.11, an editable copy of ``xeofs``,
development tools, and the ``complete`` optional dependencies. Pixi manages
the environment in ``.pixi/``; no separate Conda environment or activation is
needed. Use ``pixi run`` to execute commands, or ``pixi shell`` for an
interactive shell.

The dependency declarations remain in ``pyproject.toml`` and the resolved
versions for Linux, macOS, and Windows are recorded in ``pixi.lock``. After
changing dependencies, run ``pixi lock`` and commit both files. To update
resolved packages without changing the requirements, run ``pixi update``.

Other environments are available for specific tasks:

.. code-block:: bash

    pixi run -e minimal-py311 test
    pixi run -e minimal-py312 test
    pixi run -e complete-py312 test
    pixi run -e minimal-py313 test
    pixi run -e complete-py313 test
    pixi run -e lint lint
    pixi run -e lint format-check
    pixi run -e docs docs
    pixi run build

The documentation environment includes R, rpy2, and Pandoc and is available
on Linux and macOS. The other locked environments also support Windows.

CI additionally tests Python 3.11 with the lowest supported PyPI dependencies.
Because Pixi does not expose PyPI's ``lowest-direct`` resolution strategy,
this check uses ``uv`` inside a separate Pixi-managed environment:

.. code-block:: bash

    pixi run -e lowest install-lowest '.[dev]'
    pixi run -e lowest coverage
    pixi run -e lowest install-lowest '.[dev,complete]'
    pixi run -e lowest coverage --doctest-glob=README.md

These compatibility-check dependencies are intentionally resolved afresh,
not locked. To switch back from complete to minimal dependencies, remove
only ``.pixi/envs/lowest`` first so optional packages do not leak into the
minimal check.

Additionally, install the pre-commit hooks:

.. code-block:: bash

    pixi run pre-commit install
    pixi run pre-commit install --hook-type commit-msg

If you've completed Steps 1 and 2, you can move directly to Step 4.

3. Update your local copy
-------------------------

Before diving into your contribution, ensure your local main branch is updated:

.. code-block:: bash

    git checkout main
    git fetch upstream
    git merge upstream/main

This syncs your local main branch with the latest from the primary ``xeofs`` repository.

4. Create a new branch
----------------------

For your new contribution, initiate a separate branch. Ensure your branch name reflects the essence of your contribution:

.. code-block:: bash

    git checkout -b my-new-feature

5. Make your changes
--------------------

After making your updates, remember to commit them:

.. code-block:: bash

    git add .
    git commit -m "fix: concise description of the bug fix"


.. note::
    We use the `conventional commit`_ format for commit messages in ``xeofs``. 
    This format helps us automatically release new versions. Key points to note:

    - Use **fix:** prefix for **bug fixes**. This will trigger a patch release.
    - Use **feat:** prefix for **new features**. This will initiate a minor release.
    - Add **!** for **breaking changes**, for example ``feat!: change the public API``.
      This will trigger a major release.
    - Use **chore:** for maintenance that should not trigger a release.

    Releases run automatically after merging onto ``main``. Use the same
    conventional commit format for the pull request title when squash-merging.
    Python Semantic Release continues to update ``project.version`` in
    ``pyproject.toml`` and build the wheel and source distribution with
    setuptools. Publishing does not depend on Pixi or its development lockfile.



6. Test your changes
--------------------

It's essential to test any modifications to ensure compatibility with the existing code. Run the following test from the repository's root directory:

.. code-block:: bash

    pixi run test
    pixi run test-readme

If you introduce a new feature or function, please also add corresponding tests in the `tests` directory.

7. Submit a pull request
------------------------

Once satisfied with your changes, push them to your GitHub fork:

.. code-block:: bash

    git push origin my-new-feature


Then, on your GitHub fork page, select "Compare & pull request" to initiate the pull request.


.. _conventional commit: https://www.conventionalcommits.org/en/v1.0.0/
.. _xarray Contributing Guide: https://docs.xarray.dev/en/stable/contributing.html#version-control-git-and-github
