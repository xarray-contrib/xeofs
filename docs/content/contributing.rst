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

Using the commands below, prepare your environment:

.. code-block:: bash

    conda create -n xeofs python=3.11 rpy2 pandoc
    conda activate xeofs
    pip install -e .[docs,dev]

This will install all necessary dependencies, including those for development and documentation. If you're only updating the code (without modifying online documentation), you can skip the docs dependency:

.. code-block:: bash

    pip install -e .[dev]

On the other hand, if you're just updating documentation:

.. code-block:: bash

    pip install -e .[docs]

Additionally, install the pre-commit hooks:

.. code-block:: bash

    pre-commit install

If you've completed Steps 1 and 2, you can move directly to Step 4.

3. Update your local copy
-------------------------

Before diving into your contribution, ensure your local main branch is updated:

.. code-block:: bash

    git checkout main
    git fetch upstream
    git merge upstream/main

This syncs your local main branch with the latest from the primary `xeofs` repository.

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
    git commit -m "concise commit message"


.. note::
    We use the `conventional commit`_ format for commit messages in ``xeofs``. 
    This format helps us automatically release new versions. Key points to note:

    - Use **fix:** prefix for **bug fixes**. This will trigger a patch release.
    - Use **feat:** prefix for **new features**. This will initiate a minor release.



6. Test your changes
--------------------

It's essential to test any modifications to ensure compatibility with the existing code. Run the following test from the repository's root directory:

.. code-block:: bash

    pytest

If you introduce a new feature or function, please also add corresponding tests in the `tests` directory.

7. Submit a pull request
------------------------

Once satisfied with your changes, push them to your GitHub fork:

.. code-block:: bash

    git push origin my-new-feature


Then, on your GitHub fork page, select "Compare & pull request" to initiate the pull request.


.. _convential commit: https://www.conventionalcommits.org/en/v1.0.0/
.. _xarray Contributing Guide: https://docs.xarray.dev/en/stable/contributing.html#version-control-git-and-github

