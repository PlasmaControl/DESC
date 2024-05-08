Contributing to DESC
====================

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to DESC These are
mostly guidelines, not rules. Use your best judgment, and feel free to
propose changes to this document in a pull request.

Table Of Contents
^^^^^^^^^^^^^^^^^

* `I don’t want to read this whole thing, I just have a question!!! <#i-dont-want-to-read-this-whole-thing-i-just-have-a-question>`__

* `How Can I Contribute? <#how-can-i-contribute>`__

  - `Reporting Bugs <#reporting-bugs>`__
  - `Suggesting Enhancements <#suggesting-enhancements>`__
  - `Your First Code Contribution <#your-first-code-contribution>`__
  - `Pull Requests <#pull-requests>`__

* `Styleguides <#styleguides>`__

  - `Python <#python-styleguide>`__
  - `Git Commit Messages <#git-commit-messages>`__
  - `Documentation Styleguide <#documentation-styleguide>`__


I don’t want to read this whole thing I just have a question!!!
***************************************************************

If you just want to ask a question, the simplest method is to `create an issue
on github <https://github.com/PlasmaControl/DESC/issues/new>`__ and begin the
subject line with ``Question:`` That way it will be seen by all developers, and
the answer will be viewable by other users.

As the user base expands and more people start using and contributing to
the code, we may set up some sort of user group Slack or Discord or
other forum for questions and discussion among users/developers.

How Can I Contribute?
^^^^^^^^^^^^^^^^^^^^^

Reporting Bugs
**************

How Do I Submit A (Good) Bug Report?
------------------------------------

Bugs are tracked as `GitHub issues <https://github.com/PlasmaControl/DESC/issues/>`__.

Explain the problem and include additional details to help maintainers
reproduce the problem:

-  **Use a clear and descriptive title** for the issue to identify the
   problem.
-  **Describe the exact steps which reproduce the problem** in as many
   details as possible. When listing steps, *don’t just say what you did, but explain how you did it*.
-  **Provide specific examples to demonstrate the steps**. Include links
   to files or copy/pasteable snippets, which you use in those examples.
   If you’re providing snippets in the issue, use
   `Markdown code blocks <https://help.github.com/articles/markdown-basics/#multiple-lines>`__.
-  **Describe the behavior you observed after following the steps** and
   point out what exactly is the problem with that behavior.
-  **Explain which behavior you expected to see instead and why.**
-  **Include plots** of results that you believe to be wrong.
-  **If you’re reporting that DESC crashed**, include the python stack
   trace and full error message. Include the stack trace in the issue in
   a `code block <https://help.github.com/articles/markdown-basics/#multiple-lines>`__,
   a `file attachment <https://help.github.com/articles/file-attachments-on-issues-and-pull-requests/>`__,
   or put it in a `gist <https://gist.github.com/>`__ and provide link
   to that gist.

Provide more context by answering these questions:

-  **Did the problem start happening recently** (e.g. after updating to
   a new version of DESC) or was this always a problem?
-  If the problem started happening recently, **can you reproduce the problem in an older version of DESC?**
   What’s the most recent version in which the problem doesn’t happen?
-  **Can you reliably reproduce the issue?** If not, provide details
   about how often the problem happens and under which conditions it
   normally happens.

Include details about your configuration and environment:

-  **Which version of DESC are you using?**
-  **Which version of JAX (and other dependencies) are you using**
-  **What’s the name and version of the OS you’re using**?
-  **Are you running DESC in a virtual machine?** If so, which VM
   software are you using and which operating systems and versions are
   used for the host and the guest?
-  **Are you running DESC locally or on a cluster?** If so, which
   cluster, with which modules loaded?
-  **What hardware are you running on?** Which CPU, GPU, RAM, etc. If on
   a cluster, what resources are you allocating?

Suggesting Enhancements
***********************

This section guides you through submitting an enhancement suggestion for
DESC, including completely new features and minor improvements to
existing functionality.

Before creating enhancement suggestions, please check `this list <#before-submitting-an-enhancement-suggestion>`__
as you might find out that you don’t need to create one. When you are creating an
enhancement suggestion, please `include as many details as possible <#how-do-i-submit-a-good-enhancement-suggestion>`__,
including the steps that you imagine you would take if the feature you’re
requesting existed.

Before Submitting An Enhancement Suggestion
-------------------------------------------

-  `Check the documentation <https://desc-docs.readthedocs.io/en/latest/>`__
   for tips — you might discover that the enhancement is already available.
-  `Perform a cursory search <https://github.com/PlasmaControl/DESC/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement>`__
   to see if the enhancement has already been suggested. If it has, add
   a comment to the existing issue instead of opening a new one.

How Do I Submit A (Good) Enhancement Suggestion?
------------------------------------------------

Enhancement suggestions are tracked as `GitHub issues <https://guides.github.com/features/issues/>`__.
After you’ve followed the above steps and verified that an issue does not already
exist, create an issue and provide the following information:

-  **Use a clear and descriptive title** for the issue to identify the
   suggestion.
-  **Provide a step-by-step description of the suggested enhancement**
   in as many details as possible.
-  **Provide specific examples to demonstrate the steps**. Include
   copy/pasteable snippets which you use in those examples, as
   `Markdown code blocks <https://help.github.com/articles/markdown-basics/#multiple-lines>`__.
-  **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
-  **Explain why this enhancement would be useful** to other users.
-  **Provide references** (if relevant) to papers that discuss the
   physics behind the enhancement, or describe the enhancement in some
   way.

Your First Code Contribution
****************************

Unsure where to begin contributing to DESC? You can start by looking
through these ``good first issue`` and ``help wanted`` issues:

-  `Good first issues <https://github.com/PlasmaControl/DESC/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`__ - issues which should only require a few lines of code, and a test or two.
-  `Help wanted issues <https://github.com/PlasmaControl/DESC/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`__ - issues which should be a bit more involved than beginner issues.

Pull Requests
*************

Once you've made your changes on a local branch, `open a pull request <https://github.com/PlasmaControl/DESC/pulls>`_
on github. In the description, give a summary of what is being changed and why. Try to keep pull requests small and atomic,
with each PR focused on a adding or fixing a single thing. Large PRs will generally take much longer to review and approve.

Opening a PR will trigger a suite of tests and style/formatting checks that must pass before new code can be merged.
We also require approval from at least one (ideally multiple) of the main DESC developers, who may have suggested changes
or edits to your PR.

What if the ``test_compute_everything`` test fails, or there is a conflict in ``master_compute_data.pkl``?
----------------------------------------------------------------------------------------------------------
When the outputs of the compute quantities tested by the`test_compute_everything` [test](https://github.com/PlasmaControl/DESC/blob/master/tests/test_compute_funs.py) are changed in a PR, that test will fail.
The three main reasons this could occur are:

-  The PR was not intended to change how things are computed, but messed up something unexpected and now the compute quantities are incorrect, if you did not expect these changes in the PR then look into why these differences are happening and fix the PR.
-  The PR updated the way one of the existing compute index quantities are computed (either by a redefinition or perhaps fixing an error present in ``master``)
-  The PR added a new class parametrization (such as a new subclass of ``Curve`` like ``LinearCurve`` etc)

If the 2nd case is the reason, then you must update the ``master_compute_data.pkl`` file with the correct quantities being computed by your PR:

-  First, run the test with ``pytest tests -k test_compute_everything`` and inspect the compute quantities whose values are in error, to ensure that only the quantities you expect to be different are shown (and that the new values are indeed the correct ones, you should have a test elsewhere for that though).
-  If the values are as expected and only the expected compute quantities are different, then replace the block

.. code-block:: python

   except AssertionError as e:
      error = True
      print(e)

with

.. code-block:: python

   except AssertionError as e:
      error = False
      update_master_data = True
      print(e)


-  rerun the test ``pytest tests -k test_compute_everything``, now any compute quantity that is different between the PR and master will be updated with the PR value
-  ``git restore tests/test_compute_funs.py`` to remove the change you made to the test
-  ``git add tests/inputs/master_compute_data.pkl`` and commit to commit the new data file

If the 3rd case is the reason, then you must simply add the new parametrization to the ``test_compute_everything`` [test](https://github.com/PlasmaControl/DESC/blob/master/tests/test_compute_funs.py)

-  ``things`` dictionary with a sensible example instance of the class to use for the test, and
-  to the ``grid`` dictionary with a sensible default grid to use when computing the compute quantities for the new class
-  Then, rerunning the test  ``pytest tests -k test_compute_everything`` will add the compute quantities for the new class and save them to the ``.pkl`` file
-  ``git add tests/inputs/master_compute_data.pkl`` and commit to commit the new data file

Styleguides
^^^^^^^^^^^

Python Styleguide
*****************

-  `Follow the PEP8 format <https://www.python.org/dev/peps/pep-0008/>`__ where possible
-  Format code using `black <https://github.com/psf/black>`__ before committing - with formatting, consistency is better than "correctness." We use version ``22.10.0`` (there are small differences between versions). Install with ``pip install "black==22.10.0"``.
-  Check code with ``flake8``, settings are in ``setup.cfg``
-  We recommend installing ``pre-commit`` with ``pip install pre-commit`` and then running ``pre-commit install`` from the root of the repository. This will automatically run a number of checks every time you commit new code, reducing the likelihood of committing bad code.
-  -  Use `Numpy Style Docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`__ - see the code for plenty of examples. At a minimum, the docstring should include a description of inputs and outputs, and a short description of what the function or method does. Code snippets showing example usage strongly encouraged.
-  **Readability** and **usability** are more important than speed 99%
   of the time.
-  If it takes more than 30 seconds to understand what a line or block
   of code is doing, include a comment summarizing what it does.
-  If a function has more than ~5 inputs and/or return values, consider
   packaging them in a dictionary or custom class.
-  Make things modular. Focus on small functions that `do one thing and do it well <https://en.wikipedia.org/wiki/Unix_philosophy#Origin>`__,
   and then combine them together. Don’t try to shove everything into a
   single function.
-  *It’s not Fortran*! You are not limited to 6 character variable
   names. Please no variables or functions like ``ma00ab`` or
   ``psifac``. Make names descriptive and clear. If the name and meaning
   of a variable is not immediately apparent, the name is probably
   wrong.
-  Sometimes, a shorter, less descriptive name may make the code more
   readable. If you want to use an abbreviation or shorthand, include a
   comment with the keyword ``notation:`` explaining the notation at the
   beginning of the function or method explaining it, eg
   ``# notation: v = vartheta, straight field line poloidal angle in radians``.

``jnp`` vs ``np``
-----------------

DESC makes heavy use of the JAX library for accelerating code through
JIT compiling and automatic differentiation. JAX has a submodule,
``jax.numpy``, commonly abbreviated as ``jnp`` which offers an API
almost identical to ``numpy``.

-  If the function will ever be used for optimization (i.e., called as
   part of an objective function), use ``jnp``.
-  Similarly, if the function will need to be called multiple times and
   could benefit from JIT compiling, use ``jnp`` and ``jit``. However,
   in general it is best to only ``jit`` the outermost function, not
   each subfunction individually.
-  If the function will ever need to be differentiated through, use
   ``jnp`` and ``jacfwd``, ``jacrev``, or ``grad``.
-  If you are certain it will only ever be used during initialization or
   post processing (i.e. plotting), feel free to use ``np``, as it can be
   slightly faster without JIT compilation, and has fewer tricks
   necessary to make it work as expected.
-  If in doubt, ``jnp`` is usually a safe bet.
-  ``jax.numpy`` is *almost* a drop in replacement for ``numpy``, but
   there are some `subtle and important differences <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__.

``pytest``
----------

The testing suite in DESC is based on `pytest <https://docs.pytest.org/>`__, and makes use of several plugins for specialized testing. You can install all the necessary tools with ``pip install -r devtools/dev-requirements.txt``. You can run the tests from the root of the repository with ``pytest -m unit`` for unit tests or ``pytest -m regression`` for full regression tests (unit tests should take ~1hr on a standard laptop, regression tests may take several hours). To only run selected tests you can use ``pytest -k foo`` which will only run tests that have ``foo`` in the test or file name.

**Note**: when adding new tests to DESC, they **must** either be marked with ``@pytest.mark.unit`` or ``@pytest.mark.regression``, otherwise they will not be run as part of the automatic CI testing.

Additional useful flags include:

- ``--mpl`` tells pytest to also compare the output of plotting functions with saved baseline images in ``tests/baseline/`` using `pytest-mpl <https://pypi.org/project/pytest-mpl/>`__. These baseline images can be regenerated with ``pytest -k plotting --mpl-generate-path=tests/baseline/``.
- ``--cov`` will tell it to also report how much of the code is covered by tests using `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`__. A summary of the coverage is printed to the terminal at the end of the tests, and detailed information is saved to a ``.coverage`` file, which can then be turned into a simple HTML page with ``coverage html``. This will create a ``htmlcov/`` directory in the root of the repository that can be viewed in a browser to see line by line coverage.


`Git Commit Messages <https://chris.beams.io/posts/git-commit/>`__
*******************************************************************

-  A commit message template is included in the repository, ``.gitmessagetemplate``
-  You can set the template to be the default with ``git config commit.template .gitmessagetemplate``

Some helpful rules to follow (also included in the template):

-  Separate subject line from body with a single blank line.
-  Limit the subject line to 50 characters or less, and wrap body lines
   at 72 characters.
-  Capitalize the subject line.
-  Use the present tense (“Add feature” not “Added feature”) and the
   imperative mood (“Fix issue…” not “Fixes issue…”) in the subject
   line.
-  Reference issues and pull requests liberally in the body, including
   specific issue numbers. If the commit resolves an issue, note that at
   the bottom like ``Resolves: #123``.
-  Explain *what* and *why* vs *how*. Leave implementation details in
   the code. The commit message should be about what was changed and
   why.

Documentation Styleguide
************************

-  Use `SphinxDoc <https://www.sphinx-doc.org/en/master/index.html>`__.
-  Use `Numpy Style Docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`__.
-  Use `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__.
