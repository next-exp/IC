How to create a good commit history
===================================

A good commit history should make it easy to understand how and why a change was
introduced. Each commit should tell a clear, reviewable story.


Commit structure
----------------

- Make each commit focused.

  A commit should contain one logical change. Avoid grouping unrelated changes
  together just because they were made around the same time.

  Good examples:

  ::

      Add new normalization strategy

  ::

      Refactor corona algorithm

  ::

      Fix bug in hit-voxel association

  Bad examples:

  ::

      Update stuff

  ::

      Address reviewer comments

  ::

      Refactor code and add new feature

- Do not mix unrelated types of changes.

  For example, avoid combining formatting changes, renames, refactors, bug
  fixes, and feature work in the same commit unless they are part of the same
  logical change.

  Prefer this:

  ::

     Reorder function parameters

  ::

     Rename interface for energy corrections

  ::

     Add fallback for missing data

  Instead of this:

  ::

     Clean up correction application and add default fallback

- Commit incrementally while working.

  It is fine to create small work-in-progress commits locally. They help you
  save progress and make changes easier to reorganize later. This applies both
  to the process of developing a new feature on your own and while applying the
  reviewer feedback during a PR review.

  Before merging, however, clean up the history so that the final commits are
  meaningful, focused, and easy to review. Small independent commits can be
  reordered, edited, squashed, or split more easily than large mixed commits.

- Each commit should leave the project in a reasonable state.

  As much as possible, every commit should build, pass tests, and make sense on
  its own. This makes debugging, bisecting, reverting, and reviewing easier.


Commit messages
---------------

We follow `this <https://chris.beams.io/posts/git-commit/>`__ guideline. In short,
this is a summary of a good commit message. But, please at least take a look to
the details in the link above.

     | The seven rules of a great Git commit message
     |
     | 1. Separate subject from body with a blank line
     | 2. Limit the subject line to 50 characters
     | 3. Capitalize the subject line
     | 4. Do not end the subject line with a period
     | 5. Use the imperative mood in the subject line
     | 6. Wrap the body at 72 characters
     | 7. Use the body to explain what and why vs. how


Or in a bit more detail...

Write commit messages that explain the change clearly.

Use this format::

    Short imperative summary

    Optional longer explanation of what changed and why.
    Include context that is not obvious from the diff.

Guidelines for the subject line:

- Use the imperative mood.

  Good::

      Add validation for PSF arguments

  Bad::

      Added validation for PSF arguments

  ::

      Adds validation for PSF arguments

- Keep it short and specific.

  Aim for about 50 characters when possible.

- Capitalize the subject line.

- Do not end the subject line with a period.

Guidelines for the body:

- Separate the subject from the body with a blank line.
- Wrap the body at about 72 characters.
- Explain what changed and why.
- Avoid describing only how the code changed; the diff already shows that.
- Mention relevant trade-offs, assumptions, bugs, tickets, or PR discussions.

Example::

    Fix bug in hit-voxel association

    In the paolina algorithm, some hits were being assigned to the wrong voxel.
    as reported in (issue) #1234.
    This is fixed by increasing the voxel size by a tiny amount, which is enough
    to avoid floating-point discrepancies.


Commits after PR review
-----------------------

These guidelines also apply to changes made during review.

Avoid commits like:

  ::

     Fix review comments

  ::

     Apply feedback

  ::

     Changes requested by reviewer

Instead, describe the actual change:

  ::

     Rename function for clarity

  ::

     Add tests for module X

  ::

     Simplify logic of whatever

If the commit needs to refer to a PR discussion, include that context in the
commit body::

    Simplify logic of whatever

    The amount of cases that had to be handled can be grouped into a few
    well-defined groups. This version is much more easy to interpret. The
    grouping choice was discussed in detail in #1234.

Cleaning up before merge
------------------------

Before merging a branch, review the commit history from the perspective of
someone reading it later. Imagine that you will have to understand your own code
10 years into the future and your only information is the commit history. You
want to make it as detailed as necessary and as concise as possible.

Ask yourself:

- Does each commit represent one logical change?
- Are unrelated changes split apart?
- Are temporary commits removed or squashed?
- Do commit messages explain the purpose of the change?
- Would it be easy to revert or debug one commit independently?

A clean history does not mean having the fewest possible commits. It means
having commits that are useful, understandable, and intentional.
