How to create a good commit history
===================================

Commit structure
-----------------

- Focus on a specific change. Don't try to change everything at once.
  If the plan is to implement a series of things, do them one by one.

- Do not mix changes concerning different features in one commit.
  For instance, do not mix cosmetical changes with other, more relevant ones.

- Commit frequently and in an incremental manner.
  Do not be afraid of making too many commits, they can be combined afterwards
  (this is known as squashing in git).
  Besides, the history can be easily cleaned when the changes are small and independent of each other.


Commit style
--------------

- Just follow https://chris.beams.io/posts/git-commit/

This is a summary of a good commit message.
But, please at least take a look to the details in the link above.


     | The seven rules of a great Git commit message
     |
     | 1. Separate subject from body with a blank line
     | 2. Limit the subject line to 50 characters
     | 3. Capitalize the subject line
     | 4. Do not end the subject line with a period
     | 5. Use the imperative mood in the subject line
     | 6. Wrap the body at 72 characters
     | 7. Use the body to explain what and why vs. how

Commits after PR review
-----------------------

- Remember that these guidelines apply also to changes requested during a review process.
  Don't make commits like ``Changes requested by the reviewer``.

- Follow the same guidelines listed above and,
  if you want to refer to the discussion happening in the PR,
  do it using the second part of the commit message.
  You may also refer to the PR by using ``#PR_NUMBER`` or
  by posting a link to a specific comment in the discussion.
