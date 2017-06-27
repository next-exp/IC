How to use notebooks in IC
==========================

The problem with notebooks
--------------------------

Notebooks are as very useful tool for prototyping and preparing analysis, but
they can create a lot of conflicts under a version control system (like
git). Usually this is due to the outputs, which can change from one execution to
another (simply the execution counter of each notebook cell can give
conflicts).

To solve those issues we will only put into git notebooks without output. We use
`git filters 
<http://pascalbugnion.net/blog/ipython-notebooks-and-git.html>`_ to automatize
this process. This solution requires a filter script (``ipynb_drop_output``) and
some git configuration: the files ``.gitattributes`` and ``.gitconfig`` already
included in the repository and a small change in ``.git/config`` which can be
done with this command:

``git config --add include.path $ICTDIR/.gitconfig``


Another source of troubles are filenames, since each developer can have different
paths and this will require changes in the code to be able to read them. The
easiest way to avoid this conflicts is by following a naming convention, in our
case **all files must be under $IC_DATA**. Each user have to define ``$IC_DATA``
to some folder and put the files needed in the notebook there.

Workflow
--------

1. First thing you will need to do is to configure your environment,  this can
   be done as usual: ``source manage.sh  work_in_python_version 3.6``.

2. Define your ``$IC_DATA`` if you don't have it yet.

3. Create/checkout your development branch for the notebook (see step 1 in
   :doc:`contributing`)

4. Once you have it, you can simply run ``jupyter notebook`` and get started
   working on your new notebook. Remember to use ``$IC_DATA`` for your input
   files in case you need them.

5. You can commit your work and push it to you fork as many times you want, but
   keep in mind that the commited version will not have the output as they will
   pass through the stripping filter.

6. In principle that should be enough, if you simply run a notebook without
   changing the code, no changes should be detected by git. But they will
   usually appear if you run ``git status``, for example:
   
::

 $ git status
 On branch notebooks
 Changes not staged for commit:
   (use "git add <file>..." to update what will be committed)
   (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   ../invisible_cities/cities/diomira.ipynb


If you try to see the changes, you won't be able to see them, ``git diff`` will
not show any output. `This has been reported by other git users
<http://stackoverflow.com/questions/19807979/why-does-git-status-ignore-the-gitattributes-clean-filter>`_. To
solve it, git index has to be updated, the easiest way to do this is running
``git add notebook.ipynb``.


In magit, the notebook won't appear in ``magit-status`` but if you try to change
to another branch, rebase, etc. git will complain:


``GitError! The following untracked working tree files would be overwritten by
checkout:  [Type `$' for details]``

By pressing ``$`` you can see the error:

::  

   1 git … checkout master
    error: Your local changes to the following files would be overwritten by checkout:
	invisible_cities/cities/diomira.ipynb
   Please, commit your changes or stash them before you can switch branches.

You can add the file typing ``s`` for the file stage menu and then putting the
notebook giving the conflict. After that git won't detect the changes and you
will be able to change branch or whatever you need.
