Server configuration
--------------------

We can use a gitlab server as an external LFS server. To do that, we need to create a repository that will hold the files, for example: https://nextdesk1.ific.uv.es:4433/jmbenlloch/IC. There are 3 requisites for this repository:

- Gitlab must have LFS enabled (https://docs.gitlab.com/ee/workflow/lfs/lfs_administration.html)
- The repository must have public visibility if we want people to be able to checkout anonymously (this is also useful for travis).
- Every developer will need an account on this server with writing permissions for this repository to push commits (besides the their github account).


Client configuration
--------------------

To configure LFS for the repository every user must have git lfs installed on their machines. On debian-like linux can be done with:

    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs

For Mac OS X it is possible to install it using brew:

	brew install git-lfs

For other systems or alternatives downloads check this: https://git-lfs.github.com/

For git lfs to work there are some hooks that must be setup by running the following command on your repository dir:

    git lfs install

If you clone a repositoy which is already using lfs I don't think you need to run that command, but we can do more tests on that.

Setting up the external server
------------------------------
We need to add a `.lfsconfig` file with the address of our external lfs server:

```
[lfs]
    url = https://nextdesk1.ific.uv.es:4433/jmbenlloch/IC.git/info/lfs
```

Issues:

- So far we are using a self-signed certificate for the gitlab server, so unless we fix that (maybe https://letsencrypt.org/) we need to add this to `.bashrc`:

	`export GIT_SSL_NO_VERIFY=1`

- Git LFS does **NOT** support ssh yet (https://github.com/git-lfs/git-lfs/issues/1044). So developers will need to input their gitlab's user&pass after every commit.


Tracking files
--------------

To track files the command is:

	git lfs track '*.h5'

that will add a new line to `.gitattributes` like `*.h5 filter=lfs diff=lfs merge=lfs -text`


Migration
---------

In principle it is possible to migrate all the binary files in an existing repository using a tool provided by the git-lfs client:

	git lfs migrate import --include="*.h5" --include-ref=refs/heads/master

That will migrate all the h5 files from the master branch history and will rewrite all the commits. There is a caveat, though: since we are using an external server for lfs, the file .lfsconfig is needed to checkout the files and that file won't be present in older commits, so in principle it wouldn't work unless we put that file somehow in the past commits.

In any case, we could use LFS for the new files and keep the old ones in the repository if there is no simple solution for this issue.

You can find an already-migrated branch here: https://github.com/jmbenlloch/IC-1/tree/lfs-migrate

Travis is working well with that branch as you can see here: https://travis-ci.org/jmbenlloch/IC-1/jobs/425180907


Checking out a branch with lfs for the first time
-------------------------------------------------
The first time a user checkouts a branch using lfs, errors will appear becuase lfs won't be configured yet until he gets that branch.

The procedure to avoid that error is the following:

```
export GIT_LFS_SKIP_SMUDGE=1
export GIT_SSL_NO_VERIFY=1
checkout the branch (either with git or magit)
git lfs pull
```
