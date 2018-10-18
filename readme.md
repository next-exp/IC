# How to migrate your branches to IC's LFS-enabled `master` branch

1. Install the git lfs client: https://git-lfs.github.com/

2. Download the migration scripts

   `git clone https://github.com/nextic/IC.git --single-branch --branch lfs-migration-scripts <destination dir>`

3. `cd <destination dir>`

4. `bash create-migration-clone.sh <IC migration clone destination dir>`

5. Any branch that you have been working on and will want to merge to
   `nextic` needs to be migrated to LFS. The migration scipts will
   fetch such branches from a `nextic` fork. You must push **all** the
   work you have done on such branches, to your fork *before* using
   `migrate-branch-to-lfs.sh`. In other words make sure that the branch in your fork
   is not behind the corresponding branch in your local repository.

6. Add your fork as a remote in the clone you placed into `<IC
   migration clone destination dir>` in step 4 above.

7. `bash migrate-branch-to-lfs.sh <migration repo> <remote> <branch>`

   where

   + `<migration repo>` is the clone you created in step 4 above.

   + `<remote>` is the one you added in step 6 above.

   + `<branch>` is the name of a branch on `<remote>` that you want to migrate to LFS.

8. If the process succeeds, a new branch with the name `LFS.<branch>`
   will be created in `<migration repo>`. Once you are satisfied that
   this branch is OK, you can push it to your fork.
