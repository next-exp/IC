#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: migrate-branch-to-lfs.sh repo_folder remote_name branch_name"
	exit
fi

IFS=$'\n'

REPO=$1
REMOTE=$2
BRANCH=$3

cd $REPO
MERGEBASE=`git merge-base origin/pre-lfs-master $REMOTE/$BRANCH`

echo MERGEBASE: $MERGEBASE

DATE=`git show -s --format=%ci $MERGEBASE`
SUBJECT=`git show -s --format=%s $MERGEBASE`


for COMMIT in `git rev-list origin/lfs-migration-master`; do
	NEWDATE=`git show -s --format=%ci $COMMIT`
	
	if [ $NEWDATE != $DATE ]; then
		continue
	fi

	NEWSUBJECT=`git show -s --format=%s $COMMIT`
	if [ $NEWSUBJECT == $SUBJECT ]; then
		NEWMERGEBASE=$COMMIT
		echo $DATE, $SUBJECT, $MERGEBASE
		echo $NEWDATE, $NEWSUBJECT, $NEWMERGEBASE
		break
	fi
done


FOUND=""
for COMMIT in `git rev-list $REMOTE/$BRANCH`; do
	PARENT=`git show -s --format=%P $COMMIT`
	if [ $PARENT == $MERGEBASE ]; then
		FOUND=1
		MERGEBASECHILD=$COMMIT
		break
	fi
done

if [ -z "$FOUND" ]; then
	echo "Did not find cherry pick range"
	exit
fi


echo NEWMERGEBASE:   $NEWMERGEBASE
echo MERGEBASECHILD: $MERGEBASECHILD

git checkout -b LFS."$BRANCH" $NEWMERGEBASE

BRANCHHASH=`git show -s --format=%H $REMOTE/$BRANCH`
echo BRANCHHASH: $BRANCHHASH


echo range: $MERGEBASECHILD, $BRANCHHASH


if which tac; then
        TOCHERRYPICK=`git rev-list $BRANCHHASH | sed "/$MERGEBASECHILD/q" | tac`
              else
        TOCHERRYPICK=`git rev-list $BRANCHHASH | sed "/$MERGEBASECHILD/q" | tail -r`
fi




echo $TOCHERRYPICK

for COMMIT in $TOCHERRYPICK; do
	echo $COMMIT
	if git cherry-pick $COMMIT; then
		git diff --name-only | xargs git add
		git commit --amend --no-edit --
	else
		for FILE in `git diff --name-only`; do
			git show $COMMIT:$FILE > $FILE
			git add $FILE
		done
		git -c core.editor=true cherry-pick --continue
	fi
done

