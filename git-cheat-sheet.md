# Git Commands:

## See branches:

git branch

## Checkout new branch:

git checkout -b <branchName> -- Remember to be in master

## Add branch remotely

git push origin -u <branchName>

## Check out existing branch:

git checkout <branchName>

## Delete branch locally:

git branch -d <branchName>

## Delete remotely

git push origin --delete <branchName>

## Pull specific branch from remote and create one locally:

git checkout -t origin/<branchName>

## Committing with a message:

git commit -m ""

## Adding to previous commit, used for small fixes:

git commit --fixup HEAD/commit

## Checking number of commits in current working branch
git rev-list master.. --count