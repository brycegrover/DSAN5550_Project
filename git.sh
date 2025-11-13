#!/bin/bash

printf 'Would you like to sync with the github server (y/n)? '
read answer
if [ "$answer" != "${answer#[Yy]}" ] ; then

    git pull
    
    read -p 'Enter commit message: ' message
    git add .
    git commit -m "$message"
    git push

else
    echo "Not syncing to GitHub."
fi
