#!/bin/sh
export INSTANCE="10.22.33.8"
echo $INSTANCE
scp -i instance/ubuntu-main.key -r ubuntu@$INSTANCE:"/opt/sandbox/v.sandulescu/vol/replay/experiments/results*.csv" ../experiments/
