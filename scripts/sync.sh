#!/bin/sh
export INSTANCE="10.22.33.8"
echo $INSTANCE
scp -i instance/ubuntu-main.key -r ../[!.]* ubuntu@$INSTANCE:"/opt/sandbox/v.sandulescu/vol/replay"
#ssh -v -i instance/ubuntu-main.key -X ubuntu@$INSTANCE
#scp -i instance/ubuntu-main.key -r ubuntu@$INSTANCE:"/opt/sandbox/v.sandulescu/vol/cross-device/data/common_dataset.csv" cross-device/data/
#scp -i instance/ubuntu-main.key -r ubuntu@$INSTANCE:'/opt/sandbox/v.sandulescu/vol/storage4' ppas-bid-landscaping/storage
#scp -i instance/ubuntu-main.key -r cross-device/data/seznam_test_dataset.csv ubuntu@$INSTANCE:"/vol/cross-device/data"
