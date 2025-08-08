#!/bin/bash
set -e
qsub jobs/J0001.sh
sleep 1
qsub jobs/J0002.sh
sleep 1
qsub jobs/J0003.sh
sleep 1
qsub jobs/J0004.sh
sleep 1
qsub jobs/J0005.sh
sleep 1
