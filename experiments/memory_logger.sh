#!/bin/bash

# Continuously log memory usage of the system and sleep 30 seconds in between
# each measurement.
while true
do
    free -h >> memory.log
    sleep 30
done