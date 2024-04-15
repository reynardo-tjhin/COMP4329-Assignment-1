#!/bin/bash

# remove any unecessary temporary files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$|instance)" | xargs rm -rf
rm -rf temp