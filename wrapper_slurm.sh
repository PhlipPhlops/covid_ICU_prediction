#!/bin/bash -l
export PYTHONPATH=$PYTHONPATH:/home/pkinne2/CovidICUPrediction
env
scl enable rh-python36 'python cast_notes_to_glove.py'
