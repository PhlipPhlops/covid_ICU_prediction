#!/bin/bash

### Name the project in the batch job queue
#SBATCH -J PhiTimeseries 
##SBATCH --comment='Casting Covid19 unstructured notes into GloVe embedding'

### You must specify 'gpu' or another partition to have access to the system GPUs.
#SBATCH -p gpu
#SBATCH -G 1

### (REQUIRED) if you don't want your job to end after 8 hours!
#SBATCH -t 5:0:0

### (optional) Output and error file definitions. To have all output in a file named
#SBATCH -o /home/pkinne2/cluster-logs/rnn.out
#SBATCH -e /home/pkinne2/cluster-logs/rnn.err

## (REQUIRED) RAM
#SBATCH --mem 24G

### Request 1 node - Only specify a value other than 1 for this option when you know that
###                  your code will run across multiple systems concurrently. Otherwise
###                  you're just wasting resources that could be used by others.
#SBATCH -N 1

### scl enable rh-python36 '/home/mynetid/my_wrapper_script.sh'
### Otherwise, you're probably not running everything you think you are in the SCL environment.
###scl enable rh-python36 'python ./cast_to_glove_note_timeseries.py'
scl enable rh-python36 'python ./train_on_timeseries.py'
