#!/bin/bash
mkdir figures
mkdir data
mkdir logs
source dqnTelemetry/bin/activate

pip install --upgrade pip
pip install -r requirements.txt