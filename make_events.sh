#! /bin/bash
# TODO : Support providing defaults for `argo-workflows create`

python remote_flow.py --branch rank16 argo-workflows create 
python data_prep_flow.py argo-workflows create 
python data_prep_flow.py argo-workflows trigger --raise-event True