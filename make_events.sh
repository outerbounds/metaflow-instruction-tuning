#! /bin/bash
# TODO : Support providing defaults for `argo-workflows create`

python remote_flow.py --branch test1 argo-workflows create 
python data_prep_flow.py --environment=pypi argo-workflows create 
python data_prep_flow.py --environment=pypi argo-workflows trigger --raise-event True