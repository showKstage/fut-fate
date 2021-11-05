#!/bin/bash
set -x
python /fate/python/fate_flow/fate_flow_client.py -f upload -c /fate/examples/federatedrec-examples/hetero_fm/upload_data_guest.json
python /fate/python/fate_flow/fate_flow_client.py -f upload -c /fate/examples/federatedrec-examples/hetero_fm/upload_data_host.json
python /fate/python/fate_flow/fate_flow_client.py -f submit_job -c examples/federatedrec-examples/hetero_fm/test_hetero_fm_train_job_conf.json -d examples/federatedrec-examples/hetero_fm/test_hetero_fm_train_job_dsl.json