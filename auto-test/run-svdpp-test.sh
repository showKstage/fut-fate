#!/bin/bash
set -x
python /fate/python/fate_flow/fate_flow_client.py -f upload -c /fate/examples/federatedrec-examples/hetero_svdpp/upload_data_guest.json
python /fate/python/fate_flow/fate_flow_client.py -f upload -c /fate/examples/federatedrec-examples/hetero_svdpp/upload_data_host.json
python /fate/python/fate_flow/fate_flow_client.py -f submit_job -c /fate/examples/federatedrec-examples/hetero_svdpp/test_hetero_svdpp_train_job_conf.json -d /fate/examples/federatedrec-examples/hetero_svdpp/test_hetero_svdpp_train_job_dsl.json