#!/bin/bash
set -x

python /fate/python/fate_flow/fate_flow_client.py -f upload -c /fate/examples/federatedrec-examples/hetero_gmf/upload_data_guest.json
python /fate/python/fate_flow/fate_flow_client.py -f upload -c /fate/examples/federatedrec-examples/hetero_gmf/upload_data_host.json
python /fate/python/fate_flow/fate_flow_client.py -f upload -c /fate/examples/federatedrec-examples/hetero_gmf/upload_eval_guest.json
python /fate/python/fate_flow/fate_flow_client.py -f submit_job -c /fate/examples/federatedrec-examples/hetero_gmf/test_hetero_gmf_train_then_predict_conf.json -d /fate/examples/federatedrec-examples/hetero_gmf/test_hetero_gmf_train_then_predict_dsl.json