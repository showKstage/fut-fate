#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from fate_flow.utils.api_utils import get_json_result
from fate_flow.settings import stat_logger
from flask import Flask, request
import json, requests
from fate_flow.settings import API_VERSION, HTTP_PORT

fate_flow_server_host = 'http://127.0.0.1:{}/{}'.format(HTTP_PORT, API_VERSION)

manager = Flask(__name__)

@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))

# 查找目标用户
# 参数样例 '{"job_id":"20211030114602298164131", "role":"guest", "party_id":10000, "component_name":"hetero_svd_0"}'

@manager.route('/target/user/count', methods=['POST'])
def count_target_user():
    base_request_data = request.json
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }

    response = requests.post('{}/tracking/component/output/data'.format(fate_flow_server_host), data=base_request_data,
                             headers=headers)
    # print(json.loads(response.text))
    return json.loads(response.text)

    return 0
if __name__ == '__main__':

    # 测试自己写的API
    base_request_data = '{"job_id":"20211030114602298164131", "role":"guest", "party_id":10000, "component_name":"hetero_svd_0"}'
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }

    response = requests.post('{}/fate/target/user/count'.format(fate_flow_server_host), data=base_request_data,
                             headers=headers)