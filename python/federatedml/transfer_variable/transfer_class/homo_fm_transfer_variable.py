#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

################################################################################
#
# AUTO GENERATED TRANSFER VARIABLE CLASS. DO NOT MODIFY
#
################################################################################

from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


# noinspection PyAttributeOutsideInit
'''
定义此模块的传递变量py文件并生成传递变量对象
在该定义文件中，您需要创建需要的 transfer_variable 类，并继承BaseTransferVariables类，
然后定义相应的变量，并为其赋予需要的传输权限
'''

'''
name
变量名
src
应为 “guest”，”host”，”arbiter” 的某些组合，它表示发送交互信息从何处发出。
dst
应为 “guest”，”host”，”arbiter” 的某些组合列表，用于定义将交互信息发送到何处。
'''
class HomoFMTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.aggregated_model = self._create_variable(name='aggregated_model', src=['arbiter'], dst=['guest', 'host'])
        self.dh_ciphertext_bc = self._create_variable(name='dh_ciphertext_bc', src=['arbiter'], dst=['guest', 'host'])
        self.dh_ciphertext_guest = self._create_variable(name='dh_ciphertext_guest', src=['guest'], dst=['arbiter'])
        self.dh_ciphertext_host = self._create_variable(name='dh_ciphertext_host', src=['host'], dst=['arbiter'])
        self.dh_pubkey = self._create_variable(name='dh_pubkey', src=['arbiter'], dst=['guest', 'host'])
        self.guest_loss = self._create_variable(name='guest_loss', src=['guest'], dst=['arbiter'])
        self.guest_model = self._create_variable(name='guest_model', src=['guest'], dst=['arbiter'])
        self.guest_party_weight = self._create_variable(name='guest_party_weight', src=['guest'], dst=['arbiter'])
        self.guest_uuid = self._create_variable(name='guest_uuid', src=['guest'], dst=['arbiter'])
        self.host_loss = self._create_variable(name='host_loss', src=['host'], dst=['arbiter'])
        self.host_model = self._create_variable(name='host_model', src=['host'], dst=['arbiter'])
        self.host_party_weight = self._create_variable(name='host_party_weight', src=['host'], dst=['arbiter'])
        self.host_uuid = self._create_variable(name='host_uuid', src=['host'], dst=['arbiter'])
        self.is_converge = self._create_variable(name='is_converge', src=['arbiter'], dst=['guest', 'host'])
        self.predict_result = self._create_variable(name='predict_result', src=['arbiter'], dst=['host'])
        self.predict_wx = self._create_variable(name='predict_wx', src=['host'], dst=['arbiter'])
        self.uuid_conflict_flag = self._create_variable(name='uuid_conflict_flag', src=['arbiter'], dst=['guest', 'host'])
