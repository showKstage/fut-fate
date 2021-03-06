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

from arch.api.utils import log_utils
from fate_flow.entity.metric import MetricType, MetricMeta, Metric
from federatedml.util import consts
from federatedml.optim.convergence import converge_func_factory
from federatedml.federatedrec.svd.hetero_svd.hetero_svd_base import HeteroSVDBase

LOGGER = log_utils.getLogger()


class HeteroSVDArbiter(HeteroSVDBase):
    def __init__(self):
        super(HeteroSVDArbiter, self).__init__()
        self.role = consts.ARBITER#consts有定义 设置角色

    def _init_model(self, params):
        super()._init_model(params)
        early_stop = self.model_param.early_stop #early_stop早停法 防止过拟合
        self.converge_func = converge_func_factory(early_stop.converge_func, early_stop.eps).is_converge #check是否收敛
        self.loss_consumed = early_stop.converge_func != "weight_diff" #返回一个bool 对应下面不同的情况

    def callback_loss(self, iter_num, loss):
        metric_meta = MetricMeta(name='train',
                                 metric_type=MetricType.LOSS,
                                 extra_metas={
                                     "unit_name": "iters",
                                 })

        self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
        self.callback_metric(metric_name='loss',
                             metric_namespace='train',
                             metric_data=[Metric(iter_num, loss)])
    #检测状态
    def _check_monitored_status(self):
        loss = self.aggregator.aggregate_loss(suffix=self._iter_suffix())
        LOGGER.info(f"loss at iter {self.aggregator_iter}: {loss}")
        self.callback_loss(self.aggregator_iter, loss)
        if self.loss_consumed:
            converge_args = (loss,) if self.loss_consumed else (self.aggregator.model,)
            return self.aggregator.send_converge_status(self.converge_func,
                                                        converge_args=converge_args,
                                                        suffix=self._iter_suffix())
#converge看其收敛  发送状态
    def fit(self, data_inst):
        while self.aggregator_iter < self.max_iter:
            self.aggregator.aggregate_and_broadcast(suffix=self._iter_suffix())

            if self._check_monitored_status():
                LOGGER.info(f"early stop at iter {self.aggregator_iter}") #提前结束
                break
            self.aggregator_iter += 1
        else:#如果已经超过了最大迭代次数
            LOGGER.warn(f"reach max iter: {self.aggregator_iter}, not converged")

    def save_model(self):
        return self.aggregator.model
