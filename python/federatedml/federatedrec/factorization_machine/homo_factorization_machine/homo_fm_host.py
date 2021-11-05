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

import functools
import numpy as np

from arch.api.utils import log_utils
from federatedml.framework.homo.procedure import aggregator
from federatedml.federatedrec.factorization_machine.homo_factorization_machine.homo_fm_base import HomoFMBase
from federatedml.federatedrec.factorization_machine.fm_model_weight import FactorizationMachineWeights
from federatedml.model_selection import MiniBatch
from federatedml.federatedrec.optim.gradient.homo_fm_gradient import FactorizationGradient
from federatedml.util import consts
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class HomoFMHost(HomoFMBase):
    def __init__(self):
        super(HomoFMHost, self).__init__()
        self.gradient_operator = None   # def _init_model中有
        self.loss_history = []
        self.is_converged = False       #guest无
        self.role = consts.HOST         #与guest不同
        self.aggregator = aggregator.Host() #与guest不同
        self.model_weights = None       #guest无

    def _init_model(self, params):
        super()._init_model(params)
        self.gradient_operator = FactorizationGradient()

    def fit(self, data_instances, validate_data=None):
        LOGGER.debug("Start data count: {}".format(data_instances.count()))

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)

        validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        self.model_weights = self._init_model_variables(data_instances)

        LOGGER.debug("After init, model_weights: {}".format(self.model_weights.unboxed))

        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)
        model_weights = self.model_weights

        # 计算梯度
        degree = 0
        while self.n_iter_ < self.max_iter+1:
            LOGGER.info("iter:{}".format(self.n_iter_))
            batch_data_generator = mini_batch_obj.mini_batch_data_generator()

            if (self.n_iter_ > 0 and self.n_iter_ % self.aggregate_iters == 0) or self.n_iter_ == self.max_iter:
                weight = self.aggregator.aggregate_then_get(weight, degree=degree,
                                                            suffix=self.n_iter_)

                weight._weights = np.array(weight._weights)
                # This weight is transferable Weight, should get the parameter back
                self.model_weights.update(weight)

                LOGGER.debug("Before aggregate: {}, degree: {} after aggregated: {}".format(
                    model_weights.unboxed / degree,
                    degree,
                    self.model_weights.unboxed))

                loss = self._compute_loss(data_instances)
                self.aggregator.send_loss(loss, degree=degree, suffix=(self.n_iter_,))
                degree = 0

                self.is_converged = self.aggregator.get_converge_status(suffix=(self.n_iter_,))
                LOGGER.info("n_iters: {}, is_converge: {}".format(self.n_iter_, self.is_converged))
                if self.is_converged:
                    break
                model_weights = self.model_weights

            batch_num = 0
            for batch_data in batch_data_generator:
                n = batch_data.count()
                LOGGER.debug("before compute_gradient,w_:{},embed_:{}".format(model_weights.w_,model_weights.embed_))
                f = functools.partial(self.gradient_operator.compute_gradient,
                                      w=model_weights.w_,
                                      embed=model_weights.embed_,
                                      intercept=model_weights.intercept_,
                                      fit_intercept=self.fit_intercept)
                grad = batch_data.mapPartitions(f).reduce(fate_operator.reduce_add)
                grad /= n
                weight = self.optimizer.update_model(model_weights, grad, has_applied=False)
                weight._weights = np.array(weight._weights)
                model_weights.update(weight)
                batch_num += 1
                degree += n

            validation_strategy.validate(self, self.n_iter_)
            self.n_iter_ += 1

        LOGGER.info("Finish Training task, total iters: {}".format(self.n_iter_))

    def predict(self, data_instances):

        LOGGER.info(f'Start predict task')
        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)


        predict_wx_plus_fm = self.compute_wx_plus_fm(data_instances, self.model_weights)

        pred_table = self.classify(predict_wx_plus_fm, self.model_param.predict_param.threshold)

        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = pred_table.join(predict_result, lambda x, y: [y, x[1], x[0],
                                                                       {"1": x[0], "0": 1 - x[0]}])
        return predict_result
