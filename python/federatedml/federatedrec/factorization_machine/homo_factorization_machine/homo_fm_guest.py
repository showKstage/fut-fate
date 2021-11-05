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
from federatedml.federatedrec.factorization_machine.fm_model_weight import FactorizationMachineWeights
from federatedml.federatedrec.factorization_machine.homo_factorization_machine.homo_fm_base import HomoFMBase
from federatedml.model_selection import MiniBatch
from federatedml.federatedrec.optim.gradient.homo_fm_gradient import FactorizationGradient
from federatedml.util import consts
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class HomoFMGuest(HomoFMBase):
    def __init__(self):
        super(HomoFMGuest, self).__init__()     #使用父类的初始化方法

        self.gradient_operator = FactorizationGradient()
        self.loss_history = []
        self.role = consts.GUEST
        self.aggregator = aggregator.Guest()

    def _init_model(self, params):
        super()._init_model(params)
        #在父类中self.aggregate_iters = params.aggregate_iters

    '''
    Train fm model of role guest
        Parameters
        ----------
        data_instances: DTable of Instance, input data
    '''
#重载了modelbase.py的fit接口，该接口的作用是进行模型训练
    def fit(self, data_instances, validate_data=None):

        self._abnormal_detection(data_instances)    #检查数据
        self.init_schema(data_instances)            #将数据初始化为表格

        #This module is used for evaluating the performance of model during training process.训练期间评估性能
        #it will be called only in fit process of models.模型拟合过程中被调用
        validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        self.model_weights = self._init_model_variables(data_instances)
        '''
               model_weights是fm公式中的权重theta部分，包括线性部分和特征交叉部分
               model_weights.intercept_ ：截距 线性部分的θ0
               model_weights.w_  ：线性部分的θ1-θn，list
               model_weights.embed_  ：特征交叉部分的权重矩阵分解后VVT的V，二维矩阵
        '''

        #Generate mini-batch data or index  生成小批量的数据或索引
        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)
        model_weights = self.model_weights

        #
        degree = 0
        while self.n_iter_ < self.max_iter+1:  #iter迭代次数
            LOGGER.info("iter:{}".format(self.n_iter_))
            batch_data_generator = mini_batch_obj.mini_batch_data_generator()  #fate写的生成器（迭代器的一种）
            #迭代器的所有数据都在内存里，很耗内存。生成器是可以迭代的，但是只可以读取它一次。因为用的时候才生成。

#send model(gradient) for aggregate, then set aggegated(聚合后的) model(gradient) to local
            self.optimizer.set_iters(self.n_iter_)
            if (self.n_iter_ > 0 and self.n_iter_ % self.aggregate_iters == 0) or self.n_iter_ == self.max_iter:
                # This loop will run after weight has been created,weight will be in LRweights
                #weight变量接收arbiter传来的weight，然后更新guest的权重模型
                weight = self.aggregator.aggregate_then_get(weight, degree=degree,
                                                            suffix=self.n_iter_)
                weight._weights = np.array(weight._weights)
                # This weight is transferable Weight, should get the parameter back
                self.model_weights.update(weight)


                LOGGER.debug("Before aggregate: {}, degree: {} after aggregated: {}".format(
                    model_weights.unboxed / degree,
                    degree,
                    self.model_weights.unboxed))

                #发送损失到Arbiter（aggregator，coordinator）
                loss = self._compute_loss(data_instances)  #计算损失
                self.aggregator.send_loss(loss, degree=degree, suffix=(self.n_iter_,))  #发送
                degree = 0

                #判断模型是否收敛，若是则跳出循环
                self.is_converged = self.aggregator.get_converge_status(suffix=(self.n_iter_,))  #从ari处拿到收敛状态
                LOGGER.info("n_iters: {}, loss: {} converge flag is :{}".format(self.n_iter_, loss, self.is_converged))
                if self.is_converged:
                    break
                model_weights = self.model_weights

            #mini-batch梯度下降
            batch_num = 0
            for batch_data in batch_data_generator:
                n = batch_data.count()
                LOGGER.debug("In each batch, fm_weight: {}, batch_data count: {},w:{},embed:{}"
                             .format(model_weights.unboxed, n, model_weights.w_, model_weights.embed_))
                # self.gradient_operator.compute_gradient 计算梯度
                f = functools.partial(self.gradient_operator.compute_gradient,
                                      w=model_weights.w_,
                                      embed=model_weights.embed_,
                                      intercept=model_weights.intercept_,
                                      fit_intercept=self.fit_intercept)
                grad = batch_data.mapPartitions(f).reduce(fate_operator.reduce_add)
                grad /= n
                LOGGER.debug('iter: {}, batch_index: {}, grad: {}, n: {}'.format(
                    self.n_iter_, batch_num, grad, n))

                weight = self.optimizer.update_model(model_weights, grad, has_applied=False)
                weight._weights = np.array(weight._weights)
                model_weights.update(weight)
                batch_num += 1
                degree += n

            validation_strategy.validate(self, self.n_iter_)
            self.n_iter_ += 1

    #重载了modelbase的predict的接口，该接口的作用是运用模型进行预测
    #Data_inst 是一个 Table. 用于建模组件的预测功能
    def predict(self, data_instances):
        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)
        #fm模型计算
        predict_wx_plus_fm = self.compute_wx_plus_fm(data_instances, self.model_weights)
        pred_table = self.classify(predict_wx_plus_fm, self.model_param.predict_param.threshold)
        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = pred_table.join(predict_result, lambda x, y: [y, x[1], x[0],
                                                                       {"1": x[0], "0": 1 - x[0]}])
        return predict_result
