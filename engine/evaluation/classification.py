# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import platform
import paddle
import numpy as np

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger


def average_precision(pred, target):
    r"""Calculate the average precision for a single class.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        pred (np.ndarray): The model prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).

    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap


def mAP(pred, target):
    """Calculate the mean average precision with respect of classes.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.

    Returns:
        float: A single float as mAP value.
    """

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean()
    return mean_ap


def classification_eval(engine, epoch_id=0):
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]
    calculate_mAP = engine.config["Global"].get("use_multilabel", False)

    metric_key = None
    tic = time.time()
    accum_samples = 0
    total_samples = len(
        engine.eval_dataloader.
        dataset) if not engine.use_dali else engine.eval_dataloader.size
    max_iter = len(engine.eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(engine.eval_dataloader)
    preds = []
    targets = []
    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        batch[0] = paddle.to_tensor(batch[0]).astype("float32")
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")

        # image input
        if engine.amp:
            amp_level = engine.config['AMP'].get("level", "O1").upper()
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=amp_level):
                out = engine.model(batch[0])
                # calc loss
                if engine.eval_loss_func is not None:
                    loss_dict = engine.eval_loss_func(out, batch[1])
                    for key in loss_dict:
                        if key not in output_info:
                            output_info[key] = AverageMeter(key, '7.5f')
                        output_info[key].update(loss_dict[key].numpy()[0],
                                                batch_size)
        else:
            out = engine.model(batch[0])
            # calc loss
            if engine.eval_loss_func is not None:
                loss_dict = engine.eval_loss_func(out, batch[1])
                for key in loss_dict:
                    if key not in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')
                    output_info[key].update(loss_dict[key].numpy()[0],
                                            batch_size)

        # just for DistributedBatchSampler issue: repeat sampling
        current_samples = batch_size * paddle.distributed.get_world_size()
        accum_samples += current_samples

        # calc metric
        if engine.eval_metric_func is not None:
            if paddle.distributed.get_world_size() > 1:
                label_list = []
                paddle.distributed.all_gather(label_list, batch[1])
                labels = paddle.concat(label_list, 0)

                if isinstance(out, dict):
                    if "Student" in out:
                        out = out["Student"]
                    elif "logits" in out:
                        out = out["logits"]
                    else:
                        msg = "Error: Wrong key in out!"
                        raise Exception(msg)
                if isinstance(out, list):
                    pred = []
                    for x in out:
                        pred_list = []
                        paddle.distributed.all_gather(pred_list, x)
                        pred_x = paddle.concat(pred_list, 0)
                        pred.append(pred_x)
                else:
                    pred_list = []
                    paddle.distributed.all_gather(pred_list, out)
                    pred = paddle.concat(pred_list, 0)

                if accum_samples > total_samples and not engine.use_dali:
                    pred = pred[:total_samples + current_samples -
                                accum_samples]
                    labels = labels[:total_samples + current_samples -
                                    accum_samples]
                    current_samples = total_samples + current_samples - accum_samples
                metric_dict = engine.eval_metric_func(pred, paddle.clip(labels, 0))
                preds.append(pred.numpy())
                targets.append(labels.numpy())
            else:
                metric_dict = engine.eval_metric_func(out, paddle.clip(batch[1], 0))
                preds.append(out.numpy())
                targets.append(batch[1].numpy())

            for key in metric_dict:
                if metric_key is None:
                    metric_key = key
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')

                output_info[key].update(metric_dict[key].numpy()[0],
                                        current_samples)

        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].val)
                for key in output_info
            ])
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()
    if engine.use_dali:
        engine.eval_dataloader.reset()
    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, output_info[key].avg) for key in output_info
    ])
    if calculate_mAP:
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        mean_ap = mAP(preds, targets)
        metric_msg += ", {}: {:.5f}".format("mAP", mean_ap)

    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    # do not try to save best eval.model
    if engine.eval_metric_func is None:
        return -1
    # return 1st metric in the dict
    return output_info[metric_key].avg
