import os

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'
import mxnet as mx


# define multi task accuracy
class Multi_Accuracy(mx.metric.EvalMetric):
    def __init__(self, num=None, output_names = None):
        self.num = num
        super(Multi_Accuracy, self).__init__('multi_accuracy', num)
        self.output_names = output_names

    def reset(self):
        ''' Resets the internal evaluation result to initial state.'''
        self.num_inst = 0 if self.num is None else [0] * self.num
        self.sum_metric = 0.0 if self.num is None else [0.0] * self.num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels,preds)

        if self.num != None:
            assert len(labels) == self.num

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label,pred_label)

            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)

    def get(self):
        if self.num is None:
            return super(Multi_Accuracy, self).get()
        else:
            return zip(*(('%s-task%d' % (self.name, i), float('nan') if self.num_inst[i] == 0
            else self.sum_metric[i] / self.num_inst[i])
                         for i in range(self.num)))


