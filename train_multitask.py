import argparse
import mxnet as mx
import logging

from metric import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

class MultiTask_iter(mx.io.DataIter):
    def __init__(self, data_iter):
        super(MultiTask_iter,self).__init__('multitask_iter')
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        # the name of the label if corresponding to the model you define in get_fine_tune_model() function
        return [('softmax_label', provide_label[1]), \
                ('softmax_multitask_label', provide_label[1])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]
        # we set task 2 as: if label>0 or not
        label2 = mx.nd.array(label.asnumpy()>0).astype('float32')
        return mx.io.DataBatch(data=batch.data, label=[label,label2], \
                pad=batch.pad, index=batch.index)

def get_fine_tune_model(sym, num_classes, layer_name, batchsize):
    
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    fc = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    sm = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    fc_multitask = mx.symbol.FullyConnected(data=net, num_hidden=2, name='fc_multitask')
    sm_multitask = mx.symbol.SoftmaxOutput(data=fc_multitask, name='softmax_multitask')
    softmax = mx.symbol.Group([sm,sm_multitask])

    return softmax

def multi_factor_scheduler(begin_epoch, epoch_size, step=[5,10], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def train_model(model, gpus, batch_size, image_shape, epoch=0, num_epoch=20, kv='device'):
    train = mx.image.ImageIter(
        batch_size          = args.batch_size,
        data_shape          = (3,224,224),        
        label_width         = 1,
        path_imglist        = args.data_train,
        path_root           = args.image_train,
        part_index          = kv.rank,
        num_parts           = kv.num_workers,
        shuffle             = True,
        data_name           = 'data',
        aug_list            = mx.image.CreateAugmenter((3,224,224),resize=224,rand_crop=True,rand_mirror=True,mean = True,std = True))
    
    val = mx.image.ImageIter(
        batch_size          = args.batch_size,
        data_shape          = (3,224,224),
        label_width         = 1,
        path_imglist        = args.data_val,
        path_root           = args.image_val,
        part_index          = kv.rank,
        num_parts           = kv.num_workers,       
        data_name           = 'data',
        aug_list            = mx.image.CreateAugmenter((3,224,224),resize=224, mean = True,std = True))

    train = MultiTask_iter(train)
    val = MultiTask_iter(val)

    kv = mx.kvstore.create(args.kv_store)

    prefix = model
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    new_sym = get_fine_tune_model(
        sym, args.num_classes, 'flatten', args.batch_size)

    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    lr_scheduler=multi_factor_scheduler(args.epoch, epoch_size)

    optimizer_params = {
            'learning_rate': args.lr,
            'momentum' : args.mom,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler}
    initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]
        
    model = mx.mod.Module(
        context       = devs,
        symbol        = new_sym,
        data_names    = ['data'],
        label_names   = ['softmax_label','softmax_multitask_label']
    )

    checkpoint = mx.callback.do_checkpoint(args.save_result + args.save_name)

    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(Multi_Accuracy(num=2,output_names=['softmax_output','softmax_multitask_output']))

    model.fit(train,
              begin_epoch=epoch,
              num_epoch=num_epoch,
              eval_data=val,
              eval_metric=eval_metric,
              validation_metric=eval_metric,
              kvstore=kv,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              arg_params=arg_params,
              aux_params=aux_params,
              initializer=initializer,
              allow_missing=True,
              batch_end_callback=mx.callback.Speedometer(args.batch_size, 20),
              epoch_end_callback=checkpoint)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model',         type=str, required=True,)
    parser.add_argument('--gpus',          type=str, default='0')
    parser.add_argument('--batch-size',    type=int, default=200)
    parser.add_argument('--epoch',         type=int, default=0)
    parser.add_argument('--image-shape',   type=str, default='3,224,224')
    parser.add_argument('--data-train',    type=str)
    parser.add_argument('--image-train',   type=str)
    parser.add_argument('--data-val',      type=str)
    parser.add_argument('--image-val',     type=str)
    parser.add_argument('--num-classes',   type=int)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--num-epoch',     type=int, default=2)
    parser.add_argument('--kv-store',      type=str, default='device', help='the kvstore type')
    parser.add_argument('--save-result',   type=str, help='the save path')
    parser.add_argument('--num-examples',  type=int, default=20000)
    parser.add_argument('--mom',           type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd',            type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--save-name',     type=str, help='the save name of model')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    kv = mx.kvstore.create(args.kv_store)

    if not os.path.exists(args.save_result):
        os.mkdir(args.save_result)
    hdlr = logging.FileHandler(args.save_result+ '/train.log')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logging.info(args)

    train_model(model=args.model, gpus=args.gpus, batch_size=args.batch_size,
          image_shape='3,224,224', epoch=args.epoch, num_epoch=args.num_epoch, kv=kv)
