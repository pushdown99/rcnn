import torch as t
from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

num_gpus    = t.cuda.device_count()
num_workers = num_gpus * 4

class Config:
    data = 'nia'
    data_dir = 'dataset/NIA/'
    n_class = 168

    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = num_workers
    test_num_workers = num_workers

    load_path = None

    lr = 0.01
    epoch = 10
    momentum = 0.9
    log_interval = 1000
    step_size = 2
    resume_training = True
    resume_model = ''
    load_RPN = True
    enable_clip_gradient = True
    use_normal_anchors = True

    disable_language_model = True
    mps_feature_len = 1024
    dropout = True
    MPS_iter = 1
    gate_width = 128
    nhidden_caption = 512
    nembedding = 256
    rnn_type = 'LSTM_normal' # LSTM_baseline LSTM_im LSTM_normal
    caption_use_bias = True
    caption_use_dropout = 0.  # const=0.5 hyhwang
    enable_bbox_reg = True
    disable_bbox_reg = False

    region_bbox_reg = True
    use_kernel_function = True

    seed = 1
    saved_model_path = 'output/RPN/RPN_region_full_best.h5'
    dataset_option = 'normal' # normal fat small
    output_dir = 'output/HDN'
    model_name = 'HDN'
    nesterov = True
    finetune_language_model = True
    optimizer = 0 # [0: SGD | 1: Adam | 2: Adagrad]

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
