class TrainArgs:
    def __init__(self, **kwargs):
        self.depth = False
        # self.dataset = 'cityscapes'
        self.dataset = 'e1'
        self.workers = 4
        self.base_size = 1024
        self.crop_size = 768
        self.loss_type = 'ce'
        
        #self.epochs = kwargs.get("epochs", 2)
        self.epochs = kwargs.get("epochs", 1)
        self.FT_epochs = 1
        self.start_epoch = 0

        #self.batch_size = 4
        self.batch_size = 8
        
        self.val_batch_size = 1
        self.use_balanced_weights = False
        self.num_class = 24
        
        self.lr = kwargs.get("learning_rate", 1e-4)
        self.FT_lr = 1e-5
        
        self.weight_decay = 2.5e-5
        
        self.lr_scheduler = 'cos'
        self.momentum = 0.9
        
        self.no_cuda = False
        # self.gpu_ids = [0,1,2,3]
        self.gpu_ids = [0,1,2]

        self.seed = 1
        self.resume = None
        self.checkname = 'RFNet'
        self.ft = True
        self.eval_interval = kwargs.get("eval_interval", 2)
        self.no_val = kwargs.get("no_val", True)
        self.cuda = True


class ValArgs:
    def __init__(self, **kwargs):
        # self.dataset = 'cityscapes'
        self.dataset = 'e1'
        self.workers = 4
        self.base_size = 1024
        self.crop_size = 768
        self.batch_size = 6
        self.val_batch_size = 1
        self.test_batch_size = 1
        self.num_class = 24
        self.no_cuda = False
        self.gpu_ids = '0'
        self.checkname = None
        self.weight_path = "./models/530_exp3_2.pth"
        self.save_predicted_image = False
        self.color_label_save_path = './test/color'
        self.merge_label_save_path = './test/merge'
        self.label_save_path = './test/label'
        self.merge = True
        self.depth = False
        self.cuda = False
