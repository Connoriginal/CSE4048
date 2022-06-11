from torch.nn.utils.rnn import pad_sequence
from tensorboardX import SummaryWriter
import torch


#######################################################
#               Collate fn
#######################################################

'''
class to add padding to the batches
collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class is used on single example
'''

class MyCollate:
    def __init__(self, pad_idx, batch_first):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        
    
    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        #get all source indexed sentences of the batch
        source = [item[0] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        source = pad_sequence(source, batch_first=self.batch_first, padding_value = self.pad_idx) 
        
        #get all target indexed sentences of the batch
        target = [item[1] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        target = pad_sequence(target, batch_first=self.batch_first, padding_value = self.pad_idx)
        return source, target

class TensorboardPlotter(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir) # log가 저장될 경로

    def loss_plot(self, tag, type, scalar_value, global_step):
        self.writer.add_scalar(tag + '/' + type, torch.tensor(scalar_value), global_step)

    def overlap_plot(self, tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(tag, tag_scalar_dict, global_step)
    
    def img_plot(self, title, img, global_step):
        self.writer.add_images(title, img, global_step)

    def text_plot(self, tag, text, global_step):
        self.writer.add_text(tag, text, global_step)

    def close(self):
        self.writer.close()