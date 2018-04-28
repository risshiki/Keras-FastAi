bs = 64
from os import getcwd
path = getcwd() + "\dogscats\\"
from vgg16 import Vgg16

vgg = Vgg16()
batches = vgg.get_batches(path+'train', batch_size = bs)
val_batches = vgg.get_batches(path+'valid',batch_size = bs*2)
vgg.finetune(batches)
vgg.fit(batches,val_batches,bs,nb_epoch=1)
