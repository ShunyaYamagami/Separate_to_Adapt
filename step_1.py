from data import *
from utilities import *
from networks import *
import matplotlib.pyplot as plt
import numpy as np
import tensorlayer as tl
from tqdm import tqdm
from run_steps import config

# def main():
def skip(data, label, is_train):
    return False

def transform(data, label, is_train):
    label = one_hot(config.num_classes, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label
ds = FileListDataset(config.labeled_txt_path, config.nas_root, transform=transform, skip_pred=skip, is_train=True, imsize=256)
source_train = CustomDataLoader(ds, batch_size=config.batch_size, num_threads=4)

def transform(data, label, is_train):
    if label in range(config.num_known_classes):
        label = one_hot(config.num_classes, label)
    else:
        label = one_hot(config.num_classes, config.num_known_classes)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label
# ds1 = FileListDataset('/nas/data/syamagami/GDA/data/Office31/webcam.txt', config.nas_root, transform=transform, skip_pred=skip, is_train=True, imsize=256)
ds1 = FileListDataset(config.unlabeled_txt_path, config.nas_root, transform=transform, skip_pred=skip, is_train=True, imsize=256)
target_train = CustomDataLoader(ds1, batch_size=config.batch_size, num_threads=4)

def transform(data, label, is_train):
    label = one_hot(config.num_all_classes, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label
# ds2 = FileListDataset('/nas/data/syamagami/GDA/data/Office31/webcam.txt', config.nas_root, transform=transform, skip_pred=skip, is_train=False, imsize=256)
ds2 = FileListDataset(config.test_txt_path, config.nas_root, transform=transform, skip_pred=skip, is_train=False, imsize=256)
target_test = CustomDataLoader(ds2, batch_size=config.batch_size, num_threads=4)

# setGPU('5')
# log = Logger('log/step_1', clear=True)
log = Logger(os.path.join(config.log_dir, 'step_1'), clear=False)
import logging
logger = logging.getLogger(__name__)
logger.info('''
=========================step 1=========================
''')


discriminator_t = CLS_0(2048,2,bottle_neck_dim = 256).cuda()
discriminator_p = Discriminator(n = config.num_known_classes).cuda()
feature_extractor = ResNetFc(model_name='resnet50', model_path=config.resnet_pretrained_path)
cls = CLS(feature_extractor.output_num(), config.num_classes, bottle_neck_dim=256)
net = nn.Sequential(feature_extractor, cls).cuda()

scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_discriminator_t = OptimWithSheduler(optim.SGD(discriminator_t.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_discriminator_p = OptimWithSheduler(optim.SGD(discriminator_p.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)

# =========================train the multi-binary classifier
k=0
# while k <500:
while k <1:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(zip(source_train.generator(), target_train.generator())):

        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()
        
        fs1, feature_source, __, predict_prob_source = net.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net.forward(im_target)
        
        p0 = discriminator_p.forward(fs1)
        p1 = discriminator_p.forward(ft1)
        p2 = torch.sum(p1, dim = -1)
    
        # =========================rank the output of the multi-binary classifiers
        __,_,_,dptarget = discriminator_t.forward(ft1.detach())
        r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][config.num_all_classes - 1:]
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
        w = torch.sort(p2.detach(),dim = 0)[1][config.num_all_classes - 1:]
        h = torch.sort(p2.detach(),dim = 0)[1][0:2]
        feature_otherep2 = torch.index_select(ft1, 0, w.view(2))
        feature_otherep1 = torch.index_select(ft1, 0, h.view(2))
        _,_,_,pred00 = discriminator_t.forward(feature_otherep2)
        _,_,_,pred01 = discriminator_t.forward(feature_otherep1)

        # =========================loss function
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        d1 = BCELossForMultiClassification(label_source[:,0:config.num_known_classes],p0)
        
        with OptimizerManager([optimizer_cls, optimizer_discriminator_p]):
            loss = ce + d1  
            loss.backward()
            
        k += 1

        log.step += 1

        if log.step % 20 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train', 'd1'], globals())

        # if log.step % 100 == 0:
        #     clear_output()
            
# =========================train the known/unknown discriminator
k=0
# while k <400:
while k <1:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):

        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()
        
        fs1, feature_source, __, predict_prob_source = net.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net.forward(im_target)
        
        p0 = discriminator_p.forward(fs1)
        p1 = discriminator_p.forward(ft1)
        p2 = torch.sum(p1, dim = -1)
    
        # =========================rank the output of the multi-binary classifiers
        __,_,_,dptarget = discriminator_t.forward(ft1.detach())
        r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][config.num_all_classes - 1:]
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
        w = torch.sort(p2.detach(),dim = 0)[1][config.num_all_classes - 1:]
        h = torch.sort(p2.detach(),dim = 0)[1][0:2]
        feature_otherep2 = torch.index_select(ft1, 0, w.view(2))
        feature_otherep1 = torch.index_select(ft1, 0, h.view(2))
        _,_,_,pred00 = discriminator_t.forward(feature_otherep2)
        _,_,_,pred01 = discriminator_t.forward(feature_otherep1)

        # =========================loss function
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        d1 = BCELossForMultiClassification(label_source[:,0:config.num_known_classes],p0)
        d2 = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.ones((2,1)), np.zeros((2,1))), axis = -1).astype('float32'))).cuda(),pred00)
        d2 += CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,1)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(),pred01)
        
        with OptimizerManager([optimizer_cls, optimizer_discriminator_p, optimizer_discriminator_t]):
            loss = ce + d1 +d2 
            loss.backward()
            
        k += 1
        log.step += 1

        if log.step % 20 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train', 'd1', 'd2'], globals())

# =========================save the parameters of the known/unknown discriminator
    torch.save(discriminator_t.state_dict(), os.path.join(config.log_dir, 'discriminator_a.pkl'))

# if __name__ == '__main__':
#     main()