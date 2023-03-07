#  Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from torch.utils.data import DataLoader
# Python imports
from tqdm import tqdm
import os
from os.path import join as ospj
import numpy as np
import random
from flags import parser
import csv
#Local imports
from model.common import Evaluator
from config_model import configure_model
from data import dataset as dset
from utils.utils import load_args

best_attr = 0.0
best_obj = 0.0
best_seen = 0.0
best_unseen = 0.0
best_auc = 0.0
best_hm = 0.0
best_epoch = 0.0

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

def main():
    # Get arguments and start logging
    print('> Initialize parameters')
    args = parser.parse_args()
    args.dataset = 'mit-states'
    args.main_root = os.path.dirname(__file__)
    args.data_root = '/root/datasets/'
    device = 0
    args.test_set = 'val'
    
    config_path = ospj(args.main_root, 'configs', args.dataset, 'CANet.yml')
    if os.path.exists(config_path):
        load_args(config_path, args)
        print('  Load parameter values from file {}'.format(config_path))
    else:
        print('  No yml file found. Keep default parameter values in flags.py')

    # Choose device
    args.device = 'cuda:{}'.format(device)
    print('> Choose device: {}'.format(args.device))

    # Tensorboard
    print('> Initialize tensorboard')
    logpath = ospj(args.main_root, 'logs', args.dataset)
    writer = SummaryWriter(log_dir=logpath, flush_secs=30)
    os.makedirs(logpath, exist_ok=True)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './logs'])
    tb.launch()

    # Get dataset
    print('> Load dataset {}'.format(args.dataset))
    trainset = dset.CompositionDataset(
        args=args,
        root=ospj(args.data_root, args.dataset),
        phase='train',
        split=args.splitname,
        model =args.image_extractor,
        update_image_features = args.update_image_features,
        train_only= args.train_only,
)
    testset = dset.CompositionDataset(
        args=args,
        root=ospj(args.data_root, args.dataset),
        phase=args.test_set,
        split=args.splitname,
        model =args.image_extractor,
        update_image_features = args.update_image_features,
    )

    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    print(model)

    # Dataloaders
    print('> Initialize trainset and {}set dataloaders'.format(args.test_set))
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    testloader = DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers)
    
    # Train an epoch
    train = train_normal
    
    # Evaluate an epoch
    print('> Initialize evaluator')
    evaluator = Evaluator(testset, args)

    for epoch in range(args.max_epochs):
        print('Epoch {} | Best Attr: {:.2f}% | Best Obj: {:.2f}% | Best Seen: {:.2f}% | Best Unseen: {:.2f}% | Best HM: {:.2f}% | Best AUC: {:.2f} | Best Epoch: {:.0f}'.\
            format(epoch+1, best_attr*100, best_obj*100, best_seen*100, best_unseen*100, best_hm*100, best_auc*100, best_epoch))
        train(args, epoch, image_extractor, model, trainloader, optimizer, writer)
        with torch.no_grad():
            test(args, epoch, image_extractor, model, testloader, evaluator, logpath)

def train_normal(args, epoch, image_extractor, model, trainloader, optimizer, writer):
    '''
    Runs training for an epoch
    '''
    if args.update_image_features: 
        image_extractor.train()
    model = model.train() # Let's switch to training

    train_loss = 0.0
    trainloader = tqdm(trainloader, desc='|--Training')
    for idx, data in enumerate(trainloader):
        data = [d.to(args.device) for d in data]
        if args.update_image_features:
            data[0] = image_extractor(data[0])
        loss = model(data)[0]

        trainloader.set_description(desc='|--Training | Batch Loss: {:.4f}'.format(loss.item()))
        if loss == None:
            trainloader.close()
            return

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
    
    trainloader.close()
    train_loss = train_loss/len(trainloader)
    print('|----Train Loss: {:.4f}'.format(train_loss))
    writer.add_scalar('Loss/train_total', train_loss, epoch)

def test(args, epoch, image_extractor, model, testloader, evaluator, logpath):
    '''
    Runs testing for an epoch
    '''
    def save_checkpoint(filename):
        state = {
            'epoch': epoch+1,
            'AUC': stats['AUC']
        }
        state['net'] = model.state_dict()
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, ospj(args.main_root, 'logs', args.dataset, 'ckpt_{}_{}.t7'.format(filename, args.dataset)))

    if args.update_image_features: image_extractor.eval()
    model = model.eval()

    all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], []
    
    testloader = tqdm(testloader, desc='|--Testing')
    for idx, data in enumerate(testloader):
        data = [d.to(args.device) for d in data]
        if args.update_image_features:
            data[0] = image_extractor(data[0])
        predictions = model(data)[1]

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    del predictions, attr_truth, obj_truth, pair_truth
    testloader.close()
    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    global best_attr, best_obj, best_seen, best_unseen, best_auc, best_hm, best_epoch
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat([all_pred[i][k].cpu() for i in range(len(all_pred))])
    del all_pred

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch
    
    # print(result)
    attr_acc = stats['closed_attr_match']
    obj_acc = stats['closed_obj_match']
    seen_acc = stats['best_seen']
    unseen_acc = stats['best_unseen']
    HM = stats['best_hm']
    AUC = stats['AUC']
    print('|----Test Finished: Attr Acc: {:.2f}% | Obj Acc: {:.2f}% | Seen Acc: {:.2f}% | Unseen Acc: {:.2f}% | HM: {:.2f}% | AUC: {:.2f}'.\
        format(attr_acc*100, obj_acc*100, seen_acc*100, unseen_acc*100, HM*100, AUC*100))

    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)
    if AUC > best_auc:
        best_auc = AUC
        best_attr = attr_acc
        best_obj = obj_acc
        best_seen = seen_acc
        best_unseen = unseen_acc
        best_hm = HM
        best_epoch = epoch
        print('|----New Best AUC {:.2f}. SAVE to local disk!'.format(best_auc*100))
        save_checkpoint('best_auc')

    # Logs
    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)

if __name__ == '__main__':
    print('======== Welcome! ========')
    print('> Program start') 
    main()
    print('> Program terminated')
