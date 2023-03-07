#  Torch imports
import torch
import numpy as np
# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
# Local imports
from data import dataset as dset
from model.common import Evaluator
from utils.utils import load_args
from config_model import configure_model
from flags import parser

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main():
    args = parser.parse_args()
    args.dataset = 'ut-zap50k' # Choose from ut-zap50k | mit-states | cgqa
    args.main_root = os.path.dirname(__file__)
    args.data_root = '/root/datasets'
    device = 0 # Your GPU order. If you don't have a GPU, ignore this.

    # Get arguments and start logging
    print('> Initialize parameters')
    config_path = ospj(args.main_root, 'configs', args.dataset, 'CANet.yml')
    if os.path.exists(config_path):
        load_args(config_path, args)
        print('  Load parameter values from file {}'.format(config_path))
    else:
        print('  No yml file found. Keep default parameter values in flags.py')

    if torch.cuda.is_available():
        args.device = 'cuda:{}'.format(device)
    else:
        args.device = 'cpu'
    print('> Choose device: {}'.format(args.device))
    
    # Get dataset
    print('> Load dataset')
    args.phase = 'test'
    testset = dset.CompositionDataset(
        args=args,
        root=os.path.join(args.data_root, args.dataset),
        phase=args.phase,
        split=args.splitname,
        model =args.image_extractor,
        update_image_features = args.update_image_features,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    # Get model and optimizer
    image_extractor, model = configure_model(args, testset, train=False)
    print(model)

    # load saved model
    print('> Load saved trained model')
    save_path = os.path.join(args.main_root, 'saved model')
    if os.path.exists(save_path):
        checkpoint = torch.load(ospj(save_path, 'saved_{}.t7'.format(args.dataset)), map_location=args.device)
    else:
        print('  No saved model found in local disk. Please run train.py to train the model first')
        return
    if image_extractor:
        try:
            image_extractor.load_state_dict(checkpoint['image_extractor'])
            image_extractor.eval()
        except:
            print('  No saved image extractor in checkpoint file')
    model.load_state_dict(checkpoint['net'])
    model.eval()

    print('> Initialize evaluator')
    evaluator = Evaluator(testset, args)

    with torch.no_grad():
        test(image_extractor, model, testloader, evaluator, args)

def test(image_extractor, model, testloader, evaluator, args):
    if image_extractor:
        image_extractor.eval()

    model.eval()

    all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], []

    for _, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(args.device) for d in data]
        if image_extractor:
            data[0] = image_extractor(data[0])
        _, predictions = model.val_forward(data)
        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
        'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k].to('cpu') for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                            topk=args.topk)
    
    attr_acc = stats['closed_attr_match']
    obj_acc = stats['closed_obj_match']
    seen_acc = stats['best_seen']
    unseen_acc = stats['best_unseen']
    HM = stats['best_hm']
    AUC = stats['AUC']
    print('|----Test Finished: Attr Acc: {:.2f}% | Obj Acc: {:.2f}% | Seen Acc: {:.2f}% | Unseen Acc: {:.2f}% | HM: {:.2f}% | AUC: {:.2f}'.\
        format(attr_acc*100, obj_acc*100, seen_acc*100, unseen_acc*100, HM*100, AUC*100))

if __name__ == '__main__':
    print('======== Welcome! ========')
    print('> Program start') 
    main()
    print('=== Program terminated ===')
