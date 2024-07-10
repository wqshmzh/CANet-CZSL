# external libs
from tqdm import tqdm
from PIL import Image
import os
from os.path import join as ospj
from glob import glob
# torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
# local libs
from utils.utils import get_norm_values, chunks
from model.image_extractor import get_image_extractor
from itertools import product

class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        img = Image.open(ospj(self.root_dir, img)).convert('RGB')  # We don't want alpha
        return img


def dataset_transform(phase, norm_family='imagenet'):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use
        
        Returns
            transform: A list of pytorch transforms
    '''
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform

# Dataset class now
class CompositionDataset(Dataset):
    '''
    Inputs
        root: String of base dir of dataset
        phase: String train, val, test
        split: String dataset split
        subset: Boolean if true uses a subset of train at each epoch
        num_negs: Int, numbers of negative pairs per batch
        pair_dropout: Percentage of pairs to leave in current epoch
    '''

    def __init__(
            self,
            args,
            root,
            phase,
            split='compositional-split',
            model='resnet18',
            norm_family='imagenet',
            update_image_features=False,
            train_only=False,
    ):
        self.args = args
        self.root = root
        self.phase = phase
        self.split = split
        self.norm_family = norm_family
        self.update_image_features = update_image_features
        self.feat_dim = 512 if 'resnet18' in model else 2048  # todo, unify this with models
        self.device = args.device

        # attrs [115], objs [245], pairs [1962], train_pairs [1262], val_pairs [600], test_pairs [800]
        self.attrs, self.objs, self.pairs, self.train_pairs, \
        self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        self.full_pairs = list(product(self.attrs, self.objs))

        # Clean only was here
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}  # all pairs [1962], for val in training

        if train_only and self.phase == 'train':
            print('  Using only train pairs during training')
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}  # train pairs [1262]
        else:
            print('  Using all pairs as classification classes during {} process'.format(self.phase))
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}  # all pairs [1962]

        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('  Invalid training phase')
        print('  Use data from {} set'.format(self.phase))

        self.all_data = self.train_data + self.val_data + self.test_data

        # Keeping a list of all pairs that occur with each object
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data + self.test_data if obj == _obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj == _obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Load based on what to output
        self.transform = dataset_transform(self.phase, self.norm_family)
        self.loader = ImageLoader(ospj(self.root, 'images'))

        if not self.update_image_features:
            feat_file = ospj(root, model + '_feature_vectors.t7')
            if not os.path.exists(feat_file):
                print('  Feature file not found. Now get one!')
                with torch.no_grad():
                    self.generate_features(feat_file, model)
            self.phase = phase
            print(f'  Using {model} and feature file {feat_file}')
            activation_data = torch.load(feat_file)
            self.activations = dict(
                zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)

    def parse_split(self):
        '''
        Helper function to read splits of object attribute pair
        Returns
            all_attrs: List of all attributes
            all_objs: List of all objects
            all_pairs: List of all combination of attrs and objs
            tr_pairs: List of train pairs of attrs and objs
            vl_pairs: List of validation pairs of attrs and objs
            ts_pairs: List of test pairs of attrs and objs
        '''

        def parse_pairs(pair_list):
            '''
            Helper function to parse each phase to object attribute vectors
            Inputs
                pair_list: path to textfile
            '''
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [line.split() for line in pairs]
                pairs = list(map(tuple, pairs))

            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.root, self.split, 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.root, self.split, 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.root, self.split, 'test_pairs.txt')
        )

        # now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        '''
        Helper method to read image, attrs, objs samples

        Returns
            train_data, val_data, test_data: List of tuple of image, attrs, obj
        '''
        data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))

        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], \
                                        instance['obj'], instance['set']
            curr_data = [image, attr, obj]

            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                # Skip incomplete pairs, unknown pairs and unknown set
                continue

            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val':
                val_data.append(curr_data)
            else:
                test_data.append(curr_data)

        return train_data, val_data, test_data

    def generate_features(self, out_file, model):
        '''
        Inputs
            out_file: Path to save features
            model: String of extraction model
        '''
        # data = self.all_data
        data = ospj(self.root, 'images')
        files_before = glob(ospj(data, '**', '*.jpg'), recursive=True)
        files_all = []
        for current in files_before:
            parts = current.split('/')
            if "cgqa" in self.root:
                files_all.append(parts[-1])
            else:
                files_all.append(os.path.join(parts[-2], parts[-1]))
        transform = dataset_transform('test', self.norm_family) # Do not use any image augmentation, because we have a trained image backbone
        feat_extractor = get_image_extractor(arch=model).eval()
        feat_extractor = feat_extractor.to(self.device)

        image_feats = []
        image_files = []
        for chunk in tqdm(chunks(files_all, 1024), total=len(files_all) // 1024, desc=f'Extracting features {model}'):
            files = chunk
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).to(self.device))
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, dim=0)
        print('features for %d images generated' % (len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)

    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        index = self.sample_indices[index]

        image, attr, obj = self.data[index]

        # Decide what to output
        if not self.update_image_features:
            if self.args.dataset == 'mit-states':
                pair, img = image.split('/')
                pair = pair.replace('_', ' ')
                image = pair + '/' + img
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)

        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

        return data

    def __len__(self):
        '''
        Call for length
        '''
        return len(self.sample_indices)
