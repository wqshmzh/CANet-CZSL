import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.word_embedding import load_word_embeddings
from model.common import MLP

class HyperNet(nn.Module):
    def __init__(self, struct):
        super(HyperNet, self).__init__()
        self.struct = struct  # channel config of the primary network
        for key, value in struct.items():
            setattr(self, key, nn.Sequential(
                nn.Linear(value[0], value[1])
                ))

    def forward(self, control_signal):
        weights = {}
        for key, _ in self.struct.items():
            weight = getattr(self, key)(control_signal)
            weights[key] = weight
        return weights

class HyperNetStructure():
    '''
    This class only defines the structure of Attribute Hyper Learner.
    '''
    def __init__(self, input_dim, num_hiddenLayer, hidden_dim, output_dim):
        self.structure = {}
        if num_hiddenLayer == 0:
            self.structure['InGenerator'] = [input_dim, hidden_dim]
            self.structure['OutGenerator'] = [input_dim, output_dim]
        else:
            self.structure['InGenerator'] = [input_dim, hidden_dim]
            for i in range(num_hiddenLayer):
                self.structure['HiddenGenerator_{}'.format(i+1)] = [input_dim, hidden_dim]
            self.structure['OutGenerator'] = [input_dim, output_dim]
    
    def get_structure(self):
        return self.structure

class AttrAdapter(nn.Module):
    def __init__(self, input_dim, hypernet_struct, relu=True):
        super().__init__()
        self.hypernet_struct = hypernet_struct
        self.hyper_net = HyperNet(struct=hypernet_struct)
        self.num_generator = len(hypernet_struct)
        self.relu = relu
        i = 0
        incoming = input_dim
        for _, value in hypernet_struct.items():
            setattr(self, 'AdapterLayer_{}'.format(i), nn.Linear(incoming, value[1]))
            if i < self.num_generator - 1:
                setattr(self, 'AdapterLayer_{}_extend'.format(i), nn.Sequential(
                    nn.LayerNorm(value[1]),
                    nn.ReLU(True),
                    nn.Dropout()
                    ))
            incoming = value[1]
            i += 1
        if relu:
            setattr(self, 'AdapterLastReLU', nn.ReLU(True))
    
    def forward(self, control_signal, attr_emb):
        batch_size = control_signal.shape[0] # 256
        num_attr = attr_emb.shape[0] # 115
        attr_emb = attr_emb.unsqueeze(dim=0).repeat(batch_size, 1, 1) # 256x115x300
        weights = self.hyper_net(control_signal)
        i = 0
        for key, _ in self.hypernet_struct.items():
            attr_emb = getattr(self, 'AdapterLayer_{}'.format(i))(attr_emb)
            weights_extend = weights[key].unsqueeze(dim=1).repeat(1, num_attr, 1)
            attr_emb *= weights_extend
            if i < self.num_generator - 1:
                attr_emb = getattr(self, 'AdapterLayer_{}_extend'.format(i))(attr_emb)
            i += 1
        if self.relu:
            attr_emb = getattr(self, 'AdapterLastReLU')(attr_emb)
        return attr_emb

class CANet(nn.Module):
    def __init__(self, dset, args):
        super(CANet, self).__init__()
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(args.device)
            objs = torch.LongTensor(objs).to(args.device)
            pairs = torch.LongTensor(pairs).to(args.device)
            return attrs, objs, pairs

        # Validation - Use all pairs to validate
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        # All attrs and objs without repetition
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(args.device), \
                                          torch.arange(len(self.dset.objs)).long().to(args.device)
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
                
        '''========== Word embedder for attrs and objs =========='''
        attr_word_emb_file = '{}_{}_attr.save'.format(args.dataset, args.emb_type)
        attr_word_emb_file = os.path.join(args.main_root, 'word embedding', attr_word_emb_file)
        obj_word_emb_file = '{}_{}_obj.save'.format(args.dataset, args.emb_type)
        obj_word_emb_file = os.path.join(args.main_root, 'word embedding', obj_word_emb_file)

        print('  Load attribute word embeddings--')
        pretrained_weight_attr = load_word_embeddings(dset.attrs, args)
        emb_dim = pretrained_weight_attr.shape[1]
        self.attr_embedder = nn.Embedding(len(dset.attrs), emb_dim).to(args.device)
        self.attr_embedder.weight.data.copy_(pretrained_weight_attr)

        print('  Load object word embeddings--')
        pretrained_weight_obj = load_word_embeddings(dset.objs, args)
        self.obj_embedder = nn.Embedding(len(dset.objs), emb_dim).to(args.device)
        self.obj_embedder.weight.data.copy_(pretrained_weight_obj)
        '''======================================================'''

        '''====================== HyperNet ======================'''
        AttrHyperNet_struct = HyperNetStructure(input_dim=emb_dim, num_hiddenLayer=args.nhiddenlayers, 
            hidden_dim=emb_dim*2, output_dim=emb_dim)
        self.AttrHyperNet_struct = AttrHyperNet_struct.get_structure()
        '''======================================================'''

        '''=================== Attr adapter ====================='''
        self.attr_adapter = AttrAdapter(input_dim=emb_dim, hypernet_struct=self.AttrHyperNet_struct, relu=False)
        '''======================================================'''

        '''======================= Mapper ======================='''
        self.image_embedder_attr = MLP(dset.feat_dim, emb_dim, num_layers=args.nlayers, relu=False, bias=True,
                                       dropout=True, norm=True, layers=[])
        self.image_embedder_obj = MLP(dset.feat_dim, emb_dim, num_layers=args.nlayers, relu=False, bias=True,
                                      dropout=True, norm=True, layers=[])
        self.image_embedder_both = MLP(dset.feat_dim, emb_dim, num_layers=args.nlayers, relu=False, bias=True,
                                       dropout=True, norm=True, layers=[])
        '''======================================================'''

        # static inputs
        if not args.update_word_features:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False
        
        self.projection = MLP(emb_dim*2, emb_dim, relu=True, bias=True, dropout=True, norm=True,
                              num_layers=2, layers=[])

        self.img_obj_compose = MLP(emb_dim+dset.feat_dim, emb_dim, relu=True, bias=True, dropout=True, norm=True,
                                   num_layers=2, layers=[emb_dim])
        
        self.alpha = args.alpha # weight factor
        self.τ = args.cosine_scale # temperature factor
                
    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], -1)
        output = self.projection(inputs)
        return output

    def val_forward(self, input_batch):
        x = input_batch[0]
        del input_batch
        # Map the input image embedding
        ω_a_x = self.image_embedder_attr(x)
        ω_c_x = self.image_embedder_both(x)
        ω_o_x = self.image_embedder_obj(x)

        # Acquire word embeddings of all attrs and objs
        v_a = self.attr_embedder(self.uniq_attrs)
        v_o = self.obj_embedder(self.uniq_objs)
        
        # Pred obj  
        ω_o_x_norm = F.normalize(ω_o_x, dim=-1)
        v_o_norm = F.normalize(v_o, dim=-1)
        d_cos_oi = ω_o_x_norm @ v_o_norm.t()
        P_oi = (d_cos_oi + 1) / 2
        o_star = torch.argmax(d_cos_oi, dim=-1)
        v_o_star = self.obj_embedder(o_star)
        
        # Pred attr
        β = self.img_obj_compose(torch.cat((v_o_star, x), dim=-1))
        e_a = self.attr_adapter(β, v_a)
        ω_a_x_norm = F.normalize(ω_a_x, dim=-1)
        e_a = F.normalize(e_a, dim=-1)
        d_cos_ei = torch.einsum('bd,bad->ba', ω_a_x_norm, e_a)
        P_ei = (d_cos_ei + 1) / 2

        # Pred composition
        pair_embed = self.compose(self.val_attrs, self.val_objs)
        ω_c_x = F.normalize(ω_c_x, dim=-1)
        pair_embed = F.normalize(pair_embed, dim=-1)
        d_cos_ci = ω_c_x @ pair_embed.t()
        P_ci = (d_cos_ci + 1) / 2

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = (1 - self.alpha) * P_ci[:, self.dset.all_pair2idx[pair]]
            scores[pair] += self.alpha * (P_ei[:, self.dset.attr2idx[pair[0]]] * P_oi[:, self.dset.obj2idx[pair[1]]])
        
        return None, scores

    def train_forward(self, input_batch):
        x, a, o, c = input_batch[0], input_batch[1], input_batch[2], input_batch[3]
        del input_batch

        # Map the input image embedding
        ω_a_x = self.image_embedder_attr(x)
        ω_c_x = self.image_embedder_both(x)
        ω_o_x = self.image_embedder_obj(x)

        # Acquire word embeddings of all attrs and objs
        v_a = self.attr_embedder(self.uniq_attrs)
        v_o = self.obj_embedder(self.uniq_objs)

        # Pred obj
        ω_o_x = F.normalize(ω_o_x, dim=-1)
        v_o = F.normalize(v_o, dim=-1)
        d_cos_oi = ω_o_x @ v_o.t() # Eq.2
        o_star = torch.argmax(d_cos_oi, dim=-1)
        d_cos_oi = d_cos_oi / self.τ
        v_o_star = self.obj_embedder(o_star)

        # Pred attr
        β = self.img_obj_compose(torch.cat((v_o_star, x), dim=-1)) # Eq.3
        e_a = self.attr_adapter(β, v_a) # Eq.5
        ω_a_x = F.normalize(ω_a_x, dim=-1)
        e_a = F.normalize(e_a, dim=-1)
        d_cos_ei = torch.einsum('bd,bad->ba', ω_a_x, e_a) # Eq.7
        d_cos_ei = d_cos_ei / self.τ

        # Pred composition
        v_ao = self.compose(self.train_attrs, self.train_objs)
        ω_c_x = F.normalize(ω_c_x, dim=-1)
        v_ao = F.normalize(v_ao, dim=-1)
        d_cos_ci = ω_c_x @ v_ao.t() # Eq.8
        d_cos_ci = d_cos_ci / self.τ
        
        L_o = F.cross_entropy(d_cos_oi, o) # Eq.10
        L_a = F.cross_entropy(d_cos_ei, a) # Eq.9
        L_ao = F.cross_entropy(d_cos_ci, c) # Eq.11

        return (L_a + L_o) / 2 + L_ao, None # Eq.12

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
            return loss, pred
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
            return loss, pred
        
