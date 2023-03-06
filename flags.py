import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Program parameters
parser.add_argument('--dataset', default='ut-zap50k', help='mit-states|ut-zap50k|cgqa') # YOU HAVE TO CHOOSE A DATASET
parser.add_argument('--splitname', default='compositional-split-natural', help="dataset split")
parser.add_argument('--image_extractor', default = 'resnet18', help = 'image feature extractor')
parser.add_argument('--norm_family', default = 'imagenet', help = 'Normalization values from dataset')
parser.add_argument('--test_set', type=str, help='val|test mode')
parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size at test/eval time")
parser.add_argument('--cpu_eval', action='store_true', help='Perform test on cpu')
parser.add_argument('--train_split', default='normal', help='How to split training set: normal-no split|obj|attr')

# Training hyperparameters
parser.add_argument('--topk', type=int, default=1,help="Compute topk accuracy")
parser.add_argument('--num_workers', type=int, default=16,help="Number of workers")
parser.add_argument('--batch_size', type=int, default=256,help="Training batch size")
parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
parser.add_argument('--lrg', type=float, default=1e-3, help="Learning rate for feature extractor")
parser.add_argument('--wd', type=float, default=5e-5, help="Weight decay")
parser.add_argument('--save_every', type=int, default=1,help="Frequency of snapshots in epochs")
parser.add_argument('--eval_val_every', type=int, default=1,help="Frequency of eval in epochs")
parser.add_argument('--max_epochs', type=int, default=800,help="Max number of epochs")

# Model common parameters
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of shared embedding space')
parser.add_argument('--nlayers', type=int, default=2, help='Layers in the image embedder')
parser.add_argument('--bias', type=float, default=1e3, help='Bias value for unseen concepts')
parser.add_argument('--update_image_features', action = 'store_true', default=False, help='Whether update image features extracted from vision models such as ResNet')
parser.add_argument('--update_word_features', action = 'store_true', default=True, help='Whether update word features extracted from NLP models')
parser.add_argument('--emb_type', default='fasttext', help='gl+w2v|ft+w2v|ft+gl|ft+w2v+gl|glove|word2vec|fasttext, name of embeddings to use for initializing the primitives')
parser.add_argument('--composition', default='mlp_add', help='add|mul|mlp|mlp_add, how to compose primitives')
parser.add_argument('--relu', action='store_true', default=False, help='Use relu in the last MLP layer')
parser.add_argument('--dropout', action='store_true', default=False, help='Use dropout')
parser.add_argument('--norm', action='store_true', default=False, help='Use normalization')
parser.add_argument('--train_only', default=True, help='Optimize only for train pairs')
parser.add_argument('--cosine_scale', type=float, default=20, help="Scale for cosine similarity")
parser.add_argument('--nhiddenlayers', type=int, default=0, help='Num of hidden layers of attr adapter')