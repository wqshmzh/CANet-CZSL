import torch.optim as optim
from model.image_extractor import get_image_extractor
from model.CANet import CANet

def configure_model(args, dataset, train=True):
    image_extractor = None
    if args.update_image_features:
        if args.rank == 0: print('> Initialize feature extractor <{}>'.format(args.image_extractor))
        image_extractor = get_image_extractor(args, arch=args.image_extractor, pretrained=True)
        if not args.extract_feature_vectors:
            import torch.nn as nn
            image_extractor = nn.Sequential(*list(image_extractor.children())[:-1])
        image_extractor = image_extractor.to(args.device)

    print('> Initialize model <CANet>')
    model = CANet(dataset, args).to(args.device)

    # configuring optimizer
    if train:
        print('> Initialize optimizer <Adam>')
        model_params = [param for _, param in model.named_parameters() if param.requires_grad]
        optim_params = [{'params':model_params}]
        if args.update_image_features:
            ie_parameters = [param for _, param in image_extractor.named_parameters()]
            optim_params.append({'params': ie_parameters, 'lr': args.lrg})
        optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)
        return image_extractor, model, optimizer
    else:
        return image_extractor, model