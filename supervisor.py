import torch
import os
from utils.word2vec import *
from dataloaders.rumor_dataloader import *
from dataloaders.socialbot_dataloader import *
from dataloaders.infordiffu_dataloader import *
from trainer.rumor_trainer import RumorTrainer
from trainer.socialbot_trainer import SocialbotTrainer
from trainer.infordiffu_trainer import InforDiffuTrainer
from model.RPRL import RPRL
import torch.nn as nn

def RPRL_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    if args.dataset in ['DRWeibo', 'Weibo']:

        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        if args.train_mode == 'pretrain':
            label_dataset_path = os.path.join('dataset', args.pretrain_dataset, 'dataset')
            label_source_path = os.path.join('dataset', args.pretrain_dataset, 'source')
            label_preprocess_path = os.path.join('dataset', args.pretrain_dataset, 'preprocess')

            label_source_path_target = os.path.join('dataset', args.dataset, 'source')

        elif args.train_mode == 'finetune':
            label_dataset_path = os.path.join('dataset', args.dataset, 'dataset')
            label_source_path = os.path.join('dataset', args.dataset, 'source')
            label_preprocess_path = os.path.join('dataset', args.dataset, 'preprocess')


        model_path = os.path.join('word2vec',
                            f'w2v_pretrained_{args.tokenize_mode}_{args.num_features}.model')

        # word2vec
        if not os.path.exists(model_path) and args.word_embedding == 'word2vec':
            sentences = collect_sentences(label_source_path, args.language, args.tokenize_mode)
            sentences_target = collect_sentences(label_source_path_target, args.language, args.tokenize_mode)
            sentences.extend(sentences_target)
            w2v_model = train_word2vec(sentences, args.num_features, args.seed)
            w2v_model.save(model_path)
    
        word2vec = Embedding(model_path, args.language, args.tokenize_mode) if args.word_embedding == 'word2vec' else None

        # load data

        split_dataset(label_source_path, label_dataset_path, k_shot=args.k, split=args.split)

        train_path = os.path.join(label_dataset_path, 'train')
        val_path = os.path.join(label_dataset_path, 'val')
        test_path = os.path.join(label_dataset_path, 'test')

        undirected = args.undirected
        
        os.makedirs(label_preprocess_path, exist_ok=True)
    
        train_dataset = RumorDataloader(train_path, args.word_embedding, word2vec, undirected, args=args)
        val_dataset = RumorDataloader(val_path, args.word_embedding, word2vec, undirected, args=args)
        test_dataset = RumorDataloader(test_path, args.word_embedding, word2vec, undirected, args=args)

        datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        model =  RPRL(args=args, device=device).to(device)
        if args.train_mode == 'finetune':
            state = torch.load("pretrained_models/"+args.pretrain_dataset+"_rprl_pretrained.pth", map_location=device)
            model.load_state_dict(state, strict=False)  
        
        optimizer = model.init_optimizer(args)

        datasets = [train_dataset, val_dataset, test_dataset]

        trainer = RumorTrainer(datasets, model, optimizer, args, device)

        acc, auc, f1 = trainer.train_process()

        scores = [acc, auc, f1]

    elif args.dataset in ['twibot-22', 'MGTAB']:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        if args.train_mode == 'pretrain':
            dataloader = SocialBotDataLoader(args.pretrain_dataset, args, device=device)
        elif args.train_mode == 'finetune':
            dataloader = SocialBotDataLoader(args.dataset, args, device=device)

        dataset = dataloader.load_data()

        model = RPRL(args=args, device=device).to(device)
        if args.train_mode == 'finetune':
            state = torch.load("pretrained_models/"+args.pretrain_dataset+"_rprl_pretrained.pth", map_location=device)
            model.load_state_dict(state, strict=False)  

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(),
                            lr=args.lr,weight_decay=args.weight_decay)

        trainer = SocialbotTrainer(dataset, model, optimizer, loss, args, device)

        acc, auc, f1 = trainer.train_process()

        scores = [acc, auc, f1]
    
    elif args.dataset in ['christianity', 'twitter']:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        if args.train_mode == 'pretrain':
            user_size, total_cascades, timestamps, train, valid, test = split_data(args, args.pretrain_dataset,
                                                                            load_dict=True, max_len=args.max_len)
            
            relation_graph = build_social_graph(args.pretrain_dataset)
        elif args.train_mode == 'finetune':
            user_size, total_cascades, timestamps, train, valid, test = split_data(args, args.dataset,
                                                                            load_dict=True, max_len=args.max_len)
            
            relation_graph = build_social_graph(args.dataset)
        
        train_data = InforDiffuDataLoader(args, train, args.batch_size, load_dict=True, cuda=False, _need_shsuffle=True)
        valid_data = InforDiffuDataLoader(args, valid, args.batch_size, load_dict=True, cuda=False, _need_shsuffle=False)
        test_data = InforDiffuDataLoader(args, test, args.batch_size, load_dict=True, cuda=False, _need_shsuffle=False)

        dataset = [train_data, valid_data, test_data, relation_graph]

        args.user_size = user_size

        model = RPRL(args=args, device=device).to(device)

        params = model.parameters()

        loss = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        trainer = InforDiffuTrainer(dataset, model, optimizer, loss, args, device)

        scores = trainer.train_process()

    return scores


