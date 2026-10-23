import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import argparse
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer
import glob

CFG = {  # 训练的参数配置
    'seed': 43,
    'context_model': r'/data/yyang/Temp_Model/chinese-roberta-wwm-ext',  # 预训练模型
    'text_model': r'/data/yyang/Temp_Model/chinese-roberta-wwm-ext',  # 预训练模型
    'epochs': 100,
    'max_len': 200,
    'train_bs': 32,
    'valid_bs': 8,
    'lr': 1e-5,
    'num_workers': 0,
    'accum_iter': 4,
    'weight_decay': 1e-5,
    'device': 'cuda:0',
    'patience': 10, 
}

tokenizer = AutoTokenizer.from_pretrained(CFG['context_model'])

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed'])
device = torch.device(CFG['device'])
torch.cuda.set_device(device)

def load_sticker_features(pth, device='cpu'):
    sticker_features = dict()
    filenames = glob.glob(os.path.join(pth, '*.pt'))
    for filename in filenames:
        sticker_id = filename.split('/')[-1].split('.')[0]
        sticker_features[sticker_id] = torch.load(filename, map_location=device)
    return sticker_features

all_sticker_features = load_sticker_features('/data/yyang/Temp_Model/M^4_clip_features', device='cpu')

def load_meme_annotations(path):
    with open(path, 'r', encoding='utf-8') as f:
        meme_data = json.load(f)
    meme_caption_map = {}
    for item in meme_data:
        img_id = item['image_id']
        caption = item.get('qwen_meme_caption')
        if caption is None or caption.strip() == '':
            caption = "没有表情包描述"
        meme_caption_map[str(img_id)] = caption
    return meme_caption_map

meme_caption_map = load_meme_annotations('/data/yyang/M^4Comment_Detection/meme_dataset/meme_annotation.json')

# ========== 十个任务标签映射 ==========
TASK_CONFIG = {
    'sentiment': {  # multimodal_sentiment
        'field': 'multimodal_sentiment',
        'categories': ['neutral', 'positive', 'negative'],
        'num_classes': 3
    },
    'emotion': {    # multimodal_emotion
        'field': 'multimodal_emotion',
        'categories': ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"],
        'num_classes': 7
    },
    'intention': {  # multimodal_intention
        'field': 'multimodal_intention',
        'categories': [
            "complain", "praise", "agree", "compromise", "query", "joke", "oppose", "inform", "comfort", "apologize",
            "ask-for-help", "greet", "taunt", "introduce", "guess", "leave", "advise", "flaunt", "criticize", "thank", "other"
        ],
        'num_classes': 21
    },
    'stance': {     # multimodal_stance
        'field': 'multimodal_stance',
        'categories': ["favor", "against", "neutral"],
        'num_classes': 3
    },
    'humor': {      # multimodal_humor
        'field': 'multimodal_humor',
        'categories': ["yes", "no"],
        'num_classes': 2
    },
    'metaphor': {   # multimodal_metaphor
        'field': 'multimodal_metaphor',
        'categories': ["yes", "no"],
        'num_classes': 2
    },
    'pun': {        # multimodal_pun
        'field': 'multimodal_pun',
        'categories': ["no_pun", "semantic_pun", "homophonic_pun"],
        'num_classes': 3
    },
    'sarcasm': {    # multimodal_sarcasm
        'field': 'multimodal_sarcasm',
        'categories': ["yes", "no"],
        'num_classes': 2
    },
    'rhetoric': {   # multimodal_rhetoric
        'field': 'multimodal_rhetoric',
        'categories': [
            "none", "allusion", "exaggeration", "irony", "anthropomorphism",
            "personification", "metaphor", "simile", "contrast"
        ],
        'num_classes': 9
    }
}

# 为每个任务生成 str->int 映射字典
task_label_maps = {}
for task_name, cfg in TASK_CONFIG.items():
    categories = cfg['categories']
    task_label_maps[task_name] = {cat: idx for idx, cat in enumerate(categories)}

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    processed = []
    for item in data:
        attribute = item.get('attribute')
        if attribute == 'comment':
            text = item.get('comment_content', '')
            meme_idx = item.get('comment_meme_idx')
        elif attribute == 'reply':
            text = item.get('reply_content', '')
            meme_idx = item.get('reply_meme_idx')
        else:
            continue

        text = str(text) if text is not None else ""
        if not text.strip():
            text = "没有文字评论"

        if meme_idx is None or meme_idx == "无":
            meme_id_str = None
        else:
            meme_id_str = str(meme_idx)

        sticker_text = meme_caption_map.get(meme_id_str, "没有表情包描述")

        # 提取所有任务标签，检查有效性
        labels = {}
        valid = True
        for task_name, cfg in TASK_CONFIG.items():
            field = cfg['field']
            raw = item.get(field, '').strip().lower()
            mapping = task_label_maps[task_name]
            if raw in mapping:
                labels[task_name] = mapping[raw]
            else:
                valid = False
                break
        if not valid:
            continue  # 任意一个任务标签无效则跳过该样本

        processed.append({
            'context': text,
            'sticker': meme_id_str if meme_id_str is not None else '',
            'sticker_text': sticker_text,
            **labels  # 将各任务标签展平到字典中
        })
    return processed

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df[idx]
        return data

def collate_fn(batch):
    context_ids, context_masks, token_type_ids = [], [], []
    stickertext_ids, stickertext_masks, stickertext_type_ids = [], [], []
    sticker_features = []
    # 所有任务的标签列表，按 TASK_CONFIG 顺序组织
    task_labels_dict = {task: [] for task in TASK_CONFIG}

    for x in batch:
        sticker_id = x['sticker']
        sf = all_sticker_features.get(sticker_id, torch.zeros(2048, dtype=torch.float32))
        if sf.dim() == 2:
            sf = sf.squeeze(0)
        sticker_features.append(sf)

        text = x['context']
        token = tokenizer(text, padding='max_length', truncation=True, max_length=CFG['max_len'], return_tensors='pt')
        context_ids.append(token['input_ids'][0])
        context_masks.append(token['attention_mask'][0])
        token_type_ids.append(token['token_type_ids'][0])

        sticker_text = x['sticker_text']
        if not sticker_text or sticker_text == '没有表情包描述':
            sticker_text = '图中没有文本'
        token_st = tokenizer(sticker_text, padding='max_length', truncation=True, max_length=10, return_tensors='pt')
        stickertext_ids.append(token_st['input_ids'][0])
        stickertext_masks.append(token_st['attention_mask'][0])
        stickertext_type_ids.append(token_st['token_type_ids'][0])

        for task in TASK_CONFIG:
            task_labels_dict[task].append(x[task])

    context_ids = torch.stack(context_ids)
    context_masks = torch.stack(context_masks)
    token_type_ids = torch.stack(token_type_ids)
    stickertext_ids = torch.stack(stickertext_ids)
    stickertext_masks = torch.stack(stickertext_masks)
    stickertext_type_ids = torch.stack(stickertext_type_ids)
    sticker_features = torch.stack(sticker_features)

    labels_tensors = [torch.tensor(task_labels_dict[task]) for task in TASK_CONFIG]

    return context_ids, context_masks, token_type_ids, stickertext_ids, stickertext_masks, stickertext_type_ids, \
           sticker_features, *labels_tensors