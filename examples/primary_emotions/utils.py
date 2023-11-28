import json
import numpy as np
import random
import os

def primary_emotions_concept_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[emotion] = list(set(json.load(file)))[:200]

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)] # zip iterates over the shortest list
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data


def primary_emotions_concept_dataset_zh(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    template_str = '{user_tag} 考虑如下场景中的{emotion}情绪:\n场景: {scenario}\n答案: {assistant_tag} '
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[emotion] = list(set(json.load(file)))[:200]

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)] # zip iterates over the shortest list
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data

def primary_emotions_function_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    train_template_str = '{user_tag} Act as if you are extremely {emo}. {assistant_tag} {scenario}' 
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    with open(os.path.join(data_dir, "all_truncated_outputs.json"), 'r') as file:
        all_truncated_outputs = json.load(file)
    
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    emotions_adj = [
        ("joyful", "happy", "cheerful"), 
        ("sad", "depressed", "miserable"),
        ("angry", "furious", "irritated"),
        ("fearful", "scared", "frightened"),
        ("disgusted", "sicken", "revolted"), 
        ("surprised", "shocked", "astonished")
    ]
    emotions_adj_ant = [
        ("dejected", "unhappy", "dispirited"), 
        ("cheerful", "optimistic", "happy"),
        ("pleased", "calm", "peaceful"),
        ("fearless", "bold", "unafraid"),
        ("approved", "delighted", "satisfied"), 
        ("unimpressed", "indifferent", "bored")
    ]

    formatted_data = {}
    for emotion, emotion_adj, emotion_adj_ant in zip(emotions, emotions_adj, emotions_adj_ant):
        emotion_train_data_tmp = [[
            train_template_str.format(emo=np.random.choice(emotion_adj), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag), 
            train_template_str.format(emo=np.random.choice(emotion_adj_ant), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag)
        ] for s in all_truncated_outputs]
        
        train_labels = []
        for d in emotion_train_data_tmp:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])

        emotion_train_data = np.concatenate(emotion_train_data_tmp).tolist()

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
        }
    return formatted_data

def primary_emotions_function_dataset_zh(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    train_template_str = '{user_tag} 请表现出你当前的情绪非常{emo}. {assistant_tag} {scenario}' 
    with open(os.path.join(data_dir, "all_truncated_outputs.json"), 'r') as file:
        all_truncated_outputs = json.load(file)
    
    emotions = ["幸福", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶"]
    emotions_adj = [
        ("高兴", "快乐", "愉快"),
        ("悲伤", "沮丧", "痛苦"),
        ("愤怒", "暴怒", "恼怒"),
        ("恐惧", "害怕", "惊恐"),
        ("厌恶", "厌恶", "反感"),
        ("惊讶", "震惊", "惊讶")
    ]
    emotions_adj_ant = [
        ("沮丧", "不快乐", "失落"), 
        ("快乐", "乐观", "快乐"),
        ("高兴", "平静", "平静"),
        ("无畏", "大胆", "无畏"),
        ("赞成", "高兴", "满意"), 
        ("无动于衷", "冷漠", "无聊")
    ]

    formatted_data = {}
    for emotion, emotion_adj, emotion_adj_ant in zip(emotions, emotions_adj, emotions_adj_ant):
        emotion_train_data_tmp = [[
            train_template_str.format(emo=np.random.choice(emotion_adj), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag), 
            train_template_str.format(emo=np.random.choice(emotion_adj_ant), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag)
        ] for s in all_truncated_outputs]
        
        train_labels = []
        for d in emotion_train_data_tmp:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])

        emotion_train_data = np.concatenate(emotion_train_data_tmp).tolist()

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
        }
    return formatted_data