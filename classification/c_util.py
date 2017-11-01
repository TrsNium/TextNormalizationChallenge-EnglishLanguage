import os
import random
import numpy as np

def char_byte(str_):
    return len(str_.encode("utf-8"))

def read_dict(path):
    with opne(path) as fs:
        lines = [line.split('\n')[0] for line in fs.readlines()]
    return lines

def save_dict(dict_, save_path):
    with open(save_path, "a") as fs:
        fs.write("\n".join(dict_))
    return None

def mk_word_dict(paths):
    lines = []
    for path in paths:
        with open(path) as fs:
            lines += [line.split("\n")[0] for line in fs.readlines()]
    return sorted(list(set(list("||".join(lines).split("||")))))

def mk_charactor_dict(paths):
    lines = []
    for path in paths:
        with open(path) as fs:
            lines += [line.split("\n")[0] for line in fs.readlines()]
    
    return sorted(list(set(list("".join(lines)))))

def mk_class_dict(path):
    import pandas as pd
    df = pd.read_csv(path)
    return list(set(df["class"]))

def convert_sentence2word_idx(sentences, indexs, time_step, word_length):
    r = []
    for sentence in sentences:
        words = sentence.split(" ")
        t = []
        for word in words[:-1]:
            converted = [indexs.index(char) for char in word]
            while len(converted) != word_length and len(converted) <= word_length:
                converted.append(len(indexs))
            t.append(converted[:word_length])
            
        while len(t) != time_step and len(t) <= time_step:
            t.insert(0, [len(indexs)+1]*word_length)
        
        r.append(t[:time_step])
    return r    

def convert_label(labels):
    r = []
    for label in labels:
        content = [0]*2
        content[int(label)] = 1
        r.append(content)
    return np.array(r)

def mk_train_func(training_data_path, class_data_path, char_dict_path, class_dict_path, batch_size, word_length, max_time_step, max_word_length=10, test=True, p=0.7):
    training_data = read_dict(training_data_path)
    label_data = read_dict(class_data_path)
    char_dict = read_dict(char_dict_path)
    class_dict = read_dict(class_dict_path)
    
    if test:
        choiced_idx = [idx for idx in range(len(training_data)) if np.random.binomial(n=1, p)]
        training_data = [training_data[idx] for idx in choiced_idx]
        label_data = [label_data[idx] for idx in choiced_idx]
        test_data = [training_data[idx] for idx in range(len(training_data)) if idx not in choiced_idx]
        test_label_data = [label_data[idx] for idx in range(len(label_data)) if idx not in choiced_idx]

    def train_func():
        data_size = len(training_data)
        training_data = np.array(training_data)
        label_data = np.array(label_data)
        while True:
            choiced_idx = np.random.choice(data_size, batch_size)
            choiced_s = training_data[choiced_idx].tolist()

            


            yield

    def test_func():
        while True:

            yield
        

    if test:
        return train_func, test_func

    return train_func

#save_dict(mk_word_dict(["../data/test_before_sentence_.txt", "../data/train_before_sentence_.txt"]), "../data/word_dict.txt")
#save_dict(mk_charactor_dict(["../data/test_before_sentence_.txt", "../data/train_before_sentence_.txt"]), "../data/char_dict.txt")
#save_dict(mk_class_dict("../data/en_train.csv"), "../data/class_dict.txt")
