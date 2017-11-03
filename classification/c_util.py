import os
import random
import numpy as np

def char_byte(str_):
    return len(str_.encode("utf-8"))

def read_dict(path):
    with open(path) as fs:
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

def convert_s2WIdx_eFeature(sentences, indexs, time_step, word_length):
    r = []
    fr = []
    for sentence in sentences:
        words = sentence.split("||")
        t = []
        ft = []
        for i, word in enumerate(words[:-1]):
            converted = [indexs.index(char) for char in word]
            ft.append([len(word.encode("utf-8")), len(word), i])
            while len(converted) != word_length and len(converted) <= word_length:
                converted.append(len(indexs))
            t.append(converted[:word_length])
            
        while len(t) != time_step and len(t) <= time_step:
            i+=1
            t.append([len(indexs)+1]*word_length)
            ft.append([0,0,i])
        
        r.append(t[:time_step])
        fr.append(ft[:time_step])
    return np.array(r), np.array(fr)

def convert_label(labels, label_dict, time_step):
    b_ = []
    for label in labels:
        t_=[]
        for t_label in label.split("||"):
            content = [0]*(len(label_dict)+1)
            content[label_dict.index(t_label)] = 1
            t_.append(content)

        while len(t_) != time_step and len(t_) <= time_step:
            content = [0]*(len(label_dict)+1)
            content[-1] = 1
            t_.append(content)

        b_.append(t_[:time_step])
    return np.array(b_)

def mk_train_func(training_data_path, class_data_path, char_dict_path, class_dict_path, batch_size, max_time_step, max_word_length=10, test=True, p=0.3):
    training_data = np.array(read_dict(training_data_path))
    label_data = np.array(read_dict(class_data_path))
    char_dict = read_dict(char_dict_path)
    class_dict = read_dict(class_dict_path)
    
    test_data = None
    test_label_data = None
    if test:
        choiced_idx = np.array([idx for idx in range(len(training_data)) if np.random.binomial(n=1, p=p)])
        test_data = training_data[choiced_idx]
        test_label_data = label_data[choiced_idx]
        
        choiced_idx = np.delete(range(len(training_data)), choiced_idx).tolist()
        training_data =  training_data[choiced_idx]
        label_data = label_data[choiced_idx]

    def train_func():
        data_size = len(training_data)

        ##feature
        ##byte size of word, charactor length, token
        while True:
            choiced_idx = np.random.choice(data_size, batch_size)
            choiced_s = training_data[choiced_idx].tolist()
            choiced_label = label_data[choiced_idx].tolist()

            data, feature = convert_s2WIdx_eFeature(choiced_s, char_dict, max_time_step, max_word_length)
            label = convert_label(choiced_label, class_dict, max_time_step)
            yield data, feature, label

    def test_func():
        nonlocal test_data
        nonlocal test_label_data
        end_s = -(len(test_data)%batch_size)
        test_data = test_data[:end_s]
        test_label_data = test_label_data[:end_s]
        while True:
            choiced_s = test_data[:batch_size]
            choiced_l = test_label_data[:batch_size]

            data, feature = convert_s2WIdx_eFeature(choiced_s, char_dict, max_time_step, max_word_length)
            label = convert_label(choiced_l, class_dict, max_time_step)

            test_data = np.delete(test_data, np.arange(batch_size))
            test_label_data = np.delete(test_label_data, np.arange(batch_size))

            yield data, feature, label, choiced_s, choiced_l

    if test:
        return train_func, test_func

    return train_func

def mk_score_board(sentences, labels, outputs, ans):
    #output shape =  [batch_size, time_step , label_size]
    outputs = np.argmax(outputs)
    ans = np.argmax(ans)
    r = []
    for i, (sentence, label) in enumerate(zip(sentences, labels)):
        words = sentence.split("||")
        s_labels = label.split("||")
        t = []
        for j, word, s_label in enumerate(words, s_labels):
            t.append(",".join([str(i), word, s_label, str(outputs[i,j]), "TRUE" if outputs[i,j]==ans[i,j] else "FALSE"]))
        r.append(t)
    return "\n".join(r)


        

#save_dict(mk_word_dict(["../data/test_before_sentence_.txt", "../data/train_before_sentence_.txt"]), "../data/word_dict.txt")
#save_dict(mk_charactor_dict(["../data/test_before_sentence_.txt", "../data/train_before_sentence_.txt"]), "../data/char_dict.txt")
#save_dict(mk_class_dict("../data/en_train.csv"), "../data/class_dict.txt")
