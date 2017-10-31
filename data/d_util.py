import os
import pandas as pd

def mk_reshaped_train_data(path, before_path, after_path, class_path):
    df = pd.read_csv(path)
    
    #before
    b_s = []
    #dump before
    d_b_s = []
    #after
    a_s = []
    #dump after
    d_a_s = []
    #class
    class_ = []
    #dump_class
    d_class_ = []
    s_id = 0
    for key, row in df.iterrows():
        v_s_id = row["sentence_id"]
        v_before = row["before"]
        v_after = row["after"]
        v_class = row["class"]

        if v_s_id != s_id:
            s_id = v_s_id
            
            b_s.append(" ".join(d_b_s))
            a_s.append(" ".join(d_a_s))
            class_.append(" ".join(d_class_))

            d_b_s = []
            d_a_s = []
            d_class_ =[]
            
            d_b_s.append(str(v_before))
            d_a_s.append(str(v_after))
            d_class_.append(str(v_class))
            continue

        d_b_s.append(str(v_before))
        d_a_s.append(str(v_after))
        d_class_.append(str(v_class))

    ##write data
    with open(before_path, "a") as fs:
        fs.write("\n".join(b_s))

    with open(after_path, "a") as fs:
        fs.write("\n".join(a_s))

    with open(class_path, "a") as fs:
        fs.write("\n".join(class_))
        
mk_reshaped_train_data("en_train.csv", "train_before_sentence.txt", "train_after_sentence.txt", "train_class.txt")
