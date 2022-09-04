
ori_data = "data/data_SQLv3.csv"
train_data = "data/train_SQLiV3.txt"
dev_data = "data/dev_SQLiV3.txt"

with open(ori_data, mode='r', encoding='utf8') as f:
    with open(train_data, mode='w', encoding='utf8') as train:
        with open(dev_data, mode='w', encoding='utf8') as dev:
            cnt = 0
            for line in f:
                if cnt - 4 == 0:
                    cnt = -1
                    dev.write(line)
                else:
                    train.write(line)
                cnt += 1
