
ori_data = "data/data_SQLv3.csv"
train_data = "data/train_SQLiV3.txt"
dev_data = "data/dev_SQLiV3.txt"
test_data = "data/test_SQLiV3.txt"
if __name__ == '__main__':
    with open(ori_data, mode='r', encoding='utf8') as f:
        with open(train_data, mode='w', encoding='utf8') as train:
            with open(dev_data, mode='w', encoding='utf8') as dev:
                with open(test_data, mode='w', encoding='utf8') as test:
                    cnt = 0
                    for line in f:
                        if cnt == 9:
                            cnt = -1
                            dev.write(line)
                        elif cnt == 8:
                            test.write(line)
                        else:
                            train.write(line)
                        cnt += 1
