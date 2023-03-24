import time

from requests import Session
from tqdm import tqdm


def get_payloads(data_path):
    payloads = []
    with open(data_path, mode='r', encoding='utf-8') as fp:
        for line in fp:
            if not line.startswith("###"):
                payloads.append(line.strip())
    return payloads


def send_request(session, payload):
    local_header = {
        "Host": "127.0.0.1:8080",
        "User-Agent": "Mozilla/5.0(X11;Ubuntu;Linux x86_64;rv: 88.0) Gecko/20100101 Firefox/88.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip,deflate",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Cookie": "JSESSIONID=F470BEB4E60445BC07191F29D43207EF",
        "Upgrade-Insecure-Requests": 1,
    }
    local_params = {
        "id": payload
    }
    resp = session.get(url='http://127.0.0.1:8080/vulns/012-jdbc-mysql.jsp', headers=local_header, params=local_params)
    return resp


def run_test():
    session = Session()
    resp = send_request(session, "1 or 2=2 or 3=3 #cvkr")
    print(resp.status_code)
    print(resp.url)
    print(resp.text)


def run_lab2(data_path):
    # session = Session()
    payloads = get_payloads(data_path)
    payloads = payloads[:100]
    succ_cnt = 0
    block_cnt = 0
    total_cnt = len(payloads)
    other_cnt = 0
    block_correct = block_wrong = 0  # 1
    succ_correct = succ_wrong = 0  # 0
    cnt = 0
    for payload in tqdm(payloads):
        param = payload.split('\t')
        print(param)
        assert len(param) == 2
        p = param[0]
        tag = param[1]
        cnt += 1
        print("param{}: {}, tag:{}".format(cnt, p, tag))
    #     resp = send_request(session, p)
    #     time.sleep(0.2)
    #     # print("tag: |{}|".format(tag))
    #     if resp.status_code == 200:
    #         succ_cnt += 1
    #         if tag == "0":
    #             succ_correct += 1
    #         elif tag == "1":
    #             succ_wrong += 1
    #         else:
    #             raise ValueError("Tag must be 0 or 1, var: |{}|".format(tag))
    #     elif resp.status_code == 403 and str(resp.url).startswith("https://www.baidu.com"):
    #     # elif (resp.status_code == 404 or resp.status_code == 400) \
    #     #         and str(resp.url).startswith("https://rasp.baidu.com/blocked/"):
    #         block_cnt += 1
    #         if tag == "0":
    #             block_wrong += 1
    #         elif tag == "1":
    #             block_correct += 1
    #         else:
    #             raise ValueError("Tag must be 0 or 1, var: |{}|".format(tag))
    #     else:
    #         other_cnt += 1
    # print("Total count: {}, succ_correct: {}, succ_wrong: {}, block_correct: {}, block_wrong: {}"
    #       .format(total_cnt, succ_correct, succ_wrong, block_correct, block_wrong))


def run_lab1(data_path):
    session = Session()
    payloads = get_payloads(data_path)
    # payloads = payloads[:100]
    cnt = len(payloads)
    success_cnt = 0
    block_cnt = 0
    other_cnt = 0
    with open("success.txt", mode="w", encoding="utf8") as succ:
        with open("block.txt", mode="w", encoding="utf8") as bl:
            for payload in tqdm(payloads):
                resp = send_request(session, payload)
                time.sleep(0.2)
                if resp.status_code == 200:
                    success_cnt += 1
                    succ.write(payload+"\r\n")
                elif resp.status_code == 403 and str(resp.url).startswith("https://www.baidu.com"):
                # elif (resp.status_code == 404 or resp.status_code == 400) \
                #         and str(resp.url).startswith("https://rasp.baidu.com/blocked/"):
                    block_cnt += 1
                    bl.write(payload+"\r\n")
                else:
                    other_cnt += 1
    print('Total cnt: {}, success: {}, block: {}, other_cnt: {}'.format(cnt, success_cnt, block_cnt, other_cnt))


if __name__ == '__main__':
    data_path1 = "/mnt/hgfs/VirtualMachineImage/sql-injection-payload-list/Result.txt"
    data_path2 = "/mnt/hgfs/VirtualMachineImage/test_lab2.txt"
    # run_lab1(data_path1)
    run_lab2(data_path2)
    # run_test()
