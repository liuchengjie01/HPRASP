import os


def get_file_list(input_data: str):
    file_list = []
    for dir_path, dirs, files in os.walk(input_data):
        for file in files:
            file_path = os.path.join(dir_path, file)
            file_list.append(file_path)
    return file_list


def fun1(input_path: str, output: str):
    files = get_file_list(input_path)
    for file in files:
        print(file)
    with open(output, mode='w', encoding='utf-8') as f:
        for file in files:
            f.write("### " + file)
            print("## " + file)
            with open(file, mode='r', encoding='utf-8', errors='ignore') as fp:
                for line in fp:
                    f.write(line)
                    # print(line.strip())


if __name__ == '__main__':
    input_path = "D:\\VirtualMachineImage\\sql-injection-payload-list\\Intruder"
    output_path = "D:\\VirtualMachineImage\\sql-injection-payload-list\\Result.txt"
    fun1(input_path, output_path)
