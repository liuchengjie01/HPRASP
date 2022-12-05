import os


def get_file_list(input_data: str):
    file_list = []
    for dir_path, dirs, files in os.walk(input_data):
        for file in files:
            file_path = os.path.join(dir_path, file)
            if "\\" in file_path:
                file_path = file_path.replace("\\", "/")
            file_list.append(file_path)
        for dir in dirs:
            file_list.extend(get_file_list(os.path.join(dir_path, dir)))
    return file_list


def fun1(input_path: str, output: str):
    files = get_file_list(input_path)
    with open(output, mode='w', encoding='utf8') as f:
        for file in files:
            f.write("### " + file)
            with open(file, mode='r', encoding='utf8') as fp:
                lines = fp.readlines()
                f.writelines(lines)


if __name__ == '__main__':
    input_path = ""
    output_path = ""
    fun1()
