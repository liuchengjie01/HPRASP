
import os.path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def local_min(x1, x2, x3, x4):
    return min(min(len(x1), len(x2)), min(len(x3), len(x4)))

def draw_figure(x, y, marker, label, x_label, y_label, title, target_path, sample):
    zhfont1 = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Normal.otf")
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker=marker[i], label=label[i], linestyle='solid')
    plt.title(title, fontproperties=zhfont1)
    plt.xlabel(x_label, fontproperties=zhfont1)
    plt.ylabel(y_label, fontproperties=zhfont1)
    plt.legend(loc=1)
    plt.grid(True)
    # 保存为 SVG 格式
    plt.savefig(os.path.join(target_path, sample+'_'+title+'.svg'), format="svg")
    # 显示图形
    plt.show()

def run(p1="", p2="", p3="", p4="", sample="Good", lang="en", save_path=''):
    # 1=norasp 2=pkurasp 3=openrasp 4=pkurasp+model
    use_cols = ['elapsed', 'label']
    df1 = pd.read_csv(p1, usecols=lambda c: c in use_cols)
    df2 = pd.read_csv(p2, usecols=lambda c: c in use_cols)
    df3 = pd.read_csv(p3, usecols=lambda c: c in use_cols)
    df4 = pd.read_csv(p4, usecols=lambda c: c in use_cols)
    classes = ['192.168.159.128 Disks I/O', '192.168.159.128 CPU',
               '192.168.159.128 Memory', '192.168.159.128 Network I/O']

    # no rasp
    y1_disk = df1.loc[df1['label'] == classes[0]]['elapsed']
    y1_disk = y1_disk.apply(lambda c: c * (100 if sample == "Benign" else 10) * 1000)
    y1_cpu = df1.loc[df1['label'] == classes[1]]['elapsed']
    y1_cpu = y1_cpu.apply(lambda c: c*10)
    y1_mem = df1.loc[df1['label'] == classes[2]]['elapsed']
    y1_mem = y1_mem.apply(lambda c: c*10)
    y1_net = df1.loc[df1['label'] == classes[3]]['elapsed']

    # pkurasp
    y2_disk = df2.loc[df2['label'] == classes[0]]['elapsed']
    y2_disk = y2_disk.apply(lambda c: c*10*1000)
    y2_cpu = df2.loc[df2['label'] == classes[1]]['elapsed']
    y2_cpu = y2_cpu.apply(lambda c: c*10)
    y2_mem = df2.loc[df2['label'] == classes[2]]['elapsed']
    y2_mem = y2_mem.apply(lambda c: c * 10)
    y2_net = df2.loc[df2['label'] == classes[3]]['elapsed']

    # openrasp
    y3_disk = df3.loc[df3['label'] == classes[0]]['elapsed']
    y3_disk = y3_disk.apply(lambda c: c*(100 if sample == "Benign" else 10) * 1000)
    y3_cpu = df3.loc[df3['label'] == classes[1]]['elapsed']
    y3_cpu = y3_cpu.apply(lambda c: c*10)
    y3_mem = df3.loc[df3['label'] == classes[2]]['elapsed']
    y3_mem = y3_mem.apply(lambda c: c * 10)
    y3_net = df3.loc[df3['label'] == classes[3]]['elapsed']

    # bert model
    y4_disk = df4.loc[df4['label'] == classes[0]]['elapsed']
    y4_disk = y4_disk.apply(lambda c: c * 10 * 1000)
    y4_cpu = df4.loc[df4['label'] == classes[1]]['elapsed']
    y4_cpu = y4_cpu.apply(lambda c: c * 10)
    y4_mem = df4.loc[df4['label'] == classes[2]]['elapsed']
    y4_mem = y4_mem.apply(lambda c: c * 10)
    y4_net = df4.loc[df4['label'] == classes[3]]['elapsed']

    min_cnt = local_min(y1_cpu, y2_cpu, y3_cpu, y4_cpu)
    x = []
    for i in range(min_cnt):
        x.append(i)
    x = pd.Series(x)
    if lang == 'zh':
        data_type = ['cpu', 'mem', 'disk', 'net']
        xx = [x] * 4
        y_cpu = [y1_cpu[:min_cnt], y2_cpu[:min_cnt], y3_cpu[:min_cnt], y4_cpu[:min_cnt]]
        y_mem = [y1_mem[:min_cnt], y2_mem[:min_cnt], y3_mem[:min_cnt], y4_mem[:min_cnt]]
        y_disk = [y1_disk[:min_cnt], y2_disk[:min_cnt], y3_disk[:min_cnt], y4_disk[:min_cnt]]
        y_net = [y1_net[:min_cnt], y2_net[:min_cnt], y3_net[:min_cnt], y4_net[:min_cnt]]
        labels = ['无RASP', '去掉BERT的HP-RASP', 'OpenRASP', 'HP-RASP']
        markers = ['o', '*', 's', 'x']
        draw_figure(xx, y_cpu, markers, labels, '时间编号', '性能指标', 'CPU指标', save_path, sample)
        draw_figure(xx, y_mem, markers, labels, '时间编号', '性能指标', '内存指标', save_path, sample)
        draw_figure(xx, y_disk, markers, labels, '时间编号', '性能指标', '磁盘IO指标', save_path, sample)
        draw_figure(xx, y_net, markers, labels, '时间编号', '性能指标', '网络IO指标', save_path, sample)

    elif lang == "en":
        plt.subplot(2, 2, 1)
        plt.plot(x, y1_cpu[:min_cnt], color='blue', linestyle='solid', label='no rasp')
        plt.plot(x, y2_cpu[:min_cnt], color='pink', linestyle='solid', label='HPRASP1.0')
        plt.plot(x, y3_cpu[:min_cnt], color='red', linestyle='solid', label='OpenRASP')
        plt.plot(x, y4_cpu[:min_cnt], color='green', linestyle='solid', label='HPRASP2.0')
        plt.title('CPU Metrics')
        plt.xlabel('Elapsed ID')
        plt.ylabel('Performance Metrics')
        plt.legend(loc=1)
        plt.grid(True)
        plt.savefig(os.path.join(save_path, sample+"_cpu.png"))
        # plt.show()

        plt.subplot(2, 2, 2)
        plt.plot(x, y1_mem[:min_cnt], color='blue', linestyle='solid', label='no rasp')
        plt.plot(x, y2_mem[:min_cnt], color='pink', linestyle='solid', label='HPRASP1.0')
        plt.plot(x, y3_mem[:min_cnt], color='red', linestyle='solid', label='OpenRASP')
        plt.plot(x, y4_mem[:min_cnt], color='green', linestyle='solid', label='HPRASP2.0')
        plt.title('Memory Metrics')
        plt.xlabel('Elapsed ID')
        plt.ylabel('Performance Metrics')
        plt.legend(loc=1)
        plt.grid(True)
        plt.savefig(os.path.join(save_path, sample+"_mem.png"))
        # plt.show()

        plt.subplot(2, 2, 3)
        plt.plot(x, y1_disk[:min_cnt], color='blue', linestyle='solid', label='no rasp')
        plt.plot(x, y2_disk[:min_cnt], color='pink', linestyle='solid', label='HPRASP1.0')
        plt.plot(x, y3_disk[:min_cnt], color='red', linestyle='solid', label='OpenRASP')
        plt.plot(x, y4_disk[:min_cnt], color='green', linestyle='solid', label='HPRASP2.0')
        plt.title('Disk IO Metrics')
        plt.xlabel('Elapsed ID')
        plt.ylabel('Performance Metrics')
        plt.legend(loc=1)
        plt.grid(True)
        plt.savefig(os.path.join(save_path, sample + "_disk.png"))
        # plt.show()

        plt.subplot(2, 2, 4)
        plt.plot(x, y1_net[:min_cnt], color='blue', linestyle='solid', label='no rasp')
        plt.plot(x, y2_net[:min_cnt], color='pink', linestyle='solid', label='HPRASP1.0')
        plt.plot(x, y3_net[:min_cnt], color='red', linestyle='solid', label='OpenRASP')
        plt.plot(x, y4_net[:min_cnt], color='green', linestyle='solid', label='HPRASP2.0')
        plt.title('Network IO Metrics')
        plt.xlabel('Elapsed ID')
        plt.ylabel('Performance Metrics')
        plt.legend(loc=1)
        plt.grid(True)

        plt.suptitle(sample + " Sample ")
        plt.tight_layout(pad=0.5, h_pad=1.5)
        plt.savefig(os.path.join(save_path, sample + "_net.png"))


        plt.show()


if __name__ == '__main__':
    bad_data_path1 = "S:\\VirtualMachineImage\\apache-jmeter-5.4.3\\bin\\res_lcj\\bad-norasp-more.csv"
    bad_data_path2 = "S:\\VirtualMachineImage\\apache-jmeter-5.4.3\\bin\\res_lcj\\bad-pkurasp-more.csv"
    bad_data_path3 = "S:\\VirtualMachineImage\\apache-jmeter-5.4.3\\bin\\res_lcj\\bad-openrasp.csv"
    bad_data_path4 = "S:\\VirtualMachineImage\\apache-jmeter-5.4.3\\bin\\res_lcj\\bad-bert.csv"
    good_data_path1 = "S:\\VirtualMachineImage\\apache-jmeter-5.4.3\\bin\\res_lcj\\good-norasp.csv"
    good_data_path2 = "S:\\VirtualMachineImage\\apache-jmeter-5.4.3\\bin\\res_lcj\\good-pkurasp.csv"
    good_data_path3 = "S:\\VirtualMachineImage\\apache-jmeter-5.4.3\\bin\\res_lcj\\good-openrasp.csv"
    good_data_path4 = "S:\\VirtualMachineImage\\apache-jmeter-5.4.3\\bin\\res_lcj\\good-bert-more.csv"
    save_path = "S:\\other\\hprasp"
    run(bad_data_path1, bad_data_path2, bad_data_path3, bad_data_path4, "恶意样本", 'zh', save_path)
    run(good_data_path1, good_data_path2, good_data_path3, good_data_path4, "良性样本", 'zh',save_path)
