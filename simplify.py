import os
import re
import shutil

import torch


def get_model_folder(path):
    model_lists = os.listdir(path)
    res_list = []
    filter_list = ["hubert", "xiaoma_pe", "hifigan", "checkpoints", ".yaml", ".zip"]
    for path in model_lists:
        if not any(word if word in path else False for word in filter_list):
            res_list.append(path)
    return res_list


def scan(path):
    model_str = ""
    path_lists = get_model_folder(path)
    for i in range(0, len(path_lists)):
        if re.search(u'[\u4e00-\u9fa5]', path_lists[i]):
            print(f'{path_lists[i]}:中文路径！此项跳过')
            continue
        model_str += f"{i}:{path_lists[i]}  "
        if (i + 1) % 5 == 0:
            print(f"{model_str}")
            model_str = ""
    if len(path_lists) % 5 != 0:
        print(model_str)
    return path_lists


def simplify_pth(model_name, proj_name, output_path):
    model_path = f'./checkpoints/{proj_name}'
    checkpoint_dict = torch.load(f'{model_path}/{model_name}')
    torch.save({'epoch': checkpoint_dict['epoch'],
                'state_dict': checkpoint_dict['state_dict'],
                'global_step': None,
                'checkpoint_callback_best': None,
                'optimizer_states': None,
                'lr_schedulers': None
                }, output_path)


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


if __name__ == '__main__':
    if os.path.exists("./checkpoints"):
        path_list = scan("./checkpoints")
    else:
        print("请检查checkpoints文件夹是否存在")
        exit()
    a = input("\r\n请输入序号并回车:")
    project_name = path_list[int(a)]
    path_list = scan(f"./checkpoints/{path_list[int(a)]}")
    b = input("\r\n请输入序号并回车:")
    pth_name = path_list[int(b)]

    print("\r\n选择:\r\n"
          "0.存储精简模型到对应模型目录（本地精简模型时推荐使用这个）\r\n"
          "1.存储精简模型和config.yaml到程序根目录（新建文件夹，九天毕昇上导出精简模型推荐使用这个）\r\n"
          "2.复制完整模型和config.yaml到程序根目录（新建文件夹，九天毕昇上导出完整模型推荐使用这个）\r\n"
          "输入其他退出")
    f = int(input("\r\n请输入序号并回车:"))
    if f == 0:
        print(f"已保存精简模型至对应模型目录")
        shutil.copyfile(f'./checkpoints/{project_name}/config.yaml', f"./{project_name}/config.yaml")
        output = f"./checkpoints/{project_name}/clean_{pth_name}"
        simplify_pth(pth_name, project_name, output)
    elif f == 1:
        print(f"已保存精简模型至: 根目录下新建文件夹/{project_name}")
        mkdir([f"./{project_name}"])
        shutil.copyfile(f'./checkpoints/{project_name}/config.yaml', f"./{project_name}/config.yaml")
        output = f"./{project_name}/clean_{pth_name}"
        simplify_pth(pth_name, project_name, output)
    elif f == 2:
        print(f"已保存完整模型至: 根目录下新建文件夹/{project_name}")
        mkdir([f"./{project_name}"])
        shutil.copyfile(f'./checkpoints/{project_name}/config.yaml', f"./{project_name}/config.yaml")
        shutil.copyfile(f'./checkpoints/{project_name}/{pth_name}', f"./{project_name}/{pth_name}")
    else:
        print("输入错误，程序退出")
        exit() 
