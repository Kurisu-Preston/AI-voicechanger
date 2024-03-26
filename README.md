# Diff-SVC

Singing Voice Conversion via diffusion model

## 本仓库为基于diff-svc fork重构版的AI阿夸变声器  
> 原仓库：[diff-svc](https://github.com/prophesier/diff-svc)  
> VST插件：[async http process VST plugin](https://github.com/zhaohui8969/VST_NetProcess-)

## 注意事项/Notes：

> 本项目是基于学术交流目的建立，并非为生产环境准备，不对由此项目模型产生的任何声音的版权问题负责。\
> 如将本仓库代码二次分发，或将由此项目产出的任何结果公开发表(包括但不限于视频网站投稿)，请注明原作者及代码来源(此仓库)。\
> 如果将此项目用于任何其他企划，请提前联系并告知本仓库作者,十分感谢。\
> This project is established for academic exchange purposes and is not intended for production environments. We are not
> responsible for any copyright issues arising from the sound produced by this project's model. \
> If you redistribute the code in this repository or publicly publish any results produced by this project (including
> but
> not limited to video website submissions), please indicate the original author and source code (this repository). \
> If you use this project for any other plans, please contact and inform the author of this repository in advance. Thank
> you very much.

## 使用方法/Instructions：
```
git clone https://github.com/Kurisu-Preston/AI-aqua-vc
```
将[4.0整合包](https://www.bilibili.com/video/BV16g4y1376m/?spm_id_from=333.999.0.0)中的minianaconda和Dependencies拖入直接双击运行bat.  
模型下载后放在aqua文件夹，checkpoints直接在根目录下解压.  
自动变调功能为将调整输入源音高以适应数据集音域，可以使得声音更像，但是不是完全自动，如效果不好须在vst插件内调整pitch.  
不提供训练代码和相关辅助.  
Drag and drop the mini-anaconda and Dependencies from the 4.0 integration package into the bat file for direct execution. After downloading, place the model in the aqua folder, and unzip the checkpoints directly in the root directory. The automatic pitch shifting feature adjusts the pitch of the input source to fit the dataset's range, which can make the sound more similar, but it is not completely automatic; if the effect is not good, adjust the pitch inside the VST plugin. Training code and related support are not provided.

## Acknowledgements

项目基于[diffsinger](https://github.com/MoonInTheRiver/DiffSinger)、[diffsinger(openvpi维护版)](https://github.com/openvpi/DiffSinger)、[soft-vc](https://github.com/bshall/soft-vc)
开发.\
同时也十分感谢openvpi成员在开发训练过程中给予的帮助。 
This project is based
on [diffsinger](https://github.com/MoonInTheRiver/DiffSinger), [diffsinger (openvpi maintenance version)](https://github.com/openvpi/DiffSinger),
and [soft-vc](https://github.com/bshall/soft-vc). We would also like to thank the openvpi members for their help during
the development and training process.  
> 注意：此项目与同名论文[DiffSVC](https://arxiv.org/abs/2105.13871)无任何联系，请勿混淆！\
> Note: This project has no connection with the paper of the same name [DiffSVC](https://arxiv.org/abs/2105.13871),
> please
> do not confuse them!
