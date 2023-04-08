from infer import *
import logging
from infer_tools.infer_tool import *
import gradio as gr
import webbrowser

logging.getLogger('numba').setLevel(logging.WARNING)
svc_model = None
project_name = "aqua"
wave_name = f"./temp.wav"
model_path = f'./aqua/clean_model_ckpt_steps_100000.ckpt'
config_path = f'./aqua/config.yaml'
spk_id = "aqua"

def infer(hubert_gpu, wav_fn, tran, accelerate, auto_key):

    model = Svc(project_name, config_path, hubert_gpu, model_path, onnx=False)

    if wav_fn is not None:
        audio_path = wav_fn
    else:
        return "请先上传wav格式的音频文件", None, None
    run_clip(raw_audio_path=audio_path, svc_model=model, key=tran, acc=accelerate, use_crepe=True,
                 spk_id=spk_id, auto_key=auto_key, project_name=project_name, out_path=wave_name)

    au_out = wave_name

    return "转换成功", au_out

app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("推理"):
            with gr.Blocks():
                with gr.Blocks():
                    with gr.Box():
                        gr.Markdown(value="""**上传音频**""")
                        with gr.Row():
                            upload_input = gr.Audio(source="upload", label="源音频", type="filepath", elem_id="audio_inputs")
                            out_audio = gr.Audio(label="输出音频")
                with gr.Blocks():
                    with gr.Box():
                        gr.Markdown(value="""**参数设置**""")
                        with gr.Row():
                            use_gpu = gr.Checkbox(label="启用GPU加速（需CUDA）", value=True)
                            auto = gr.Checkbox(label="启用自动变调", value=False)
                        with gr.Row():
                            acc_vaule = gr.Slider(1, 50, value=20, interactive=True, label="加速倍率")
                        with gr.Row():
                            pitch_vaule = gr.Slider(-96, 96, value=0, interactive=True, label="变调（半音）")
                        with gr.Row():
                            with gr.Column(scale=1):
                                infer_md = gr.Button("转换音频", variant="primary")
                with gr.Blocks():
                    with gr.Box():
                        gr.Markdown(value="""**输出日志**""")
                        infer_msg = gr.Textbox(label="日志")
            infer_md.click(infer, [use_gpu, upload_input, pitch_vaule, acc_vaule, auto], [infer_msg, out_audio])
        with gr.TabItem("协议"):
            gr.Markdown(value="""
                此为上传文件推理用GUI，需要变声功能请使用变声器\n
                原项目地址：https://github.com/openvpi/diff-svc\n
                代码修改：@ChrisPreston\n
                模型训练：@ChrisPreston\n
                音源：Aqua Ch. 湊あくあ https://www.youtube.com/@MinatoAqua カバー株式会社\n
                模型使用协议（重要）：\n
                1.请勿用于商业目的\n
                2.请勿用于会影响主播本人的行为（比如冒充本人发表争议言论）\n
                3.请勿用于血腥、暴力、性相关、政治相关内容\n
                4.不允许二次分发模型\n
                5.非个人使用场合请注明模型作者@ChrisPreston以及diff-svc原项目\n
                6.允许用于个人娱乐场景下的游戏语音、直播活动，不得用于低创内容，用于直播前请与本人联系\n
                联系方式：电邮：kameiliduo0825@gmail.com, b站：https://space.bilibili.com/18801308\n
                免责声明：由于使用本模型造成的法律纠纷本人概不负责
            """)
    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=False)
            
