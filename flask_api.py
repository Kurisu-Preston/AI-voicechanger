import io
import logging

import librosa
import soundfile
from flask import Flask, request, send_file
from flask_cors import CORS

from infer_tools.infer_tool import Svc
from utils.hparams import hparams

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    # http获得wav文件并转换
    input_wav_path = io.BytesIO(wave_file.read())
    # 模型推理
    _f0_tst, _f0_pred, _audio = svc_model.infer(input_wav_path, key=f_pitch_change, acc=accelerate, use_crepe=False)
    tar_audio = librosa.resample(_audio, hparams["audio_sample_rate"], daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio, daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':

    user_input = input("""
                    此为手动变调版变声器，具体参数请在vst插件内调整\n
                    原项目地址：https://github.com/openvpi/diff-svc\n
                    vst插件：https://github.com/zhaohui8969/VST_NetProcess-/tree/master\n
                    代码修改、模型训练：@ChrisPreston\n
                    音源：Aqua Ch. 湊あくあ https://www.youtube.com/@MinatoAqua カバー株式会社\n
                    模型使用协议（重要）：\n
                    1.请勿用于商业目的\n
                    2.请勿用于会影响主播本人的行为（比如冒充本人发表争议言论）\n
                    3.请勿用于血腥、暴力、性相关、政治相关内容\n
                    4.不允许二次分发模型与此程序\n
                    5.非个人使用场合请注明模型作者@ChrisPreston以及diff-svc原项目\n
                    6.允许用于个人娱乐场景下的游戏语音、直播活动，不得用于低创内容，用于直播前请与本人联系\n
                    联系方式：电邮：kameiliduo0825@gmail.com, b站：https://space.bilibili.com/18801308\n
                    是否同意上述内容？"""
                       "是y，否n（y/n）")
    if user_input != "y":
        print("正在退出程序")
        exit()
    # 工程文件夹名，训练时用的那个
    project_name = "aqua"
    model_path = f'./aqua/clean_model_ckpt_steps_100000.ckpt'
    config_path = f'./aqua/config.yaml'

    # 加速倍数
    accelerate = 50
    hubert_gpu = True

    svc_model = Svc(project_name, config_path, hubert_gpu, model_path)

    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
