import ChatTTS
import numpy as np
import wave
from playsound import playsound

chat = ChatTTS.Chat()
chat.load_models(compile=False) # 设置为True以获得更快速度

texts = ["\
         我是巫思远",]

# texts = ["\
#          我们公司的一个同事,负责运维的,我最喜欢他了,\
#          是一个带着眼镜很帅然后[uv_break]计算机技术非常厉害,很聪明的一个男生,我很多闺蜜私下都说想当他[laugh]女朋友[laugh]\
#          ,真的非常受欢迎",]
# texts = ["\
#          其实说真的，我觉得巫龙他[laugh]最好了[laugh],我最[uv_break]喜欢他了,\
#          是一个安静然后[uv_break]又很高,又很聪明的男生,我很多闺蜜私下都说想当他[laugh]女朋友[laugh]\
#          ,真的非常受欢迎",]

###################################
# Sample a speaker from Gaussian.

# spk_stat = torch.load('ChatTTS/asset/spk_stat.pt')
# rd = torch.randn(768)
# print("randn is ",rd)
# rand_spk = rd * spk_stat.chunk(2)[0] + spk_stat.chunk(2)[1]
#2233是女生
rand_spk = chat.sample_random_speaker(seed=2233)
params_infer_code = {
  'spk_emb': rand_spk, # add sampled speaker 
  'temperature': .3, # using custom temperature
  'top_P': 0.7, # top P decode
  'top_K': 20, # top K decode
}

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_6]'
} 

wavs = chat.infer(texts,use_decoder=True, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

# torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
audio_data = np.array(wavs[0], dtype=np.float32)
sample_rate = 24000
audio_data = (audio_data * 32767).astype(np.int16)
out_file = "./result.wav"
with wave.open(out_file, "w") as wf:
    wf.setnchannels(1)  # Mono channel
    wf.setsampwidth(2)  # 2 bytes per sample
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data.tobytes())
# 播放保存的.wav文件
playsound(out_file)
