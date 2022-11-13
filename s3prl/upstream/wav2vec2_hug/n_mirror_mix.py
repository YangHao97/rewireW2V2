from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from s3prl.upstream.interfaces import Featurizer
from s3prl.downstream.asr.expert import DownstreamExpert
from s3prl.downstream.asr.dictionary import Dictionary
#from s3prl.downstream.speech_commands.expert import DownstreamExpert
import yaml
from torch.utils.data import DistributedSampler
from tqdm import tqdm
import random
import numpy as np
import copy
import torchaudio
from pytorch_metric_learning import losses
import os

SAMPLE_RATE = 16000


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        super().__init__()

        self.processor = Wav2Vec2Processor.from_pretrained(ckpt)
        self.model = Wav2Vec2Model.from_pretrained(ckpt)
        #"/home/haoy/da33_scratch/haoy/s3prl/s3prl/upstream/wav2vec2_hug/1e6-large-sumavg-1ep"

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs: List[Tensor]):
        #print("begin!!!!!")
        device = wavs[0].device
        processor_outputs = self.processor(
            [wav.cpu().numpy() for wav in wavs],
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
            padding="longest",
        )
        attention_mask = processor_outputs.get("attention_mask", None)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device)
        model_outputs = self.model(
            processor_outputs.input_values.to(device),
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return {
            "last_hidden_state": model_outputs.last_hidden_state,
            "hidden_states": model_outputs.hidden_states,
        }

class MirrorWav2vec():
    def __init__(self):
        self.upstream = UpstreamExpert('facebook/wav2vec2-large-lv60').to('cuda')
        self.featurizer = self._get_featurizer()
        self.downstream = self._get_downstream()
        self.learning_rate = 1e-6
        self.weight_decay = 0.01
        self.infoNCE_tau = 0.04
        self.loss = losses.NTXentLoss(temperature=self.infoNCE_tau)
        self.dictionary = Dictionary.load("/home/haoy/da33_scratch/haoy/s3prl/s3prl/downstream/asr/char.dict")
        self.optimizer = torch.optim.AdamW([{'params': self.upstream.model.parameters()},],
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.model_finetuning()

    def _get_featurizer(self):
        model = Featurizer(
            upstream = self.upstream,
            feature_selection = "hidden_states",
            layer_selection = None,
            upstream_device = "cuda",
        ).to('cuda')

        return model

    def _get_downstream(self):
        with open("/home/haoy/da33_scratch/haoy/s3prl/s3prl/downstream/asr/config.yaml", 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        # with open("/home/haoy/da33_scratch/haoy/s3prl/s3prl/downstream/speech_commands/config.yaml", 'r') as file:
        #     self.config = yaml.load(file, Loader=yaml.FullLoader)
        model = DownstreamExpert(
            upstream_dim = self.featurizer.output_dim,
            upstream_rate = self.featurizer.downsample_rate,
            **self.config,
            expdir="./new"
        ).to("cuda")

        return model

    def model_finetuning(self):
        self.upstream.model.train()
        loss_list = []
        ind = []
        dataloader = self.downstream.get_dataloader("train")

        for ep in [0,1,2]:
            print("epoch:" + str(ep))
            sent_pair1 = []
            sent_pair2 = []
            for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc='train')):
                wav1 = []
                temp_wav2 = []
                wav2 = []
                wavs = [torch.FloatTensor(wav).to("cuda") for wav in wavs]
                # n_wav positive
                trans = self.dictionary.string(others[0][0]).replace(" ","").replace("|", " ").lower()
                fp = open("/home/haoy/da33_scratch/haoy/s3prl/s3prl/upstream/wav2vec2_hug/temp.txt", "w")
                fp.write(trans)
                fp.close()
                os.system("text2wave -o " + "/home/haoy/da33_scratch/haoy/s3prl/s3prl/upstream/wav2vec2_hug/temp.wav" + " temp.txt")
                n_wav, _ = torchaudio.load("/home/haoy/da33_scratch/haoy/s3prl/s3prl/upstream/wav2vec2_hug/temp.wav")
                n_wav = n_wav.view(-1).to('cuda')
                temp_wav2.append(n_wav)
                os.system("rm -rf temp.txt")
                os.system("rm -rf temp.wav")
                ran = [0, 1]
                random.shuffle(ran)
                d_a = ran[0]

                if d_a == 0:

                    if wavs[0].size(0) > 180000:
                        continue
                    if wavs[0].size(0) > 90000: # 90000
                        if batch_id % 2 == 0:
                            wav1.append(wavs[0][0: int(wavs[0].size(0)/2) - 1])
                            wav2.append(temp_wav2[0][0: int(temp_wav2[0].size(0)/2) - 1])
                        else:
                            wav1.append(wavs[0][int(wavs[0].size(0)/2) - 1: -1]) 
                            wav2.append(temp_wav2[0][int(temp_wav2[0].size(0)/2) - 1: -1])
                    else:
                        wav1.append(wavs[0])
                        wav2.append(temp_wav2[0])
                else:
                    if wavs[0].size(0) > 180000:
                        continue
                    if wavs[0].size(0) > 90000: # 90000
                        if batch_id % 2 == 0:
                            wav1.append(wavs[0][0: int(wavs[0].size(0)/2) - 1])
                        else:
                            wav1.append(wavs[0][int(wavs[0].size(0)/2) - 1: -1])
                        wav2 = copy.deepcopy(wav1)
                    else:
                        wav1.append(wavs[0])
                        wav2 = copy.deepcopy(wav1)
                    for wav in wav2:
                        num = wav.size(0)
                        mask_len = int(num/5)
                        ran_start = random.sample(range(0, int(num/5) * 4 - 1), 1)[0]
                        mask_token = torch.zeros([mask_len], dtype=torch.float)
                        wav[ran_start:ran_start+mask_len] = mask_token

                if len(sent_pair1) != 3:
                    feature1 = self.featurizer(wav1, self.upstream(wav1))[0].mean(0)
                    feature2 = self.featurizer(wav2, self.upstream(wav2))[0].mean(0)
                    sent_pair1.append(feature1)
                    sent_pair2.append(feature2)
                else:
                    feature1 = self.featurizer(wav1, self.upstream(wav1))[0].mean(0)
                    feature2 = self.featurizer(wav2, self.upstream(wav2))[0].mean(0)
                    sent_pair1.append(feature1)
                    sent_pair2.append(feature2)
                    pair1 = torch.stack(sent_pair1)
                    pair2 = torch.stack(sent_pair2)
                    sent_pair1 = []
                    sent_pair2 = []

                    query_embed = torch.cat([pair1, pair2], dim=0)
                    labels = torch.arange(pair1.size(0))
                    labels = torch.cat([labels, labels], dim=0)
                    loss = self.loss(query_embed, labels)
                    loss_list.append(loss.item())
                    # control loss
                    ind.append(batch_id + (ep*28538))
                    print(loss)
                    loss.backward()
                    self.optimizer.step()

        import pandas as pd
        from matplotlib import pyplot as plt
        di = pd.Index(ind)
        pd.DataFrame({'data1': loss_list}, index=di).plot.line()
        plt.savefig("n+mirror-mix-3ep.png")
        self.upstream.model.save_pretrained("/home/haoy/da33_scratch/haoy/s3prl/s3prl/upstream/wav2vec2_hug/n+mirror-mix-3ep")


if __name__ == '__main__':
    #14151
    torch.manual_seed(141)
    random.seed(141)
    np.random.seed(141)
    model = MirrorWav2vec()