# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn

# Base Model
from models.model import Model

# Encoders
from models.encoders import (
    ConformerEncoder,
    ConformerEncoderInterCTC
)

# Losses
from models.losses import (
    LossCTC, 
    LossInterCTC
)

# CTC Decode Beam Search
from torchaudio.models.decoder import ctc_decoder

class ModelCTC(Model):

    def __init__(self, encoder_params, tokenizer_params, training_params, decoding_params, name):
        super(ModelCTC, self).__init__(tokenizer_params, training_params, decoding_params, name)

        # Encoder
        if encoder_params["arch"] == "Conformer":
            self.encoder = ConformerEncoder(encoder_params)
        else:
            raise Exception("Unknown encoder architecture:", encoder_params["arch"])

        # FC Layer
        self.fc = nn.Linear(encoder_params["dim_model"][-1] if isinstance(encoder_params["dim_model"], list) else encoder_params["dim_model"], tokenizer_params["vocab_size"])

        # Criterion
        self.criterion = LossCTC()

        # Beam-Search Decoder Cache
        self._decoder_tokens = None
        self._beam_decoder_cache = {}

        # Compile
        self.compile(training_params)

    def forward(self, batch):

        # Unpack Batch
        x, _, x_len, _ = batch

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len, attentions = self.encoder(x, x_len)

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        return logits, logits_len, attentions

    def distribute_strategy(self, rank):
        super(ModelCTC, self).distribute_strategy(rank)

        self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[self.rank])
        self.fc = torch.nn.parallel.DistributedDataParallel(self.fc, device_ids=[self.rank])

    def load_encoder(self, path):

        # Load Encoder Params
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)
        if checkpoint["is_distributed"] and not self.is_distributed:
            self.encoder.load_state_dict({key.replace(".module.", ".").replace("encoder.", ""):value for key, value in checkpoint["model_state_dict"].items() if key[:len("encoder")] == "encoder"})
        else:
            self.encoder.load_state_dict({key.replace("encoder.", ""):value for key, value in checkpoint["model_state_dict"].items() if key[:len("encoder")] == "encoder"})

        # Print Encoder state
        if self.rank == 0:
            print("Model encoder loaded at step {} from {}".format(checkpoint["model_step"], path))

    def gready_search_decoding(self, x, x_len):

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len = self.encoder(x, x_len)[:2]

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        # Softmax -> Log > Argmax -> (B, T)
        preds = logits.log_softmax(dim=-1).argmax(dim=-1)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):

            # Blank
            blank = False

            # Pred List
            pred_list = []

            # Decoding Loop
            for t in range(logits_len[b]):

                # Blank Prediction
                if preds[b, t] == 0:
                    blank = True
                    continue

                # First Prediction
                if len(pred_list) == 0:
                    pred_list.append(preds[b, t].item())

                # New Prediction
                elif pred_list[-1] != preds[b, t] or blank:
                    pred_list.append(preds[b, t].item())
                
                # Update Blank
                blank = False

            # Append Sequence
            batch_pred_list.append(pred_list)

        # Decode Sequences
        return self.tokenizer.decode(batch_pred_list)

    def _get_decoder_tokens(self):

        if self._decoder_tokens is None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for beam-search decoding")
            self._decoder_tokens = [chr(idx + self.ngram_offset) for idx in range(self.tokenizer.vocab_size())]

        return self._decoder_tokens

    def _get_ctc_beam_decoder(self, beam_size):

        cache_key = (beam_size, self.ngram_path, self.ngram_alpha, self.ngram_beta)
        decoder = self._beam_decoder_cache.get(cache_key, None)

        if decoder is None:
            tokens = self._get_decoder_tokens()
            sil_token = tokens[1] if len(tokens) > 1 else tokens[0]
            decoder = ctc_decoder(
                lexicon=None,
                tokens=tokens,
                lm=self.ngram_path,
                nbest=1,
                beam_size=beam_size,
                beam_size_token=len(tokens),
                lm_weight=self.ngram_alpha,
                word_score=self.ngram_beta,
                blank_token=tokens[0],
                sil_token=sil_token
            )
            self._beam_decoder_cache[cache_key] = decoder

        return decoder

    def beam_search_decoding(self, x, x_len, beam_size=None):

        # Overwrite beam size
        if beam_size is None:
            beam_size = self.beam_size

        decoder = self._get_ctc_beam_decoder(beam_size)

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len = self.encoder(x, x_len)[:2]

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        # Apply Temperature
        logits = logits / self.tmp

        # Softmax -> Log
        logP = logits.log_softmax(dim=-1)

        # Beam Search Decoding (torchaudio expects CPU float tensors)
        emissions = logP.detach().float().cpu()
        lengths = logits_len.detach().cpu()
        beam_hypos = decoder(emissions, lengths)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b, hypos in enumerate(beam_hypos):
            if hypos:
                batch_pred_list.append(hypos[0].tokens.tolist())
            else:
                batch_pred_list.append([])

        # Decode Sequences
        return self.tokenizer.decode(batch_pred_list)

class InterCTC(ModelCTC):

    def __init__(self, encoder_params, tokenizer_params, training_params, decoding_params, name):
        super(ModelCTC, self).__init__(tokenizer_params, training_params, name)

        # Update Encoder Params
        encoder_params["vocab_size"] = tokenizer_params["vocab_size"]

        # Encoder
        if encoder_params["arch"] == "Conformer":
            self.encoder = ConformerEncoderInterCTC(encoder_params)

        # FC Layer
        self.fc = nn.Linear(encoder_params["dim_model"][-1] if isinstance(encoder_params["dim_model"], list) else encoder_params["dim_model"], tokenizer_params["vocab_size"])

        # Criterion
        self.criterion = LossInterCTC(training_params["interctc_lambda"])

        # Compile
        self.compile(training_params)

    def forward(self, batch):

        # Unpack Batch
        x, _, x_len, _ = batch

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len, attentions, interctc_logits = self.encoder(x, x_len)

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        return logits, logits_len, attentions, interctc_logits
