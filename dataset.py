import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(
        self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len=128
    ):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [self.tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [self.tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [self.tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )
        # self.unk_token = torch.tensor(self.tokenizer_tgt.token_to_id("[UNK]"),dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]
        enc_input_tokens = self.tokenizer_src.encode(
            src_text
        ).ids  # convert toekns to ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = (
            self.seq_len - len(enc_input_tokens) - 2
        )  # -2 for sos and eos tokens
        dec_num_padding_tokens = (
            self.seq_len - len(dec_input_tokens) - 1
        )  # -1 for eos token

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise Exception("Sequence length is too small")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens),
            ]
        )

        decoder_input = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens),
            ]
        )
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.pad_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens),
            ]
        )

        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & casual_mask(decoder_input.shape[0]),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def casual_mask(seq_len):
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0
