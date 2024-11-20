from tqdm import tqdm
import pandas as pd
from Bio import SeqIO
from prodmm.encoder.modeling_prolm import ProLMForMaskedLM
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer

def read_seq(fasta_file):
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)

def score_mutant(mutant, fasta, model, tokenizer, tag="cds"):
    seq = read_seq(fasta)
    df = pd.read_csv(mutant)
    
    tokenizer_res = tokenizer(
        [
            f"<{tag}>" + seq + f"</{tag}>",
        ],
        return_tensors="pt",
        padding=True,
    )
    input_ids = tokenizer_res["input_ids"].cuda()
    attention_mask = tokenizer_res["attention_mask"].cuda()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    logits = outputs.logits
    logits = logits[:, 1:-1, :].log_softmax(dim=-1)
    if tag == "cds":
        num_token = 3
    elif tag == "aas":
        num_token = 1
    elif tag == "ncds":
        num_token = 1
    preds = []
    for tm in tqdm(df["mutant"]):
        score = 0
        for m in tm.split(":"):
            wt, idx, mut = (
                m[:num_token],
                int(m[num_token:-num_token]) - 1,
                m[-num_token:],
            )
            wt_idx = tokenizer.get_vocab()[wt.lower() if tag == "cds" else wt]
            mut_idx = tokenizer.get_vocab()[mut.lower() if tag == "cds" else mut]
            score += logits[0, idx, mut_idx] - logits[0, idx, wt_idx]
        preds.append(score.item())
    df["prediction"] = preds
    return df

def main():
    from argparse import ArgumentParser
    from pathlib import Path
    parser = ArgumentParser()
    parser.add_argument(
        "--fasta",
        type=str,
        default="example_data/NCDS_andreasson_2020.fasta",
    )
    parser.add_argument(
        "--dms",
        type=str,
        default="example_data/NCDS_andreasson_2020.csv",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="example_data/NCDS_andreasson_2020_scored.csv",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="ncds",
    )
    args = parser.parse_args()
    print("Loading model...")
    tokenizer = UniTokenizer()
    model = ProLMForMaskedLM.from_pretrained("AI4Protein/prodmm_encoder")
    model = model.cuda()
    df = score_mutant(args.dms, args.fasta, model, tokenizer, args.tag)
    df.to_csv(args.save, index=False)
    
    if "score" in df.columns:
        from scipy.stats import spearmanr
        print(f"Spearman correlation", spearmanr(df["score"], df["prediction"]))

if __name__ == "__main__":
    main()
