# ProDMM
Source code of ProDMM. The paper is titled with "Unveiling Protein-DNA Interdependency: Harnessing Unified Multimodal Sequence Modeling, Understanding and Generation".

## Install Requirements
- PyTorch 2.5+
- transformers
- pandas
- numpy

## ProDMM-Encoder

### 1. Generate embeddings for seqeuences.
```python
from prodmm.encoder.modeling_prolm import ProLMForMaskedLM
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer
import torch

tokenizer = UniTokenizer()
model = ProLMForMaskedLM.from_pretrained("AI4Protein/prodmm_encoder")
model = model.cuda()

with torch.no_grad():
    sequences = [
        "<cds>ATGCAAAAGCACGTGGACACCGCCGTT</cds>",
        "<aas>MQKHVDTAV</aas>",
        "<ncds>TGCAATGCGCCGTT</ncds>",
        "<ncds>TGCAATGCGCCGTT</ncds><cds>ATGCAAAAGCACGTGGACACCGCCGTT</cds>",
        "<ncds>TGCAATGCGCCGTT</ncds><aas>MQKHVDTAV</aas>",
    ]
    tokenized = tokenizer(sequences, special_add="encoder")
    input_ids = tokenized["input_ids"].cuda()
    attention_mask = tokenized["attention_mask"].cuda()
    outputs = model(input_ids, attention_mask=attention_mask)
    hidden_states = outputs.hidden_states
    print(hidden_states)
```

The results:
```
tensor([[[ 0.0512, -0.0124,  0.1225,  ...,  0.0893, -0.1459, -0.0952],
         [ 0.3179,  0.0941,  0.4567,  ...,  0.1954, -0.1968, -0.2199],
         [-0.2170,  0.1231,  0.5880,  ...,  0.2153, -0.2682,  0.0507],
         ...,
         [-0.0045,  0.1775,  0.2213,  ...,  0.1863, -0.1646, -0.0833],
         [-0.0074,  0.1981,  0.1898,  ...,  0.0277, -0.0703, -0.0704],
         [-0.2140,  0.1878,  0.1525,  ...,  0.1553, -0.1101, -0.0189]],

        [[ 0.1440, -0.0677,  0.1047,  ..., -0.0579, -0.1437, -0.0665],
         [ 0.4634, -0.0403,  0.2951,  ..., -0.0953, -0.2158, -0.1262],
         [ 0.2994,  0.2982,  0.4675,  ..., -0.1016, -0.0689,  0.0978],
         ...,
         [ 0.0542,  0.1818,  0.3228,  ..., -0.1461, -0.1721, -0.0387],
         [ 0.0459,  0.1774,  0.2912,  ..., -0.3076, -0.0795, -0.0838],
         [-0.1006,  0.1369,  0.2288,  ..., -0.2153, -0.0836,  0.0046]],

        [[ 0.3526,  0.0855,  0.1176,  ..., -0.1944, -0.2888,  0.1277],
         [ 0.5955,  0.2885,  0.2375,  ..., -0.5753, -0.0208, -0.1957],
         [-0.0275,  0.1796, -0.0039,  ..., -0.2516, -0.2631,  0.1997],
         ...,
         [ 0.3683,  0.1930,  0.2510,  ..., -0.2808,  0.0963,  0.2505],
         [ 0.5449,  0.3362,  0.3273,  ..., -0.2657, -0.2670,  0.2245],
         [ 0.4916,  0.1002,  0.2347,  ..., -0.2148, -0.3014,  0.1625]],

        [[ 0.2591,  0.0240,  0.0970,  ..., -0.0801, -0.2069,  0.0305],
         [ 0.3947,  0.0695,  0.4779,  ..., -0.5862, -0.0705,  0.1387],
         [-0.0065, -0.0060, -0.0286,  ..., -0.1679,  0.0506,  0.2206],
         ...,
         [-0.1143, -0.1049,  0.2203,  ...,  0.0684, -0.1379, -0.0179],
         [ 0.1439, -0.0139,  0.3406,  ...,  0.0214, -0.0404, -0.1503],
         [ 0.2389,  0.0259,  0.1779,  ..., -0.1946, -0.0903, -0.0540]],

        [[ 0.2101, -0.0559,  0.0991,  ..., -0.1005, -0.1898, -0.0244],
         [ 0.7040,  0.1038,  0.3970,  ..., -0.3299, -0.4025, -0.0864],
         [ 0.0475,  0.1066,  0.3940,  ..., -0.2950, -0.1282,  0.0319],
         ...,
         [-0.1670, -0.3057,  0.4274,  ..., -0.1586, -0.2575,  0.0854],
         [ 0.0384,  0.1742,  0.4122,  ..., -0.2483, -0.1520, -0.1061],
         [ 0.2461, -0.0406,  0.1892,  ..., -0.1516, -0.0812, -0.1018]]],
       device='cuda:0')
```

### 2. Perplexity-based Sequence Scoring
```python
from prodmm.encoder.modeling_prolm import ProLMForMaskedLM
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer
import torch
import pandas as pd

@torch.no_grad()
def score_sequence(model, tokenizer, sequence):
    tokenized_dict = tokenizer([sequence, ], return_tensors="pt")
    outputs = model(
        input_ids=tokenized_dict["input_ids"].cuda(),
        attention_mask=tokenized_dict["attention_mask"].cuda(),
        labels=tokenized_dict["input_ids"].cuda(),
    )
    neg_ppl = -outputs.loss.exp().item()
    return neg_ppl

tokenizer = UniTokenizer()
model = ProLMForMaskedLM.from_pretrained("AI4Protein/prodmm_encoder")
model = model.cuda()

sequences = [
    "<cds>ATGGAAGACTTTGTGCGACAATGCTTCAATCCAATGATCGTCGAGCTTGCGGAAAAGGCAATGAAAGAATATGGGGAAGATCCGAAAATCGAAACTAACAAGTTTGCTGCAATATGCACACATTTGGAAGTTTGTTTCATGTATTCGGATTTCGGCTCTGGTGACCCGAATGCACTATTGAAGCACCGATTTGAGATAATTGAAGGAAGAGACCGAATCATGGCCTGGACAGTGGTGAACAGTATATGTAACACAACAGGGGTAGAGAAGCCTAAATTTCTTCCTGATTTGTATGATTACAAAGAGAACCGGTTCATTGAAATTGGAGTAACACGGAGGGAAGTCCACATATATTACCTAGAGAAAGCCAACAAAATAAAATCTGAGAAGACACACATTCACATCTTTTCATTCACTGGAGAGGAGATGGCCACCAAAGCGGACTACACCCTTGACGAAGAGAGCAGGGCAAGAATCAAAACTAGGCTTTTCACTATAAGACAAGAAATGGCCAGTAGGAGTCTATGGGATTCCTTTCGTCAGTCCGAAAGAGGCGAATAA</cds>",
    "<cds>ATGGAAGACTTCGTGCGCCAATGCTTCAACCCGATGATCGTGGAGCTGGCCGAGAAAGCGATGAAAGAATACGGCGAGGACCCGAAGATCGAGACCAACAAATTCGCGGCCATCTGCACCCATCTCGAAGTGTGCTTCATGTACAGCGACTTCGGTAGCGGCGATCCGAATGCGCTGCTCAAGCACCGTTTTGAGATCATCGAGGGTCGCGATCGCATCATGGCGTGGACCGTGGTGAACAGCATCTGCAATACGACGGGCGTGGAAAAGCCGAAATTTCTGCCGGATCTGTACGACTACAAGGAGAACCGCTTCATCGAAATCGGCGTGACCCGCCGCGAGGTGCACATCTACTATCTGGAAAAGGCCAATAAAATCAAAAGCGAGAAAACGCACATCCATATCTTCAGCTTCACCGGCGAAGAAATGGCCACCAAAGCGGATTACACGCTGGATGAGGAAAGCCGCGCGCGTATCAAAACGCGTCTGTTCACCATCCGCCAAGAAATGGCGAGTCGTAGTCTGTGGGACAGTTTCCGCCAGAGCGAACGCGGCGAATAA</cds>",
    "<cds>ATGGAAGATTTTGTGCGCCAGTGTTTTAATCCGATGATTGTTGAACTGGCAGAGAAGGCCATGAAAGAATATGGTGAAGATCCGAAAATCGAAACCAACAAATTTGCAGCCATTTGCACCCATCTGGAAGTGTGTTTTATGTATAGCGATTTTGGTAGCGGTGATCCGAATGCACTGCTGAAACATCGTTTTGAAATTATCGAAGGTCGCGATCGTATTATGGCATGGACCGTTGTTAATAGCATTTGTAATACCACCGGTGTGGAAAAACCGAAATTTCTGCCGGATCTGTATGACTATAAAGAGAACCGCTTTATTGAAATTGGTGTGACCCGTCGTGAAGTGCATATTTACTATCTGGAAAAAGCCAACAAGATCAAAAGCGAGAAAACCCACATTCACATCTTTAGCTTTACCGGTGAAGAAATGGCAACCAAAGCAGATTATACCCTGGATGAAGAAAGCCGTGCACGTATTAAAACCCGTCTGTTTACCATTCGTCAAGAGATGGCAAGCCGTAGCCTGTGGGATAGCTTTCGTCAGAGCGAACGTGGTGAATAA</cds>",
    "<cds>ATGGAAGATTTTGTCCGTCAGTGCTTTAACCCGATGATTGTCGAACTGGCGGAAAAAGCGATGAAAGAATATGGCGAAGATCCGAAAATTGAAACCAACAAATTTGCGGCGATTTGTACCCATCTGGAAGTCTGCTTTATGTATAGCGATTTTGGTTCCGGCGATCCGAACGCGCTGCTGAAACATCGTTTTGAAATTATTGAAGGTCGCGATCGCATTATGGCGTGGACCGTGGTCAACAGCATTTGTAACACCACCGGCGTGGAAAAACCGAAATTCCTGCCGGATCTGTATGATTATAAAGAAAACCGTTTTATTGAAATTGGCGTCACCCGCCGTGAAGTGCATATTTATTATCTGGAAAAAGCCAACAAAATCAAAAGCGAAAAAACCCATATTCATATTTTCAGTTTTACCGGTGAAGAAATGGCGACCAAAGCGGATTATACCCTGGATGAAGAAAGCCGTGCGCGTATTAAAACCCGTCTGTTTACCATTCGTCAGGAAATGGCGAGCCGTTCGCTGTGGGATAGCTTCCGTCAGAGCGAACGCGGTGAATAA</cds>",
    "<cds>ATGGAAGATTTTGTTCGTCAGTGCTTTAACCCGATGATTGTCGAACTGGCGGAAAAAGCGATGAAAGAATATGGTGAAGATCCGAAAATTGAAACCAACAAATTTGCGGCGATTTGCACCCATCTGGAAGTGTGCTTTATGTATTCCGATTTTGGTAGCGGCGATCCGAACGCGCTGCTGAAACATCGCTTTGAAATTATTGAAGGCCGTGATCGTATTATGGCGTGGACCGTGGTTAACAGCATTTGCAACACCACCGGCGTGGAAAAACCGAAATTTCTGCCGGATCTGTATGATTACAAAGAAAACCGTTTTATTGAAATTGGCGTTACCCGCCGTGAAGTGCATATTTATTATCTGGAAAAAGCGAACAAAATCAAAAGCGAAAAAACCCATATCCATATTTTTTCTTTTACCGGTGAAGAAATGGCGACCAAAGCGGATTATACCCTGGATGAAGAAAGCCGTGCGCGTATTAAAACCCGCCTGTTTACCATTCGCCAGGAAATGGCGAGCCGCAGCCTGTGGGATAGCTTTCGCCAGAGCGAACGCGGTGAATAA</cds>",
]

scores = []
for idx, seq in enumerate(sequences):
    score = score_sequence(model, tokenizer, seq)
    scores.append(score)
    
df = pd.DataFrame({"sequence": sequences, "score": scores})
df['rank'] = df['score'].rank(ascending=False)
print(df)
```

The results:
```
                                            sequence     score  rank
0  <cds>ATGGAAGACTTTGTGCGACAATGCTTCAATCCAATGATCGT... -1.979317   5.0
1  <cds>ATGGAAGACTTCGTGCGCCAATGCTTCAACCCGATGATCGT... -1.771894   4.0
2  <cds>ATGGAAGATTTTGTGCGCCAGTGTTTTAATCCGATGATTGT... -1.723165   3.0
3  <cds>ATGGAAGATTTTGTCCGTCAGTGCTTTAACCCGATGATTGT... -1.605115   1.0
4  <cds>ATGGAAGATTTTGTTCGTCAGTGCTTTAACCCGATGATTGT... -1.655604   2.0
```

#### Tips. Scoring metabolic pathways
```python
from prodmm.encoder.modeling_prolm import ProLMForMaskedLM
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer
import torch
import pandas as pd

@torch.no_grad()
def score_sequence(model, tokenizer, sequence):
    tokenized_dict = tokenizer([sequence, ], return_tensors="pt")
    outputs = model(
        input_ids=tokenized_dict["input_ids"].cuda(),
        attention_mask=tokenized_dict["attention_mask"].cuda(),
        labels=tokenized_dict["input_ids"].cuda(),
    )
    neg_ppl = -outputs.loss.exp().item()
    return neg_ppl

tokenizer = UniTokenizer()
model = ProLMForMaskedLM.from_pretrained("AI4Protein/prodmm_encoder_antismash_tuned")
model = model.cuda()

sequences = [
    "<ncds>GAACTGGTTAAGGTGATGGACAACGGGTGGTAATATGCTATACTAAAGGC</ncds><cds>ATGGCGCCGCGCCCGACCTCGCAATCGCAAGCGCGCACCTGCCCGACGACGCAGGTCACGCAGGTCGATATCGTCGAAAAGATGCTGGCCGCGCCGACCGACAGCACGCTTGAACTCGATGGCTATTCGCTCAATCTTGGCGATGTCGTTAGCGCCGCGCGCAAGGGCCGGCCGGTTCGCGTCAAGGATAGCGACGAAATCCGCAGCAAGATCGACAAGTCGGTCGAGTTTCTGCGCAGCCAGCTTTCGATGAGCGTCTATGGCGTCACGACCGGCTTTGGCGGCAGCGCCGACACGCGCACCGAAGATGCGATCTCGCTGCAAAAGGCGCTGCTCGAGCATCAGCTTTGCGGCGTTCTGCCCTCCAGCTTCGACAGCTTCCGCCTTGGCCGCGGCCTTGAAAACAGCCTGCCGCTCGAAGTCGTGCGCGGCGCGATGACGATCCGCGTCAACTCGCTGACGCGCGGCCATAGCGCCGTTCGGCTGGTCGTGCTCGAAGCGCTGACCAACTTCCTCAATCAGGGCATCACGCCGATCGTTCCGCTGCGCGGCACGATCTCGGCCAGCGGCGATCTTTCGCCGCTCAGCTATATCGCCGCCGCGATCTCCGGCCATCCCGACAGCAAGGTTCATGTCGTTCACGAAGGCAAGGAAAAGATCCTCTATGCCCGCGAAGCGATGGCGCTGTTCAATCTCGAGCCGGTCGTGCTTGGCCCCAAGGAAGGGCTTGGCCTGGTCAACGGCACGGCGGTTTCGGCCTCGATGGCGACGCTGGCGCTGCACGATGCGCACATGCTGTCGCTGCTGTCGCAGTCGCTGACCGCGATGACGGTCGAGGCGATGGTCGGCCATGCCGGCAGCTTCCATCCCTTCCTGCACGATGTGACGCGCCCGCATCCGACGCAGATCGAGGTTGCCGGCAATATCCGCAAGCTGCTCGAAGGCAGCCGCTTTGCCGTTCATCACGAAGAAGAGGTCAAGGTCAAGGACGACGAAGGCATTCTGCGGCAGGATCGCTATCCGCTGCGCACCTCGCCGCAATGGCTTGGGCCGCTGGTCAGCGATCTGATCCATGCCCATGCCGTGCTGACGATCGAGGCCGGCCAATCGACGACCGACAATCCGCTGATCGATGTCGAGAACAAGACCAGCCATCATGGCGGCAATTTCCAGGCCGCCGCCGTCGCCAACACGATGGAAAAGACGCGGCTTGGCCTGGCGCAGATCGGCAAGCTCAACTTCACCCAGCTGACCGAAATGCTGAATGCCGGCATGAACCGCGGCCTGCCGTCCTGCCTGGCGGCCGAAGATCCGTCGCTGTCCTATCATTGCAAGGGGCTCGATATCGCCGCCGCGGCCTACACCAGCGAACTTGGCCATCTGGCCAATCCGGTCACGACGCATGTTCAGCCGGCCGAAATGGCCAATCAGGCGGTCAACTCGCTGGCGCTGATCTCGGCGCGCCGCACGACCGAAAGCAACGATGTTCTGTCGCTGCTGCTGGCGACGCATCTCTATTGCGTGCTGCAGGCGATCGATCTGCGCGCGATCGAGTTCGAGTTCAAGAAGCAGTTCGGCCCGGCGATCGTCTCGCTGATCGATCAGCATTTCGGCTCGGCGATGACCGGCAGCAACCTGCGCGACGAACTGGTCGAAAAGGTCAACAAGACGCTCGCCAAGCGGCTGGAGCAGACCAACAGCTACGATCTCGTGCCGCGCTGGCATGATGCCTTCAGCTTTGCCGCCGGCACGGTGGTCGAAGTGCTGTCCTCGACCTCGCTGTCGCTTGCCGCGGTCAATGCCTGGAAGGTCGCCGCCGCCGAAAGCGCGATCTCGCTGACGCGGCAGGTGCGCGAAACCTTCTGGAGCGCGGCCTCGACCAGCTCGCCGGCGCTGAGCTATCTCTCGCCGCGCACGCAGATCCTCTATGCCTTCGTGCGCGAAGAACTGGGCGTGAAGGCGCGGCGCGGCGATGTGTTCCTCGGCAAGCAGGAAGTCACGATCGGCAGCAACGTCTCGAAGATCTACGAAGCGATCAAGAGCGGCCGCATCAACAACGTGCTGCTGAAGATGCTGGCC</cds><ncds>TTATGATAAGTTCACTGTTGGCTAAAGGGAGTGTAAGTCATAATGAACTT</ncds><cds>ATGGGTGATTGTGTTGCTCCAAAAGAAGATTTAATTTTTAGATCAAAATTACCTGATATTTATATTCCAAAACATTTGCCTCTTCATACTTATTGTTTTGAAAATATTTCAAAAGTTGGCGATAAATCTTGTTTGATTAATGGTGCAACAGGTGAAACTTTTACCTATTCTCAAGTTGAATTGTTATCGCGCAAAGTTGCATCTGGCTTAAATAAACTTGGCATTCAGCAAGGCGATACGATCATGCTGTTGTTGCCCAATAGCCCAGAATATTTTTTTGCTTTTTTAGGCGCATCTTATCGCGGTGCAATTTCAACAATGGCAAATCCATTTTTTACCTCTGCAGAAGTTATTAAGCAATTAAAAGCATCTCAAGCAAAACTGATTATTACGCAAGCTTGTTATGTTGATAAAGTAAAAGATTATGCTGCTGAAAAAAATATTCAGATTATTTGTATTGATGATGCGCCGCAAGATTGTCTGCATTTTTCAAAATTGATGGAAGCTGATGAAAGCGAAATGCCAGAAGTTGTTATCAATTCTGATGATGTTGTTGCTTTGCCTTATTCATCTGGCACAACAGGTTTGCCAAAAGGCGTGATGCTGACCCATAAAGGTTTGGTGACATCTGTTGCGCAGCAAGTTGATGGCGATAATCCAAATCTTTATATGCATTCTGAAGATGTGATGATTTGCATTTTGCCGCTTTTTCATATTTATTCTTTAAATGCAGTTTTATGTTGTGGTTTGCGTGCAGGTGTTACCATTTTGATCATGCAAAAATTTGATATTGTTCCATTTCTTGAACTTATTCAAAAATATAAAGTTACGATTGGTCCATTTGTTCCGCCAATTGTTTTGGCAATTGCCAAATCTCCTGTTGTTGATAAATATGATTTATCTTCTGTTAGAACGGTTATGTCTGGTGCAGCACCTTTGGGCAAAGAACTTGAAGATGCTGTTCGCGCAAAATTTCCAAATGCAAAACTTGGTCAAGGCTATGGCATGACCGAAGCTGGTCCTGTTTTGGCAATGTGTTTGGCTTTTGCCAAAGAACCTTATGAAATTAAATCTGGTGCTTGCGGCACGGTTGTTCGCAATGCTGAAATGAAAATTGTTGATCCTGAAACCAATGCCTCTTTGCCGCGCAATCAACGCGGTGAAATTTGTATTCGCGGCGATCAGATTATGAAGGGCTATTTGAATGATCCTGAAAGCACCCGCACAACAATTGATGAAGAAGGTTGGCTTCATACTGGCGATATTGGTTTTATTGATGATGATGATGAACTTTTTATTGTTGATCGCTTGAAAGAAATTATCAAATATAAAGGTTTTCAAGTTGCTCCTGCTGAACTTGAAGCTTTGTTGTTAACCCATCCAACAATTTCTGATGCTGCTGTTGTTCCGATGATTGATGAAAAAGCTGGTGAAGTTCCTGTTGCTTTTGTTGTTAGAACCAATGGCTTTACAACAACTGAAGAAGAAATTAAGCAATTTGTTTCAAAGCAAGTTGTTTTTTATAAACGTATTTTCCGCGTTTTTTTTGTTGATGCAATTCCAAAATCACCATCTGGCAAAATTTTGCGCAAAGATTTGCGCGCAAGAATTGCTTCTGGCGATTTGCCAAAA</cds><ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><cds>ATGGTTACAGTTGAAGAATATCGCAAAGCACAACGCGCAGAAGGGCCTGCAACAGTTATGGCAATTGGCACAGCAACGCCAACCAATTGCGTCGATCAATCAACCTATCCTGATTATTATTTTCGCATCACCAATAGCGAACATAAAACCGATTTAAAAGAAAAATTTAAGCGGATGTGTGAAAAATCGATGATCAAAAAACGCTATATGCATCTGACCGAAGAAATTTTGAAAGAAAATCCATCAATGTGTGAATATATGGCGCCATCTCTTGATGCGCGGCAAGATATTGTTGTTGTTGAAGTTCCAAAACTTGGCAAAGAAGCTGCGCAAAAAGCGATCAAAGAATGGGGGCAGCCAAAATCAAAAATTACGCATTTGGTTTTTTGCACCACATCTGGCGTTGATATGCCGGGCTGTGATTATCAACTGACCAAATTGCTTGGCTTACGCCCATCTGTAAAACGGTTGATGATGTATCAGCAAGGCTGTTTTGCTGGCGGCACAGTTTTGCGTTTGGCAAAAGATTTGGCAGAAAATAATAAAGGCGCGCGCGTTTTGGTTGTTTGTTCTGAAATTACTGCAGTTACTTTTCGCGGCCCAAATGATACGCATCTTGATAGTTTGGTTGGTCAAGCTTTATTTGGTGATGGTGCAGGCGCAATTATTATTGGTTCTGATCCAATTCCCGGTGTTGAACGCCCTTTATTTGAATTGGTTTCTGCAGCGCAAACATTATTGCCTGATAGCCATGGCGCAATTGATGGGCATTTGCGCGAAGTTGGCCTGACCTTTCATTTGTTAAAAGATGTTCCTGGCCTGATTTCAAAAAATATTGAAAAATCTTTGGAAGAAGCTTTTCGCCCACTTTCAATTACCGATTGGAATAGCTTATTTTGGATTGCTCATCCCGGTGGCCCTGCAATTTTGGATCAAGTTGAAATCAAACTTGGCTTAAAGCCAGAAAAATTAAAAGCAACACGCAATGTTCTTTCCAATTATGGCAATATGTCTTCTGCTTGCGTTCTGTTTATTTTGGATGAAATGCGCAAAGCATCTGCAAAAGAAGGTCTTGGCACAACTGGCGAAGGTTTGGAATGGGGCGTTTTATTTGGCTTTGGCCCTGGCTTAACGGTTGAAACAGTTGTTCTGCATTCTGTTGCAACA</cds><ncds>CAGAGAAAACGTGCTCCAGCCCAAAAAAACGTTACAATTGCCGCCCATTACTGAAAACCACAGTAAAGCGAGGTTTT</ncds><cds>ATGGCAGCAAGCATCACCGCAATCACCGTTGAAAATCTGGAATATCCGGCCGTCGTCACCTCGCCGGTGACCGGCAAAAGCTATTTTCTTGGCGGCGCGGGCGAGCGCGGCCTGACGATCGAAGGCAATTTCATCAAATTCACCGCGATCGGCGTCTATCTGGAAGATATTGCGGTCGCCTCGCTTGCCGCCAAATGGAAGGGCAAATCGAGCGAAGAACTGCTGGAAACGCTCGATTTCTATCGCGACATCATTTCCGGCCCGTTTGAAAAGCTGATCCGCGGCAGCAAAATCCGCGAACTTTCCGGCCCGGAATATAGCCGCAAGGTGATGGAAAACTGCGTCGCGCATCTGAAATCGGTCGGCACCTATGGCGATGCCGAAGCCGAAGCGATGCAGAAATTTGCCGAAGCGTTCAAGCCGGTGAATTTCCCGCCCGGCGCAAGCGTTTTCTATCGCCAATCGCCCGATGGCATTTTGGGCCTGAGCTTCTCGCCCGACACCTCGATTCCGGAAAAAGAAGCCGCGCTGATCGAAAACAAGGCGGTTTCCAGCGCCGTGCTGGAAACGATGATCGGCGAACATGCCGTTTCGCCCGATCTGAAGCGCTGCCTTGCCGCGCGCCTGCCTGCGTTGCTGAACGAAGGCGCGTTCAAGATCGGCAATTGA</cds>",
    "<ncds>GAACTGGTTAAGGTGATGGACAACGGGTGGTAATATGCTATACTAAAGGC</ncds><cds>ATGGCGCCGCGCCCGACCTCGCAATCGCAAGCGCGCACCTGCCCGACGACGCAGGTCACGCAGGTCGATATCGTCGAAAAGATGCTGGCCGCGCCGACCGACAGCACGCTTGAACTCGATGGCTATTCGCTCAATCTTGGCGATGTCGTTAGCGCCGCGCGCAAGGGCCGGCCGGTTCGCGTCAAGGATAGCGACGAAATCCGCAGCAAGATCGACAAGTCGGTCGAGTTTCTGCGCAGCCAGCTTTCGATGAGCGTCTATGGCGTCACGACCGGCTTTGGCGGCAGCGCCGACACGCGCACCGAAGATGCGATCTCGCTGCAAAAGGCGCTGCTCGAGCATCAGCTTTGCGGCGTTCTGCCCTCCAGCTTCGACAGCTTCCGCCTTGGCCGCGGCCTTGAAAACAGCCTGCCGCTCGAAGTCGTGCGCGGCGCGATGACGATCCGCGTCAACTCGCTGACGCGCGGCCATAGCGCCGTTCGGCTGGTCGTGCTCGAAGCGCTGACCAACTTCCTCAATCAGGGCATCACGCCGATCGTTCCGCTGCGCGGCACGATCTCGGCCAGCGGCGATCTTTCGCCGCTCAGCTATATCGCCGCCGCGATCTCCGGCCATCCCGACAGCAAGGTTCATGTCGTTCACGAAGGCAAGGAAAAGATCCTCTATGCCCGCGAAGCGATGGCGCTGTTCAATCTCGAGCCGGTCGTGCTTGGCCCCAAGGAAGGGCTTGGCCTGGTCAACGGCACGGCGGTTTCGGCCTCGATGGCGACGCTGGCGCTGCACGATGCGCACATGCTGTCGCTGCTGTCGCAGTCGCTGACCGCGATGACGGTCGAGGCGATGGTCGGCCATGCCGGCAGCTTCCATCCCTTCCTGCACGATGTGACGCGCCCGCATCCGACGCAGATCGAGGTTGCCGGCAATATCCGCAAGCTGCTCGAAGGCAGCCGCTTTGCCGTTCATCACGAAGAAGAGGTCAAGGTCAAGGACGACGAAGGCATTCTGCGGCAGGATCGCTATCCGCTGCGCACCTCGCCGCAATGGCTTGGGCCGCTGGTCAGCGATCTGATCCATGCCCATGCCGTGCTGACGATCGAGGCCGGCCAATCGACGACCGACAATCCGCTGATCGATGTCGAGAACAAGACCAGCCATCATGGCGGCAATTTCCAGGCCGCCGCCGTCGCCAACACGATGGAAAAGACGCGGCTTGGCCTGGCGCAGATCGGCAAGCTCAACTTCACCCAGCTGACCGAAATGCTGAATGCCGGCATGAACCGCGGCCTGCCGTCCTGCCTGGCGGCCGAAGATCCGTCGCTGTCCTATCATTGCAAGGGGCTCGATATCGCCGCCGCGGCCTACACCAGCGAACTTGGCCATCTGGCCAATCCGGTCACGACGCATGTTCAGCCGGCCGAAATGGCCAATCAGGCGGTCAACTCGCTGGCGCTGATCTCGGCGCGCCGCACGACCGAAAGCAACGATGTTCTGTCGCTGCTGCTGGCGACGCATCTCTATTGCGTGCTGCAGGCGATCGATCTGCGCGCGATCGAGTTCGAGTTCAAGAAGCAGTTCGGCCCGGCGATCGTCTCGCTGATCGATCAGCATTTCGGCTCGGCGATGACCGGCAGCAACCTGCGCGACGAACTGGTCGAAAAGGTCAACAAGACGCTCGCCAAGCGGCTGGAGCAGACCAACAGCTACGATCTCGTGCCGCGCTGGCATGATGCCTTCAGCTTTGCCGCCGGCACGGTGGTCGAAGTGCTGTCCTCGACCTCGCTGTCGCTTGCCGCGGTCAATGCCTGGAAGGTCGCCGCCGCCGAAAGCGCGATCTCGCTGACGCGGCAGGTGCGCGAAACCTTCTGGAGCGCGGCCTCGACCAGCTCGCCGGCGCTGAGCTATCTCTCGCCGCGCACGCAGATCCTCTATGCCTTCGTGCGCGAAGAACTGGGCGTGAAGGCGCGGCGCGGCGATGTGTTCCTCGGCAAGCAGGAAGTCACGATCGGCAGCAACGTCTCGAAGATCTACGAAGCGATCAAGAGCGGCCGCATCAACAACGTGCTGCTGAAGATGCTGGCC</cds><ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><cds>ATGGGTGATTGTGTTGCTCCAAAAGAAGATTTAATTTTTAGATCAAAATTACCTGATATTTATATTCCAAAACATTTGCCTCTTCATACTTATTGTTTTGAAAATATTTCAAAAGTTGGCGATAAATCTTGTTTGATTAATGGTGCAACAGGTGAAACTTTTACCTATTCTCAAGTTGAATTGTTATCGCGCAAAGTTGCATCTGGCTTAAATAAACTTGGCATTCAGCAAGGCGATACGATCATGCTGTTGTTGCCCAATAGCCCAGAATATTTTTTTGCTTTTTTAGGCGCATCTTATCGCGGTGCAATTTCAACAATGGCAAATCCATTTTTTACCTCTGCAGAAGTTATTAAGCAATTAAAAGCATCTCAAGCAAAACTGATTATTACGCAAGCTTGTTATGTTGATAAAGTAAAAGATTATGCTGCTGAAAAAAATATTCAGATTATTTGTATTGATGATGCGCCGCAAGATTGTCTGCATTTTTCAAAATTGATGGAAGCTGATGAAAGCGAAATGCCAGAAGTTGTTATCAATTCTGATGATGTTGTTGCTTTGCCTTATTCATCTGGCACAACAGGTTTGCCAAAAGGCGTGATGCTGACCCATAAAGGTTTGGTGACATCTGTTGCGCAGCAAGTTGATGGCGATAATCCAAATCTTTATATGCATTCTGAAGATGTGATGATTTGCATTTTGCCGCTTTTTCATATTTATTCTTTAAATGCAGTTTTATGTTGTGGTTTGCGTGCAGGTGTTACCATTTTGATCATGCAAAAATTTGATATTGTTCCATTTCTTGAACTTATTCAAAAATATAAAGTTACGATTGGTCCATTTGTTCCGCCAATTGTTTTGGCAATTGCCAAATCTCCTGTTGTTGATAAATATGATTTATCTTCTGTTAGAACGGTTATGTCTGGTGCAGCACCTTTGGGCAAAGAACTTGAAGATGCTGTTCGCGCAAAATTTCCAAATGCAAAACTTGGTCAAGGCTATGGCATGACCGAAGCTGGTCCTGTTTTGGCAATGTGTTTGGCTTTTGCCAAAGAACCTTATGAAATTAAATCTGGTGCTTGCGGCACGGTTGTTCGCAATGCTGAAATGAAAATTGTTGATCCTGAAACCAATGCCTCTTTGCCGCGCAATCAACGCGGTGAAATTTGTATTCGCGGCGATCAGATTATGAAGGGCTATTTGAATGATCCTGAAAGCACCCGCACAACAATTGATGAAGAAGGTTGGCTTCATACTGGCGATATTGGTTTTATTGATGATGATGATGAACTTTTTATTGTTGATCGCTTGAAAGAAATTATCAAATATAAAGGTTTTCAAGTTGCTCCTGCTGAACTTGAAGCTTTGTTGTTAACCCATCCAACAATTTCTGATGCTGCTGTTGTTCCGATGATTGATGAAAAAGCTGGTGAAGTTCCTGTTGCTTTTGTTGTTAGAACCAATGGCTTTACAACAACTGAAGAAGAAATTAAGCAATTTGTTTCAAAGCAAGTTGTTTTTTATAAACGTATTTTCCGCGTTTTTTTTGTTGATGCAATTCCAAAATCACCATCTGGCAAAATTTTGCGCAAAGATTTGCGCGCAAGAATTGCTTCTGGCGATTTGCCAAAA</cds><ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><cds>ATGGTTACAGTTGAAGAATATCGCAAAGCACAACGCGCAGAAGGGCCTGCAACAGTTATGGCAATTGGCACAGCAACGCCAACCAATTGCGTCGATCAATCAACCTATCCTGATTATTATTTTCGCATCACCAATAGCGAACATAAAACCGATTTAAAAGAAAAATTTAAGCGGATGTGTGAAAAATCGATGATCAAAAAACGCTATATGCATCTGACCGAAGAAATTTTGAAAGAAAATCCATCAATGTGTGAATATATGGCGCCATCTCTTGATGCGCGGCAAGATATTGTTGTTGTTGAAGTTCCAAAACTTGGCAAAGAAGCTGCGCAAAAAGCGATCAAAGAATGGGGGCAGCCAAAATCAAAAATTACGCATTTGGTTTTTTGCACCACATCTGGCGTTGATATGCCGGGCTGTGATTATCAACTGACCAAATTGCTTGGCTTACGCCCATCTGTAAAACGGTTGATGATGTATCAGCAAGGCTGTTTTGCTGGCGGCACAGTTTTGCGTTTGGCAAAAGATTTGGCAGAAAATAATAAAGGCGCGCGCGTTTTGGTTGTTTGTTCTGAAATTACTGCAGTTACTTTTCGCGGCCCAAATGATACGCATCTTGATAGTTTGGTTGGTCAAGCTTTATTTGGTGATGGTGCAGGCGCAATTATTATTGGTTCTGATCCAATTCCCGGTGTTGAACGCCCTTTATTTGAATTGGTTTCTGCAGCGCAAACATTATTGCCTGATAGCCATGGCGCAATTGATGGGCATTTGCGCGAAGTTGGCCTGACCTTTCATTTGTTAAAAGATGTTCCTGGCCTGATTTCAAAAAATATTGAAAAATCTTTGGAAGAAGCTTTTCGCCCACTTTCAATTACCGATTGGAATAGCTTATTTTGGATTGCTCATCCCGGTGGCCCTGCAATTTTGGATCAAGTTGAAATCAAACTTGGCTTAAAGCCAGAAAAATTAAAAGCAACACGCAATGTTCTTTCCAATTATGGCAATATGTCTTCTGCTTGCGTTCTGTTTATTTTGGATGAAATGCGCAAAGCATCTGCAAAAGAAGGTCTTGGCACAACTGGCGAAGGTTTGGAATGGGGCGTTTTATTTGGCTTTGGCCCTGGCTTAACGGTTGAAACAGTTGTTCTGCATTCTGTTGCAACA</cds><ncds>GAACTGGTTAAGGTGATGGACAACGGGTGGTAATATGCTATACTAAAGGC</ncds><cds>ATGGCAGCAAGCATCACCGCAATCACCGTTGAAAATCTGGAATATCCGGCCGTCGTCACCTCGCCGGTGACCGGCAAAAGCTATTTTCTTGGCGGCGCGGGCGAGCGCGGCCTGACGATCGAAGGCAATTTCATCAAATTCACCGCGATCGGCGTCTATCTGGAAGATATTGCGGTCGCCTCGCTTGCCGCCAAATGGAAGGGCAAATCGAGCGAAGAACTGCTGGAAACGCTCGATTTCTATCGCGACATCATTTCCGGCCCGTTTGAAAAGCTGATCCGCGGCAGCAAAATCCGCGAACTTTCCGGCCCGGAATATAGCCGCAAGGTGATGGAAAACTGCGTCGCGCATCTGAAATCGGTCGGCACCTATGGCGATGCCGAAGCCGAAGCGATGCAGAAATTTGCCGAAGCGTTCAAGCCGGTGAATTTCCCGCCCGGCGCAAGCGTTTTCTATCGCCAATCGCCCGATGGCATTTTGGGCCTGAGCTTCTCGCCCGACACCTCGATTCCGGAAAAAGAAGCCGCGCTGATCGAAAACAAGGCGGTTTCCAGCGCCGTGCTGGAAACGATGATCGGCGAACATGCCGTTTCGCCCGATCTGAAGCGCTGCCTTGCCGCGCGCCTGCCTGCGTTGCTGAACGAAGGCGCGTTCAAGATCGGCAATTGA</cds>",
    "<ncds>GAACTGGTTAAGGTGATGGACAACGGGTGGTAATATGCTATACTAAAGGC</ncds><cds>ATGGCGCCGCGCCCGACCTCGCAATCGCAAGCGCGCACCTGCCCGACGACGCAGGTCACGCAGGTCGATATCGTCGAAAAGATGCTGGCCGCGCCGACCGACAGCACGCTTGAACTCGATGGCTATTCGCTCAATCTTGGCGATGTCGTTAGCGCCGCGCGCAAGGGCCGGCCGGTTCGCGTCAAGGATAGCGACGAAATCCGCAGCAAGATCGACAAGTCGGTCGAGTTTCTGCGCAGCCAGCTTTCGATGAGCGTCTATGGCGTCACGACCGGCTTTGGCGGCAGCGCCGACACGCGCACCGAAGATGCGATCTCGCTGCAAAAGGCGCTGCTCGAGCATCAGCTTTGCGGCGTTCTGCCCTCCAGCTTCGACAGCTTCCGCCTTGGCCGCGGCCTTGAAAACAGCCTGCCGCTCGAAGTCGTGCGCGGCGCGATGACGATCCGCGTCAACTCGCTGACGCGCGGCCATAGCGCCGTTCGGCTGGTCGTGCTCGAAGCGCTGACCAACTTCCTCAATCAGGGCATCACGCCGATCGTTCCGCTGCGCGGCACGATCTCGGCCAGCGGCGATCTTTCGCCGCTCAGCTATATCGCCGCCGCGATCTCCGGCCATCCCGACAGCAAGGTTCATGTCGTTCACGAAGGCAAGGAAAAGATCCTCTATGCCCGCGAAGCGATGGCGCTGTTCAATCTCGAGCCGGTCGTGCTTGGCCCCAAGGAAGGGCTTGGCCTGGTCAACGGCACGGCGGTTTCGGCCTCGATGGCGACGCTGGCGCTGCACGATGCGCACATGCTGTCGCTGCTGTCGCAGTCGCTGACCGCGATGACGGTCGAGGCGATGGTCGGCCATGCCGGCAGCTTCCATCCCTTCCTGCACGATGTGACGCGCCCGCATCCGACGCAGATCGAGGTTGCCGGCAATATCCGCAAGCTGCTCGAAGGCAGCCGCTTTGCCGTTCATCACGAAGAAGAGGTCAAGGTCAAGGACGACGAAGGCATTCTGCGGCAGGATCGCTATCCGCTGCGCACCTCGCCGCAATGGCTTGGGCCGCTGGTCAGCGATCTGATCCATGCCCATGCCGTGCTGACGATCGAGGCCGGCCAATCGACGACCGACAATCCGCTGATCGATGTCGAGAACAAGACCAGCCATCATGGCGGCAATTTCCAGGCCGCCGCCGTCGCCAACACGATGGAAAAGACGCGGCTTGGCCTGGCGCAGATCGGCAAGCTCAACTTCACCCAGCTGACCGAAATGCTGAATGCCGGCATGAACCGCGGCCTGCCGTCCTGCCTGGCGGCCGAAGATCCGTCGCTGTCCTATCATTGCAAGGGGCTCGATATCGCCGCCGCGGCCTACACCAGCGAACTTGGCCATCTGGCCAATCCGGTCACGACGCATGTTCAGCCGGCCGAAATGGCCAATCAGGCGGTCAACTCGCTGGCGCTGATCTCGGCGCGCCGCACGACCGAAAGCAACGATGTTCTGTCGCTGCTGCTGGCGACGCATCTCTATTGCGTGCTGCAGGCGATCGATCTGCGCGCGATCGAGTTCGAGTTCAAGAAGCAGTTCGGCCCGGCGATCGTCTCGCTGATCGATCAGCATTTCGGCTCGGCGATGACCGGCAGCAACCTGCGCGACGAACTGGTCGAAAAGGTCAACAAGACGCTCGCCAAGCGGCTGGAGCAGACCAACAGCTACGATCTCGTGCCGCGCTGGCATGATGCCTTCAGCTTTGCCGCCGGCACGGTGGTCGAAGTGCTGTCCTCGACCTCGCTGTCGCTTGCCGCGGTCAATGCCTGGAAGGTCGCCGCCGCCGAAAGCGCGATCTCGCTGACGCGGCAGGTGCGCGAAACCTTCTGGAGCGCGGCCTCGACCAGCTCGCCGGCGCTGAGCTATCTCTCGCCGCGCACGCAGATCCTCTATGCCTTCGTGCGCGAAGAACTGGGCGTGAAGGCGCGGCGCGGCGATGTGTTCCTCGGCAAGCAGGAAGTCACGATCGGCAGCAACGTCTCGAAGATCTACGAAGCGATCAAGAGCGGCCGCATCAACAACGTGCTGCTGAAGATGCTGGCC</cds><ncds>TTATGATAAGTTCACTGTTGGCTAAAGGGAGTGTAAGTCATAATGAACTT</ncds><cds>ATGGGTGATTGTGTTGCTCCAAAAGAAGATTTAATTTTTAGATCAAAATTACCTGATATTTATATTCCAAAACATTTGCCTCTTCATACTTATTGTTTTGAAAATATTTCAAAAGTTGGCGATAAATCTTGTTTGATTAATGGTGCAACAGGTGAAACTTTTACCTATTCTCAAGTTGAATTGTTATCGCGCAAAGTTGCATCTGGCTTAAATAAACTTGGCATTCAGCAAGGCGATACGATCATGCTGTTGTTGCCCAATAGCCCAGAATATTTTTTTGCTTTTTTAGGCGCATCTTATCGCGGTGCAATTTCAACAATGGCAAATCCATTTTTTACCTCTGCAGAAGTTATTAAGCAATTAAAAGCATCTCAAGCAAAACTGATTATTACGCAAGCTTGTTATGTTGATAAAGTAAAAGATTATGCTGCTGAAAAAAATATTCAGATTATTTGTATTGATGATGCGCCGCAAGATTGTCTGCATTTTTCAAAATTGATGGAAGCTGATGAAAGCGAAATGCCAGAAGTTGTTATCAATTCTGATGATGTTGTTGCTTTGCCTTATTCATCTGGCACAACAGGTTTGCCAAAAGGCGTGATGCTGACCCATAAAGGTTTGGTGACATCTGTTGCGCAGCAAGTTGATGGCGATAATCCAAATCTTTATATGCATTCTGAAGATGTGATGATTTGCATTTTGCCGCTTTTTCATATTTATTCTTTAAATGCAGTTTTATGTTGTGGTTTGCGTGCAGGTGTTACCATTTTGATCATGCAAAAATTTGATATTGTTCCATTTCTTGAACTTATTCAAAAATATAAAGTTACGATTGGTCCATTTGTTCCGCCAATTGTTTTGGCAATTGCCAAATCTCCTGTTGTTGATAAATATGATTTATCTTCTGTTAGAACGGTTATGTCTGGTGCAGCACCTTTGGGCAAAGAACTTGAAGATGCTGTTCGCGCAAAATTTCCAAATGCAAAACTTGGTCAAGGCTATGGCATGACCGAAGCTGGTCCTGTTTTGGCAATGTGTTTGGCTTTTGCCAAAGAACCTTATGAAATTAAATCTGGTGCTTGCGGCACGGTTGTTCGCAATGCTGAAATGAAAATTGTTGATCCTGAAACCAATGCCTCTTTGCCGCGCAATCAACGCGGTGAAATTTGTATTCGCGGCGATCAGATTATGAAGGGCTATTTGAATGATCCTGAAAGCACCCGCACAACAATTGATGAAGAAGGTTGGCTTCATACTGGCGATATTGGTTTTATTGATGATGATGATGAACTTTTTATTGTTGATCGCTTGAAAGAAATTATCAAATATAAAGGTTTTCAAGTTGCTCCTGCTGAACTTGAAGCTTTGTTGTTAACCCATCCAACAATTTCTGATGCTGCTGTTGTTCCGATGATTGATGAAAAAGCTGGTGAAGTTCCTGTTGCTTTTGTTGTTAGAACCAATGGCTTTACAACAACTGAAGAAGAAATTAAGCAATTTGTTTCAAAGCAAGTTGTTTTTTATAAACGTATTTTCCGCGTTTTTTTTGTTGATGCAATTCCAAAATCACCATCTGGCAAAATTTTGCGCAAAGATTTGCGCGCAAGAATTGCTTCTGGCGATTTGCCAAAA</cds><ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><cds>ATGGTTACAGTTGAAGAATATCGCAAAGCACAACGCGCAGAAGGGCCTGCAACAGTTATGGCAATTGGCACAGCAACGCCAACCAATTGCGTCGATCAATCAACCTATCCTGATTATTATTTTCGCATCACCAATAGCGAACATAAAACCGATTTAAAAGAAAAATTTAAGCGGATGTGTGAAAAATCGATGATCAAAAAACGCTATATGCATCTGACCGAAGAAATTTTGAAAGAAAATCCATCAATGTGTGAATATATGGCGCCATCTCTTGATGCGCGGCAAGATATTGTTGTTGTTGAAGTTCCAAAACTTGGCAAAGAAGCTGCGCAAAAAGCGATCAAAGAATGGGGGCAGCCAAAATCAAAAATTACGCATTTGGTTTTTTGCACCACATCTGGCGTTGATATGCCGGGCTGTGATTATCAACTGACCAAATTGCTTGGCTTACGCCCATCTGTAAAACGGTTGATGATGTATCAGCAAGGCTGTTTTGCTGGCGGCACAGTTTTGCGTTTGGCAAAAGATTTGGCAGAAAATAATAAAGGCGCGCGCGTTTTGGTTGTTTGTTCTGAAATTACTGCAGTTACTTTTCGCGGCCCAAATGATACGCATCTTGATAGTTTGGTTGGTCAAGCTTTATTTGGTGATGGTGCAGGCGCAATTATTATTGGTTCTGATCCAATTCCCGGTGTTGAACGCCCTTTATTTGAATTGGTTTCTGCAGCGCAAACATTATTGCCTGATAGCCATGGCGCAATTGATGGGCATTTGCGCGAAGTTGGCCTGACCTTTCATTTGTTAAAAGATGTTCCTGGCCTGATTTCAAAAAATATTGAAAAATCTTTGGAAGAAGCTTTTCGCCCACTTTCAATTACCGATTGGAATAGCTTATTTTGGATTGCTCATCCCGGTGGCCCTGCAATTTTGGATCAAGTTGAAATCAAACTTGGCTTAAAGCCAGAAAAATTAAAAGCAACACGCAATGTTCTTTCCAATTATGGCAATATGTCTTCTGCTTGCGTTCTGTTTATTTTGGATGAAATGCGCAAAGCATCTGCAAAAGAAGGTCTTGGCACAACTGGCGAAGGTTTGGAATGGGGCGTTTTATTTGGCTTTGGCCCTGGCTTAACGGTTGAAACAGTTGTTCTGCATTCTGTTGCAACA</cds><ncds>TTATGATAAGTTCACTGTTGGCTAAAGGGAGTGTAAGTCATAATGAACTT</ncds><cds>ATGGCAGCAAGCATCACCGCAATCACCGTTGAAAATCTGGAATATCCGGCCGTCGTCACCTCGCCGGTGACCGGCAAAAGCTATTTTCTTGGCGGCGCGGGCGAGCGCGGCCTGACGATCGAAGGCAATTTCATCAAATTCACCGCGATCGGCGTCTATCTGGAAGATATTGCGGTCGCCTCGCTTGCCGCCAAATGGAAGGGCAAATCGAGCGAAGAACTGCTGGAAACGCTCGATTTCTATCGCGACATCATTTCCGGCCCGTTTGAAAAGCTGATCCGCGGCAGCAAAATCCGCGAACTTTCCGGCCCGGAATATAGCCGCAAGGTGATGGAAAACTGCGTCGCGCATCTGAAATCGGTCGGCACCTATGGCGATGCCGAAGCCGAAGCGATGCAGAAATTTGCCGAAGCGTTCAAGCCGGTGAATTTCCCGCCCGGCGCAAGCGTTTTCTATCGCCAATCGCCCGATGGCATTTTGGGCCTGAGCTTCTCGCCCGACACCTCGATTCCGGAAAAAGAAGCCGCGCTGATCGAAAACAAGGCGGTTTCCAGCGCCGTGCTGGAAACGATGATCGGCGAACATGCCGTTTCGCCCGATCTGAAGCGCTGCCTTGCCGCGCGCCTGCCTGCGTTGCTGAACGAAGGCGCGTTCAAGATCGGCAATTGA</cds>",
    "<ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><cds>ATGGCGCCGCGCCCGACCTCGCAATCGCAAGCGCGCACCTGCCCGACGACGCAGGTCACGCAGGTCGATATCGTCGAAAAGATGCTGGCCGCGCCGACCGACAGCACGCTTGAACTCGATGGCTATTCGCTCAATCTTGGCGATGTCGTTAGCGCCGCGCGCAAGGGCCGGCCGGTTCGCGTCAAGGATAGCGACGAAATCCGCAGCAAGATCGACAAGTCGGTCGAGTTTCTGCGCAGCCAGCTTTCGATGAGCGTCTATGGCGTCACGACCGGCTTTGGCGGCAGCGCCGACACGCGCACCGAAGATGCGATCTCGCTGCAAAAGGCGCTGCTCGAGCATCAGCTTTGCGGCGTTCTGCCCTCCAGCTTCGACAGCTTCCGCCTTGGCCGCGGCCTTGAAAACAGCCTGCCGCTCGAAGTCGTGCGCGGCGCGATGACGATCCGCGTCAACTCGCTGACGCGCGGCCATAGCGCCGTTCGGCTGGTCGTGCTCGAAGCGCTGACCAACTTCCTCAATCAGGGCATCACGCCGATCGTTCCGCTGCGCGGCACGATCTCGGCCAGCGGCGATCTTTCGCCGCTCAGCTATATCGCCGCCGCGATCTCCGGCCATCCCGACAGCAAGGTTCATGTCGTTCACGAAGGCAAGGAAAAGATCCTCTATGCCCGCGAAGCGATGGCGCTGTTCAATCTCGAGCCGGTCGTGCTTGGCCCCAAGGAAGGGCTTGGCCTGGTCAACGGCACGGCGGTTTCGGCCTCGATGGCGACGCTGGCGCTGCACGATGCGCACATGCTGTCGCTGCTGTCGCAGTCGCTGACCGCGATGACGGTCGAGGCGATGGTCGGCCATGCCGGCAGCTTCCATCCCTTCCTGCACGATGTGACGCGCCCGCATCCGACGCAGATCGAGGTTGCCGGCAATATCCGCAAGCTGCTCGAAGGCAGCCGCTTTGCCGTTCATCACGAAGAAGAGGTCAAGGTCAAGGACGACGAAGGCATTCTGCGGCAGGATCGCTATCCGCTGCGCACCTCGCCGCAATGGCTTGGGCCGCTGGTCAGCGATCTGATCCATGCCCATGCCGTGCTGACGATCGAGGCCGGCCAATCGACGACCGACAATCCGCTGATCGATGTCGAGAACAAGACCAGCCATCATGGCGGCAATTTCCAGGCCGCCGCCGTCGCCAACACGATGGAAAAGACGCGGCTTGGCCTGGCGCAGATCGGCAAGCTCAACTTCACCCAGCTGACCGAAATGCTGAATGCCGGCATGAACCGCGGCCTGCCGTCCTGCCTGGCGGCCGAAGATCCGTCGCTGTCCTATCATTGCAAGGGGCTCGATATCGCCGCCGCGGCCTACACCAGCGAACTTGGCCATCTGGCCAATCCGGTCACGACGCATGTTCAGCCGGCCGAAATGGCCAATCAGGCGGTCAACTCGCTGGCGCTGATCTCGGCGCGCCGCACGACCGAAAGCAACGATGTTCTGTCGCTGCTGCTGGCGACGCATCTCTATTGCGTGCTGCAGGCGATCGATCTGCGCGCGATCGAGTTCGAGTTCAAGAAGCAGTTCGGCCCGGCGATCGTCTCGCTGATCGATCAGCATTTCGGCTCGGCGATGACCGGCAGCAACCTGCGCGACGAACTGGTCGAAAAGGTCAACAAGACGCTCGCCAAGCGGCTGGAGCAGACCAACAGCTACGATCTCGTGCCGCGCTGGCATGATGCCTTCAGCTTTGCCGCCGGCACGGTGGTCGAAGTGCTGTCCTCGACCTCGCTGTCGCTTGCCGCGGTCAATGCCTGGAAGGTCGCCGCCGCCGAAAGCGCGATCTCGCTGACGCGGCAGGTGCGCGAAACCTTCTGGAGCGCGGCCTCGACCAGCTCGCCGGCGCTGAGCTATCTCTCGCCGCGCACGCAGATCCTCTATGCCTTCGTGCGCGAAGAACTGGGCGTGAAGGCGCGGCGCGGCGATGTGTTCCTCGGCAAGCAGGAAGTCACGATCGGCAGCAACGTCTCGAAGATCTACGAAGCGATCAAGAGCGGCCGCATCAACAACGTGCTGCTGAAGATGCTGGCC</cds><ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><cds>ATGGGTGATTGTGTTGCTCCAAAAGAAGATTTAATTTTTAGATCAAAATTACCTGATATTTATATTCCAAAACATTTGCCTCTTCATACTTATTGTTTTGAAAATATTTCAAAAGTTGGCGATAAATCTTGTTTGATTAATGGTGCAACAGGTGAAACTTTTACCTATTCTCAAGTTGAATTGTTATCGCGCAAAGTTGCATCTGGCTTAAATAAACTTGGCATTCAGCAAGGCGATACGATCATGCTGTTGTTGCCCAATAGCCCAGAATATTTTTTTGCTTTTTTAGGCGCATCTTATCGCGGTGCAATTTCAACAATGGCAAATCCATTTTTTACCTCTGCAGAAGTTATTAAGCAATTAAAAGCATCTCAAGCAAAACTGATTATTACGCAAGCTTGTTATGTTGATAAAGTAAAAGATTATGCTGCTGAAAAAAATATTCAGATTATTTGTATTGATGATGCGCCGCAAGATTGTCTGCATTTTTCAAAATTGATGGAAGCTGATGAAAGCGAAATGCCAGAAGTTGTTATCAATTCTGATGATGTTGTTGCTTTGCCTTATTCATCTGGCACAACAGGTTTGCCAAAAGGCGTGATGCTGACCCATAAAGGTTTGGTGACATCTGTTGCGCAGCAAGTTGATGGCGATAATCCAAATCTTTATATGCATTCTGAAGATGTGATGATTTGCATTTTGCCGCTTTTTCATATTTATTCTTTAAATGCAGTTTTATGTTGTGGTTTGCGTGCAGGTGTTACCATTTTGATCATGCAAAAATTTGATATTGTTCCATTTCTTGAACTTATTCAAAAATATAAAGTTACGATTGGTCCATTTGTTCCGCCAATTGTTTTGGCAATTGCCAAATCTCCTGTTGTTGATAAATATGATTTATCTTCTGTTAGAACGGTTATGTCTGGTGCAGCACCTTTGGGCAAAGAACTTGAAGATGCTGTTCGCGCAAAATTTCCAAATGCAAAACTTGGTCAAGGCTATGGCATGACCGAAGCTGGTCCTGTTTTGGCAATGTGTTTGGCTTTTGCCAAAGAACCTTATGAAATTAAATCTGGTGCTTGCGGCACGGTTGTTCGCAATGCTGAAATGAAAATTGTTGATCCTGAAACCAATGCCTCTTTGCCGCGCAATCAACGCGGTGAAATTTGTATTCGCGGCGATCAGATTATGAAGGGCTATTTGAATGATCCTGAAAGCACCCGCACAACAATTGATGAAGAAGGTTGGCTTCATACTGGCGATATTGGTTTTATTGATGATGATGATGAACTTTTTATTGTTGATCGCTTGAAAGAAATTATCAAATATAAAGGTTTTCAAGTTGCTCCTGCTGAACTTGAAGCTTTGTTGTTAACCCATCCAACAATTTCTGATGCTGCTGTTGTTCCGATGATTGATGAAAAAGCTGGTGAAGTTCCTGTTGCTTTTGTTGTTAGAACCAATGGCTTTACAACAACTGAAGAAGAAATTAAGCAATTTGTTTCAAAGCAAGTTGTTTTTTATAAACGTATTTTCCGCGTTTTTTTTGTTGATGCAATTCCAAAATCACCATCTGGCAAAATTTTGCGCAAAGATTTGCGCGCAAGAATTGCTTCTGGCGATTTGCCAAAA</cds><ncds>TTATGATAAGTTCACTGTTGGCTAAAGGGAGTGTAAGTCATAATGAACTT</ncds><cds>ATGGTTACAGTTGAAGAATATCGCAAAGCACAACGCGCAGAAGGGCCTGCAACAGTTATGGCAATTGGCACAGCAACGCCAACCAATTGCGTCGATCAATCAACCTATCCTGATTATTATTTTCGCATCACCAATAGCGAACATAAAACCGATTTAAAAGAAAAATTTAAGCGGATGTGTGAAAAATCGATGATCAAAAAACGCTATATGCATCTGACCGAAGAAATTTTGAAAGAAAATCCATCAATGTGTGAATATATGGCGCCATCTCTTGATGCGCGGCAAGATATTGTTGTTGTTGAAGTTCCAAAACTTGGCAAAGAAGCTGCGCAAAAAGCGATCAAAGAATGGGGGCAGCCAAAATCAAAAATTACGCATTTGGTTTTTTGCACCACATCTGGCGTTGATATGCCGGGCTGTGATTATCAACTGACCAAATTGCTTGGCTTACGCCCATCTGTAAAACGGTTGATGATGTATCAGCAAGGCTGTTTTGCTGGCGGCACAGTTTTGCGTTTGGCAAAAGATTTGGCAGAAAATAATAAAGGCGCGCGCGTTTTGGTTGTTTGTTCTGAAATTACTGCAGTTACTTTTCGCGGCCCAAATGATACGCATCTTGATAGTTTGGTTGGTCAAGCTTTATTTGGTGATGGTGCAGGCGCAATTATTATTGGTTCTGATCCAATTCCCGGTGTTGAACGCCCTTTATTTGAATTGGTTTCTGCAGCGCAAACATTATTGCCTGATAGCCATGGCGCAATTGATGGGCATTTGCGCGAAGTTGGCCTGACCTTTCATTTGTTAAAAGATGTTCCTGGCCTGATTTCAAAAAATATTGAAAAATCTTTGGAAGAAGCTTTTCGCCCACTTTCAATTACCGATTGGAATAGCTTATTTTGGATTGCTCATCCCGGTGGCCCTGCAATTTTGGATCAAGTTGAAATCAAACTTGGCTTAAAGCCAGAAAAATTAAAAGCAACACGCAATGTTCTTTCCAATTATGGCAATATGTCTTCTGCTTGCGTTCTGTTTATTTTGGATGAAATGCGCAAAGCATCTGCAAAAGAAGGTCTTGGCACAACTGGCGAAGGTTTGGAATGGGGCGTTTTATTTGGCTTTGGCCCTGGCTTAACGGTTGAAACAGTTGTTCTGCATTCTGTTGCAACA</cds><ncds>GCCGTCAGTTTAGCCAGACCGCCGAAAATTTCCAACACTTCTGAAACACC</ncds><cds>ATGGCAGCAAGCATCACCGCAATCACCGTTGAAAATCTGGAATATCCGGCCGTCGTCACCTCGCCGGTGACCGGCAAAAGCTATTTTCTTGGCGGCGCGGGCGAGCGCGGCCTGACGATCGAAGGCAATTTCATCAAATTCACCGCGATCGGCGTCTATCTGGAAGATATTGCGGTCGCCTCGCTTGCCGCCAAATGGAAGGGCAAATCGAGCGAAGAACTGCTGGAAACGCTCGATTTCTATCGCGACATCATTTCCGGCCCGTTTGAAAAGCTGATCCGCGGCAGCAAAATCCGCGAACTTTCCGGCCCGGAATATAGCCGCAAGGTGATGGAAAACTGCGTCGCGCATCTGAAATCGGTCGGCACCTATGGCGATGCCGAAGCCGAAGCGATGCAGAAATTTGCCGAAGCGTTCAAGCCGGTGAATTTCCCGCCCGGCGCAAGCGTTTTCTATCGCCAATCGCCCGATGGCATTTTGGGCCTGAGCTTCTCGCCCGACACCTCGATTCCGGAAAAAGAAGCCGCGCTGATCGAAAACAAGGCGGTTTCCAGCGCCGTGCTGGAAACGATGATCGGCGAACATGCCGTTTCGCCCGATCTGAAGCGCTGCCTTGCCGCGCGCCTGCCTGCGTTGCTGAACGAAGGCGCGTTCAAGATCGGCAATTGA</cds>",
    "<ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><aas>MAPRPTSQSQARTCPTTQVTQVDIVEKMLAAPTDSTLELDGYSLNLGDVVSAARKGRPVRVKDSDEIRSKIDKSVEFLRSQLSMSVYGVTTGFGGSADTRTEDAISLQKALLEHQLCGVLPSSFDSFRLGRGLENSLPLEVVRGAMTIRVNSLTRGHSAVRLVVLEALTNFLNQGITPIVPLRGTISASGDLSPLSYIAAAISGHPDSKVHVVHEGKEKILYAREAMALFNLEPVVLGPKEGLGLVNGTAVSASMATLALHDAHMLSLLSQSLTAMTVEAMVGHAGSFHPFLHDVTRPHPTQIEVAGNIRKLLEGSRFAVHHEEEVKVKDDEGILRQDRYPLRTSPQWLGPLVSDLIHAHAVLTIEAGQSTTDNPLIDVENKTSHHGGNFQAAAVANTMEKTRLGLAQIGKLNFTQLTEMLNAGMNRGLPSCLAAEDPSLSYHCKGLDIAAAAYTSELGHLANPVTTHVQPAEMANQAVNSLALISARRTTESNDVLSLLLATHLYCVLQAIDLRAIEFEFKKQFGPAIVSLIDQHFGSAMTGSNLRDELVEKVNKTLAKRLEQTNSYDLVPRWHDAFSFAAGTVVEVLSSTSLSLAAVNAWKVAAAESAISLTRQVRETFWSAASTSSPALSYLSPRTQILYAFVREELGVKARRGDVFLGKQEVTIGSNVSKIYEAIKSGRINNVLLKMLA</aas><ncds>GCGAAGTCGGAAAACTTCTGTTCTGTTAAATGTGTTTTGCTCATAGTGTGGTAGAATATCAGCTTACTATTGCTTTACGAAAGCGTATCCGGTGAAATAAAGTCAACCTTTAGTTGGTTAATGTTACACCAACAACGAAACCAACACGCCAGGCTTATTCCTGTGGAGTTATAT</ncds><aas>MGDCVAPKEDLIFRSKLPDIYIPKHLPLHTYCFENISKVGDKSCLINGATGETFTYSQVELLSRKVASGLNKLGIQQGDTIMLLLPNSPEYFFAFLGASYRGAISTMANPFFTSAEVIKQLKASQAKLIITQACYVDKVKDYAAEKNIQIICIDDAPQDCLHFSKLMEADESEMPEVVINSDDVVALPYSSGTTGLPKGVMLTHKGLVTSVAQQVDGDNPNLYMHSEDVMICILPLFHIYSLNAVLCCGLRAGVTILIMQKFDIVPFLELIQKYKVTIGPFVPPIVLAIAKSPVVDKYDLSSVRTVMSGAAPLGKELEDAVRAKFPNAKLGQGYGMTEAGPVLAMCLAFAKEPYEIKSGACGTVVRNAEMKIVDPETNASLPRNQRGEICIRGDQIMKGYLNDPESTRTTIDEEGWLHTGDIGFIDDDDELFIVDRLKEIIKYKGFQVAPAELEALLLTHPTISDAAVVPMIDEKAGEVPVAFVVRTNGFTTTEEEIKQFVSKQVVFYKRIFRVFFVDAIPKSPSGKILRKDLRARIASGDLPK</aas><ncds>CGAATCACTAGATCTCTGGCATTGATTTAAATAAGATAAAAGTATGACTT</ncds><aas>MVTVEEYRKAQRAEGPATVMAIGTATPTNCVDQSTYPDYYFRITNSEHKTDLKEKFKRMCEKSMIKKRYMHLTEEILKENPSMCEYMAPSLDARQDIVVVEVPKLGKEAAQKAIKEWGQPKSKITHLVFCTTSGVDMPGCDYQLTKLLGLRPSVKRLMMYQQGCFAGGTVLRLAKDLAENNKGARVLVVCSEITAVTFRGPNDTHLDSLVGQALFGDGAGAIIIGSDPIPGVERPLFELVSAAQTLLPDSHGAIDGHLREVGLTFHLLKDVPGLISKNIEKSLEEAFRPLSITDWNSLFWIAHPGGPAILDQVEIKLGLKPEKLKATRNVLSNYGNMSSACVLFILDEMRKASAKEGLGTTGEGLEWGVLFGFGPGLTVETVVLHSVAT</aas><aas>MAASITAITVENLEYPAVVTSPVTGKSYFLGGAGERGLTIEGNFIKFTAIGVYLEDIAVASLAAKWKGKSSEELLETLDFYRDIISGPFEKLIRGSKIRELSGPEYSRKVMENCVAHLKSVGTYGDAEAEAMQKFAEAFKPVNFPPGASVFYRQSPDGILGLSFSPDTSIPEKEAALIENKAVSSAVLETMIGEHAVSPDLKRCLAARLPALLNEGAFKIGN*</aas>",
    "<ncds>GAACTGGTTAAGGTGATGGACAACGGGTGGTAATATGCTATACTAAAGGC</ncds><aas>MAPRPTSQSQARTCPTTQVTQVDIVEKMLAAPTDSTLELDGYSLNLGDVVSAARKGRPVRVKDSDEIRSKIDKSVEFLRSQLSMSVYGVTTGFGGSADTRTEDAISLQKALLEHQLCGVLPSSFDSFRLGRGLENSLPLEVVRGAMTIRVNSLTRGHSAVRLVVLEALTNFLNQGITPIVPLRGTISASGDLSPLSYIAAAISGHPDSKVHVVHEGKEKILYAREAMALFNLEPVVLGPKEGLGLVNGTAVSASMATLALHDAHMLSLLSQSLTAMTVEAMVGHAGSFHPFLHDVTRPHPTQIEVAGNIRKLLEGSRFAVHHEEEVKVKDDEGILRQDRYPLRTSPQWLGPLVSDLIHAHAVLTIEAGQSTTDNPLIDVENKTSHHGGNFQAAAVANTMEKTRLGLAQIGKLNFTQLTEMLNAGMNRGLPSCLAAEDPSLSYHCKGLDIAAAAYTSELGHLANPVTTHVQPAEMANQAVNSLALISARRTTESNDVLSLLLATHLYCVLQAIDLRAIEFEFKKQFGPAIVSLIDQHFGSAMTGSNLRDELVEKVNKTLAKRLEQTNSYDLVPRWHDAFSFAAGTVVEVLSSTSLSLAAVNAWKVAAAESAISLTRQVRETFWSAASTSSPALSYLSPRTQILYAFVREELGVKARRGDVFLGKQEVTIGSNVSKIYEAIKSGRINNVLLKMLA</aas><ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><aas>MGDCVAPKEDLIFRSKLPDIYIPKHLPLHTYCFENISKVGDKSCLINGATGETFTYSQVELLSRKVASGLNKLGIQQGDTIMLLLPNSPEYFFAFLGASYRGAISTMANPFFTSAEVIKQLKASQAKLIITQACYVDKVKDYAAEKNIQIICIDDAPQDCLHFSKLMEADESEMPEVVINSDDVVALPYSSGTTGLPKGVMLTHKGLVTSVAQQVDGDNPNLYMHSEDVMICILPLFHIYSLNAVLCCGLRAGVTILIMQKFDIVPFLELIQKYKVTIGPFVPPIVLAIAKSPVVDKYDLSSVRTVMSGAAPLGKELEDAVRAKFPNAKLGQGYGMTEAGPVLAMCLAFAKEPYEIKSGACGTVVRNAEMKIVDPETNASLPRNQRGEICIRGDQIMKGYLNDPESTRTTIDEEGWLHTGDIGFIDDDDELFIVDRLKEIIKYKGFQVAPAELEALLLTHPTISDAAVVPMIDEKAGEVPVAFVVRTNGFTTTEEEIKQFVSKQVVFYKRIFRVFFVDAIPKSPSGKILRKDLRARIASGDLPK</aas><ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><aas>MVTVEEYRKAQRAEGPATVMAIGTATPTNCVDQSTYPDYYFRITNSEHKTDLKEKFKRMCEKSMIKKRYMHLTEEILKENPSMCEYMAPSLDARQDIVVVEVPKLGKEAAQKAIKEWGQPKSKITHLVFCTTSGVDMPGCDYQLTKLLGLRPSVKRLMMYQQGCFAGGTVLRLAKDLAENNKGARVLVVCSEITAVTFRGPNDTHLDSLVGQALFGDGAGAIIIGSDPIPGVERPLFELVSAAQTLLPDSHGAIDGHLREVGLTFHLLKDVPGLISKNIEKSLEEAFRPLSITDWNSLFWIAHPGGPAILDQVEIKLGLKPEKLKATRNVLSNYGNMSSACVLFILDEMRKASAKEGLGTTGEGLEWGVLFGFGPGLTVETVVLHSVAT</aas><ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTATGCTACGCAT</ncds><aas>MAASITAITVENLEYPAVVTSPVTGKSYFLGGAGERGLTIEGNFIKFTAIGVYLEDIAVASLAAKWKGKSSEELLETLDFYRDIISGPFEKLIRGSKIRELSGPEYSRKVMENCVAHLKSVGTYGDAEAEAMQKFAEAFKPVNFPPGASVFYRQSPDGILGLSFSPDTSIPEKEAALIENKAVSSAVLETMIGEHAVSPDLKRCLAARLPALLNEGAFKIGN*</aas>"
]

scores = []
for idx, seq in enumerate(sequences):
    score = score_sequence(model, tokenizer, seq)
    scores.append(score)
    
df = pd.DataFrame({"sequence": sequences, "score": scores})
df['rank'] = df['score'].rank(ascending=False)
print(df)
```

results:
```
                                            sequence     score  rank
0  <ncds>GAACTGGTTAAGGTGATGGACAACGGGTGGTAATATGCTA... -1.092542   3.0
1  <ncds>GAACTGGTTAAGGTGATGGACAACGGGTGGTAATATGCTA... -1.091994   2.0
2  <ncds>GAACTGGTTAAGGTGATGGACAACGGGTGGTAATATGCTA... -1.094170   6.0
3  <ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTA... -1.092601   4.0
4  <ncds>ATGCACTGTTACCCTTTGTAATATTGTTCAGTATGGCCTA... -1.084262   1.0
5  <ncds>GAACTGGTTAAGGTGATGGACAACGGGTGGTAATATGCTA... -1.093137   5.0
```

### 3. Perplexity-odds-based Mutant Scoring

- Score DMS of coding DNA sequence (CDS)
```shell
python score_dms.py --fasta example_data/CDS_BLAT_ECOLX_Firnberg_2014.fasta \
--dms example_data/CDS_BLAT_ECOLX_Firnberg_2014.csv \
--tag cds \
--save example_data/CDS_BLAT_ECOLX_Firnberg_2014_scored.csv
```

- Score DMS of protein sequence
```shell
python score_dms.py --fasta example_data/AAS_BLAT_ECOLX_Firnberg_2014.fasta \
--dms example_data/AAS_BLAT_ECOLX_Firnberg_2014.csv \
--tag aas \
--save example_data/AAS_BLAT_ECOLX_Firnberg_2014_scored.csv
```


- Score DMS of non-coding DNA sequence (NCDS)
```shell
python score_dms.py --fasta example_data/NCDS_zhang_2009.fasta \
--dms example_data/NCDS_zhang_2009.csv \
--tag ncds \
--save example_data/NCDS_zhang_2009_scored.csv
```


### 4. Fine-tuning on downstream tasks
See in ./fintunes

## ProDMM-Seq2Seq
### 1. Reverse Translation
