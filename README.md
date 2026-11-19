# Text-to-Text Translation: English, Hindi and French

A translation app built with Hugging Face Transformers and Gradio. The English→Hindi model is a MarianMT model **fine-tuned by me** on cleaned OPUS-100 data. The other directions use pretrained Marian models.

![App screenshot](screenshots/app_en_hi.png)

## Features

- Source and target language selection (English, Hindi, French)
- Text input box, Translate button, translated output, Swap button
- Hindi ↔ French through an English pivot
- Name protection: names are detected, masked, translated around, then restored using transliteration (see below)

| Direction | Model | Training |
|---|---|---|
| en → hi | `models/en-hi-final` (MarianMT) | Fine-tuned (this project) |
| hi → en | `Helsinki-NLP/opus-mt-hi-en` | Pretrained |
| en → fr | `Helsinki-NLP/opus-mt-en-fr` | Pretrained |
| fr → en | `Helsinki-NLP/opus-mt-fr-en` | Pretrained |
| hi ↔ fr | Two models chained via English | Pretrained |

## Project pipeline

1. **Data:** OPUS-100 `en-hi` (534,319 training pairs)
2. **Preprocessing:** Unicode normalisation, length and ratio filters, script checks, de-duplication, removal of religious-text and software-UI domains, test/train leakage check. Final set: 105,338 pairs, of which 100,000 were used for training.
3. **Baseline:** pretrained `opus-mt-en-hi`, no fine-tuning
4. **Fine-tuning:** `Seq2SeqTrainer`, 3 epochs, learning rate 2e-5, batch size 32, bf16, max length 96 (about 11 minutes on an RTX 5050)
5. **Evaluation:** BLEU and chrF with `sacrebleu`
6. **GUI:** Gradio app (`app.py`)

## Results (en → hi)

| Test set | Baseline BLEU / chrF | Fine-tuned BLEU / chrF |
|---|---|---|
| OPUS clean test (869 sentences) | 35.02 / 54.13 | 32.71 / 53.41 |
| OPUS raw test | 43.94 / 58.63 | 41.88 / 57.13 |
| Hand-written everyday sentences (30), punctuation-normalised | 34.68 / 58.57 | 41.98 / 63.33 |

**How to read this.** The OPUS test sets still contain religious and software text that the pretrained model has probably seen, so the baseline scores are likely inflated, and fine-tuning slightly lowers them. On everyday sentences the fine-tuned model scores higher, but the per-sentence win count is only 8 vs 6 (16 ties) on 30 sentences, so the improvement is modest and not conclusive.

### Other directions (pretrained, not fine-tuned)

| Direction | BLEU | chrF |
|---|---|---|
| hi → en | TODO | TODO |
| en → fr | TODO | TODO |
| fr → en | TODO | TODO |

## Setup

Python 3.10-3.12 recommended.

```bash
python -m venv nmt-env
nmt-env\Scripts\activate          # Windows
# source nmt-env/bin/activate     # Linux / Mac

# PyTorch with CUDA 12.8 (needed for RTX 50-series GPUs); use the CPU build if you have no GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Get the fine-tuned model

The model weights are too large for GitHub. Download them and place them in `models/en-hi-final/`:

> TODO: add your Hugging Face model link here (for example `https://huggingface.co/<username>/en-hi-marian-ft`)

Without it, the en→hi direction will not load. The other directions download automatically.

### Run the app

```bash
python app.py
```

Open http://127.0.0.1:7860.

### Optional: better name transliteration (IndicXlit)

Names not in `names.json` are transliterated by a local IndicXlit service if it is running, otherwise by an online endpoint, otherwise kept in Latin letters. To enable the local service, create a separate Python 3.10 environment and follow the comments in `requirements-xlit.txt`, then run:

```bash
python xlit_server.py
```

Keep it running while the app runs. The app works without it.

## Repository structure

```
.
├── app.py                  # Gradio GUI + name protection
├── xlit_server.py          # optional IndicXlit helper service
├── names.json              # manual name spellings (always win)
├── requirements.txt
├── requirements-xlit.txt
├── notebooks/              # data prep, training, evaluation
├── data/                   # small evaluation files only
├── screenshots/            # GUI screenshots
└── README.md
```

## Name handling

Marian models split unusual names into subword pieces and mistranslate them (for example "abhiraj" became "प्यासीज"). The app detects names with rules (phrases like "my name is", capitalised rare words, `[bracketed]` words), replaces them with common placeholder names, translates, then restores the real name. Spelling order: `names.json` → cache → IndicXlit → online → Latin.

## Limitations

- Name detection is rule-based and can miss lowercase names mid-sentence; use `[brackets]` to force it.
- Hindi output defaults to male verb forms because English does not mark gender.
- Training data (OPUS-100) is noisy; some weak translations remain.
- Only en→hi was fine-tuned. The hand-written test set has 30 sentences, and I wrote the Hindi references.
- Idioms and long sentences are often translated literally.

## Future work

Larger hand-checked test set, fine-tuning the other directions, an NER model for name detection, cleaner training data (for example LaBSE filtering).

## Acknowledgements

OPUS / OPUS-100, Helsinki-NLP MarianMT models, AI4Bharat IndicXlit, Hugging Face Transformers, Gradio.
