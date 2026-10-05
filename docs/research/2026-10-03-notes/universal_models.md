# Open universal / generative speech restoration and voice isolation models (2023–2026) as podcast-restoration candidates

Context: earshot has already measured LavaSR, DeepFilterNet3, UniverSR, NovaSR, UniPASE and AudioSR. UniPASE currently leads on damage recovered, but it harms clean audio and moves speaker identity. These notes look for anything that could beat it. A later scope addition also covers removing everything that is not voice (noise, hum, music beds, room sound) while keeping one or more speakers.

Licence caveat that applies throughout: an arXiv HTML page saying "License: CC BY 4.0" refers to the **paper**, not to the code or weights. Where only the paper licence was found, these notes say so. Weights licences were read from the HF model cards or the repos where possible.

Research date: 2026-10-03. I made about 35 tool calls. Several facts could not be verified and are listed under Gaps.

## Q1. Catalogue: what each model does, weights, licence, sample rate, compute, runnability, evidence

### Takeaway
Few models are open, permissively licensed, full-band (44.1/48 kHz) and multi-distortion all at once. Candidates that pass those filters:
- **Sidon**: MIT, 48 kHz, reports speaker similarity, very fast.
- **UniPASE**: MIT code; already measured.
- **VoiceFixer**: MIT, 44.1 kHz, old.
- **Resemble Enhance**: MIT, 44.1 kHz.
- **VoiceRestore**: MIT; sample rate unverified.
- **AnyEnhance**: Amphion is MIT, 44.1 kHz, but the full-model weights could not be confirmed.

Strong but non-commercial or closed:
- **NVIDIA RE-USE**: open weights, but non-commercial licence. Discriminative, 9.6M parameters, 8–48 kHz.
- **Miipher-2**: no release.
- **MaskSR/MaskSR2** and **Genhancer**: no code found.

The language-model-based models (**GenSE**, **LLaSE-G1**, **SenSE**) are all 16 kHz only.

### Cited Findings

**Sidon (U. Tokyo / SaruLab, Sep 2025; arXiv 2509.17052)**
- What it is: an open-source re-creation of Miipher-2.
  - Feature predictor: w2v-BERT 2.0 (600M), 8th-layer features, LoRA fine-tuning.
  - Vocoder: HiFi-GAN with snake activation.
  - Total: 250M parameters (198M predictor + 52.4M vocoder). Output 48 kHz; Miipher-2 outputs 24 kHz. — [arXiv 2509.17052](https://arxiv.org/html/2509.17052v1)
- Training data: 2,219 h of public data in 104 languages (LibriTTS-R, FLEURS-R, EARS, HiFi-CAPTAIN, Bible-TTS and others). — [arXiv](https://arxiv.org/html/2509.17052v1)
- Speed: about 3,390× faster than realtime at batch 8 on an NVIDIA H200. — [arXiv](https://arxiv.org/html/2509.17052v1)
- LibriTTS test-other, Sidon vs Miipher:

  | Model | WER | SpkSim | NISQA | DNSMOS |
  |---|---|---|---|---|
  | Sidon | 0.095 | 0.961 | 4.698 | 3.219 |
  | Miipher | 0.090 | 0.930 | 4.597 | 3.040 |
  | Noisy input | 0.079 | — | — | — |

  Sidon's WER is slightly worse than the noisy input. — [arXiv](https://arxiv.org/html/2509.17052v1)
- FLEURS, 100 languages, Sidon vs Miipher-2:

  | Model | CER | DNSMOS | NISQA | SpkSim |
  |---|---|---|---|---|
  | Sidon | 0.090 | 3.393 | 4.420 | 0.979 |
  | Miipher-2 | 0.094 | 3.352 | 4.475 | 0.979 |
  | Noisy input | 0.084 | — | — | — |

  — [arXiv](https://arxiv.org/html/2509.17052v1)
- The paper does not discuss hallucination, speaker drift or failure cases. — [arXiv](https://arxiv.org/html/2509.17052v1)
- Licence and runnability:
  - The GitHub repo is MIT and has 24 kHz and 48 kHz preprocessing configs.
  - `infer.py` handles a folder of wavs, or a video (it replaces the audio track).
  - 26 commits, 2 open issues. — [GitHub sarulab-speech/Sidon](https://github.com/sarulab-speech/Sidon/)
- HF model `sarulab-speech/sidon-v0.1` is licensed **MIT** and is based on w2v-BERT-2.0. It was updated 15 Dec 2025, and `sidon_raw_weight` 13 Dec 2025. — [HF sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1); [HF sarulab-speech](https://huggingface.co/sarulab-speech)

**Miipher (Google, 2023) / Miipher-2 (Google, May 2025; arXiv 2505.04457)**
- Miipher-2 design:
  - A frozen USM (300+ languages) as feature extractor, with parallel adapters.
  - WaveFit vocoder.
  - Trained on about 3,000 h of studio-quality multilingual data.
  - RTF 0.0078 on consumer-grade accelerators. — [Miipher-2 summary](https://www.themoonlight.io/en/review/miipher-2-a-universal-speech-restoration-model-for-million-hour-scale-data-restoration); [arXiv 2505.04457](https://arxiv.org/pdf/2505.04457)
- The Miipher-2 demo page links only the paper and samples. No code, weights or API release is stated. — [Miipher-2 demo page](https://google.github.io/df-conformer/miipher2/)
- Unofficial Miipher by Wataru Nakata:
  - Uses WavLM-large instead of w2v-BERT XL, a Conformer instead of DF-Conformer, and HiFi-GAN instead of WaveFit.
  - Weights trained on LibriTTS-R and JVS, licensed **CC-BY-NC-2.0**. — [GitHub Wataru-Nakata/miipher](https://github.com/Wataru-Nakata/miipher)

**UniPASE (Nanjing U., Apr 2026; arXiv 2604.14606, IEEE TASLP)** — already measured in earshot
- Design:
  - DeWavLM-Omni: a WavLM fine-tuned by distillation.
  - An acoustic enhancement stage and a vocoder.
  - A PostNet to 48 kHz, then resampling back to the input rate (8–48 kHz). — [arXiv 2604.14606](https://arxiv.org/html/2604.14606)
- Repo licence MIT. Five checkpoints on HF (`Xiaobin-Rong/unipase`). — [GitHub xiaobin-rong/unipase](https://github.com/xiaobin-rong/unipase)
- The paper reports these baselines (useful as third-party numbers on other models):

  DNS 2020 no-reverb:

  | Model | SpkSim | UTMOS | DNSMOS | dWER |
  |---|---|---|---|---|
  | TF-GridNet | 0.94 | 3.86 | 3.34 | — |
  | StoRM | 0.93 | 3.73 | 3.31 | — |
  | LLaSE-G1 | 0.77 | 3.84 | 3.42 | 12.15% |
  | AnyEnhance | 0.95 | 3.96 | 3.42 | — |
  | PASE | 0.94 | 3.95 | 3.39 | 2.71% |

  URGENT 2025 non-blind test:

  | Model | SpkSim | UTMOS | CER |
  |---|---|---|---|
  | BSRNN-FAN (URGENT 2025 #1) | 0.85 | 2.40 | 11.08% |
  | TS-URGENet (#2) | 0.83 | 2.31 | 12.06% |

  — [arXiv 2604.14606](https://arxiv.org/html/2604.14606)
- The authors say UniPASE plus TF-GridNet won first place in the objective evaluation of URGENT 2026. — [arXiv](https://arxiv.org/html/2604.14606)
- That claim needs reading against the official ranking (see the URGENT section below). The instrumental #1 entry, "WR*", was disqualified for multiple registrations. I could not verify whether WR* is the UniPASE team.

**NVIDIA RE-USE (Mar 2026; arXiv 2603.02641, Fu et al.)** — new candidate, not on the original list
- Model:
  - Discriminative: convolutional encoder/decoder plus 30-layer bidirectional Mamba, **9.6M parameters**.
  - Mono input at 8, 16, 22.05, 24, 32, 44.1 or 48 kHz.
  - Handles noise, reverb, clipping, bandwidth limitation, codec, packet loss, and low-quality microphone response.
  - Ships `inference_chunk.sh` for long files. — [HF nvidia/RE-USE](https://huggingface.co/nvidia/RE-USE)
- Licence: **NVIDIA One-Way Noncommercial License (NSCLv1)**, "for research and development only". Released 18 Mar 2026. The released checkpoint differs from the paper's: it was trained on extra degradation types. — [HF nvidia/RE-USE](https://huggingface.co/nvidia/RE-USE)
- Paper findings:
  - Time-shifted anechoic clean speech is a better dereverberation target than early-reflection speech.
  - Large uncurated corpora put a ceiling on performance.
  - Two-stage handling of the distortion/perception tradeoff.
  - Claims state of the art on the URGENT 2025 non-blind test. — [arXiv 2603.02641](https://arxiv.org/abs/2603.02641)

**VoiceFixer (2021–22; still widely used)**
- Restores noise, reverb, low resolution (2–44.1 kHz) and clipping.
- MIT, PyPI package, output 44.1 kHz, runs on CPU or GPU. Modes 0, 1 and 2; mode 2 is for severely degraded speech. — [GitHub haoheliu/voicefixer](https://github.com/haoheliu/voicefixer)
- Sidon's TTS-data experiment rated VoiceFixer-cleaned TED-LIUM well below Sidon:

  | Training data | MOS |
  |---|---|
  | Original | 3.254 |
  | VoiceFixer-cleaned | 3.771 |
  | Sidon-cleaned | 4.248 |

  — [arXiv 2509.17052](https://arxiv.org/html/2509.17052v1)

**Resemble Enhance (Resemble AI, late 2023)**
- Two modules: a denoiser (separates speech from noise) and an enhancer (restores distortions, extends bandwidth). Trained on 44.1 kHz speech.
- MIT. 13 commits, 56 open issues. — [GitHub resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance)
- The HF model card is also MIT. It demonstrates background-music removal and street-noise reduction. — [HF ResembleAI/resemble-enhance](https://huggingface.co/ResembleAI/resemble-enhance)

**VoiceRestore (Kirdey, 2024–25; arXiv 2501.00794)**
- Flow-matching transformer, about 301M parameters, with BigVGAN as the vocoder.
- MIT. v1.1 checkpoint released 16 Jan 2025 (Google Drive, mirrored on HF). 3 open issues.
- Says it "may not perform optimally" on extreme degradations. — [GitHub skirdey/voicerestore](https://github.com/skirdey/voicerestore)
- The HF card says "still in the process of training" and gives no sample rate or metrics. — [HF jadechoghari/VoiceRestore](https://huggingface.co/jadechoghari/VoiceRestore)

**AnyEnhance (Amphion / CUHK-SZ, Jan 2025; arXiv 2501.15417)**
- Model:
  - Masked generative model, **44.1 kHz**, 363.5M parameters.
  - Tasks: denoising, dereverb, declipping, super-resolution, and target speaker extraction, for both speech and singing.
  - Optional prompt-guidance and self-critic sampling. — [arXiv 2501.15417](https://arxiv.org/html/2501.15417v2)
- Librivox GSR, AnyEnhance vs MaskSR:

  | Model | DNSMOS OVRL | SpkSim |
  |---|---|---|
  | AnyEnhance | 3.308 | 0.955 |
  | MaskSR | 3.258 | 0.94 |

  — [arXiv](https://arxiv.org/html/2501.15417v2)
- Amphion's code is MIT, "free for both research and commercial use". There was a CI run named "Release AnyEnhance pretrained models". — [Amphion](https://github.com/open-mmlab/amphion); [Amphion CI run](https://github.com/open-mmlab/Amphion/actions/runs/12783274325)
- Fetching `huggingface.co/amphion/anyenhance` returned HTTP 401. The full model's weights and their licence are **unverified**.
- A lightweight 45.7M-parameter AnyEnhance-v1 (MIT) has weights. It is trained only on 200 h of CCF-AATC 2025 challenge data, without prompt-guidance or self-critic. — [GitHub viewfinder-annn/AnyEnhance-v1](https://github.com/viewfinder-annn/AnyEnhance-v1)

**MaskSR (Interspeech 2024) / MaskSR2 (Dolby)**
- MaskSR: masked token prediction on frozen 44.1 kHz DAC tokens (9 codebooks). Handles noise, reverb, clipping and low bandwidth jointly. Reports better speaker similarity than multi-stage and regression baselines. — [arXiv 2406.02092](https://arxiv.org/abs/2406.02092)
- MaskSR2 adds semantic knowledge distillation from HuBERT to improve intelligibility. — [ResearchGate MaskSR2](https://www.researchgate.net/publication/384075670_Joint_Semantic_Knowledge_Distillation_and_Masked_Acoustic_Modeling_for_Full-band_Speech_Restoration_with_Improved_Intelligibility)
- No code or weights found (see Gaps).

**GenSE (ICLR 2025)**
- Two-stage language model: noisy → semantic tokens (N2S), then semantic → acoustic tokens (S2S). Uses the single-quantizer SimCodec and a 24-layer Llama-style LM. — [arXiv 2502.02942](https://arxiv.org/pdf/2502.02942)
- Weights on HF (`yaoxunji/gen-se`). Output **16 kHz**. No licence stated in the repo. 10 commits, 6 open issues. — [GitHub yaoxunji/gen-se](https://github.com/yaoxunji/gen-se)

**LLaSE-G1 (NWPU ASLP, Mar 2025; arXiv 2503.00493)**
- Model:
  - WavLM features in, LLaMA (about 1.07B parameters, 16 layers) in the middle, X-Codec2 tokens out.
  - **16 kHz only**; the authors defer full-band to future work.
  - Tasks: noise suppression, packet loss concealment, target speaker extraction, echo cancellation, and speech separation (unseen in training). — [arXiv 2503.00493](https://arxiv.org/html/2503.00493v1)
- DNS no-reverb DNSMOS OVRL: LLaSE-G1 3.49, GenSE 3.43, MaskSR 3.34, AnyEnhance 3.42. Self-reported SimWB 0.993. — [arXiv](https://arxiv.org/html/2503.00493v1)
- Licence Apache-2.0. Code and weights on GitHub and HF. The card warns task inference "may exhibit instability in certain scenarios". — [HF ASLP-lab/LLaSE-G1](https://huggingface.co/ASLP-lab/LLaSE-G1); [GitHub Kevin-naticl/LLaSE-G1](https://github.com/Kevin-naticl/LLaSE-G1)

**SenSE (NWPU ASLP, ICME 2026; arXiv 2509.24708)**
- Semantic-aware two-stage model: LM semantic tokens, then flow-matching enhancement, with optional reference speech. — [arXiv 2509.24708](https://arxiv.org/abs/2509.24708)
- Repo MIT, weights on HF (`ASLP-lab/SenSE`), inference at **16 kHz**. — [GitHub ASLP-lab/SenSE](https://github.com/ASLP-lab/SenSE)

**FlowSE: two different models share the name**
- Lee et al., ICASSP 2025: flow matching SE; official code at `seongq/flowmse`. It is one of the URGENT 2026 Track 1 baselines. — [GitHub seongq/flowmse](https://github.com/seongq/flowmse); [URGENT 2026 paper](https://arxiv.org/pdf/2601.13531)
- Wang et al., Interspeech 2025:
  - Mel-spectrogram flow matching, optionally conditioned on a transcript.
  - Weights released (`wenetspeech4tts_Premium.pt.tar`). Repo licence not stated.
  - Claims gains in quality, intelligibility and speaker similarity. — [GitHub Honee-W/FlowSE](https://github.com/Honee-W/FlowSE); [ISCA](https://www.isca-archive.org/interspeech_2025/wang25s_interspeech.pdf)

**UniFlow (NWPU, Aug 2025; arXiv 2508.07558)**
- Waveform VAE latent plus DiT. Covers SE, TSE, echo cancellation and language-queried source separation, and compares diffusion, flow matching and mean flow.
- Authors "will open-source"; no code found. — [arXiv 2508.07558](https://arxiv.org/abs/2508.07558)

**Genhancer (UIUC/Amazon, Interspeech 2024)**
- Generates DAC-style codec tokens conditioned on the noisy input. Claims better speaker-identity retention than baselines. — [ISCA](https://www.isca-archive.org/interspeech_2024/yang24h_interspeech.html)
- The GitHub repo holds audio samples only, with no code or weights. — [GitHub haiciyang/Genhancer](https://github.com/haiciyang/Genhancer)

**SpeechFlow (Meta, 2023; arXiv 2310.16338)**
- Generative flow-matching model pre-trained on 60k h of untranscribed speech, fine-tuned for SE, separation and TTS. — [arXiv 2310.16338](https://arxiv.org/abs/2310.16338)
- No public weights found.

**Other 2025–2026 generative restorers found** (details in Q3)
- VoiceBridge
- Stream.FM
- ParaGSE
- DiTSE

**URGENT challenge top systems**
- **URGENT 2024** (NeurIPS 2024 competition): 4 distortions (noise, reverb, bandwidth limit, clipping) at variable sample rates. — [URGENT 2024](https://urgent-challenge.github.io/urgent2024/)
- A multistage system "ranked 1st in the URGENT 2024 challenge with a MOS of 3.52" and placed 4th in Track 2 of URGENT 2025. — [Le et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/le25b_interspeech.pdf) (via search snippet; code not checked)
- **URGENT 2025 (Interspeech 2025) setup**:
  - 7 distortions: noise, reverb, clipping, bandwidth limit, codec (MP3/OGG; neural codecs unseen in the blind set), packet loss, wind.
  - Up to 5 distortions at once; inputs at 8–48 kHz.
  - 32 submissions across two tracks: 2.5k h and 60k h of training data.
  - Baseline: TF-GridNet, 8.5M parameters. — [Saijo et al. 2025](https://www.isca-archive.org/interspeech_2025/saijo25_interspeech.pdf)
- **URGENT 2025 Track 1 results**:

  | System | Type | Params | PESQ | SpkSim | CAcc | MOS |
  |---|---|---|---|---|---|---|
  | T1 (winner) | discriminative sub-band RNN | ~102M | 2.64 | 0.76 | 79.8% | 3.24 |
  | T3 (D+G cascade, discrete tokens) | hybrid | — | — | 0.71 | — | 3.44 |
  | T13 (latent diffusion + vocoder) | pure generative | — | — | 0.47 (last) | 67.87% | 3.69 (highest) |
  | Noisy input | — | — | — | 0.55 | — | 2.13 |

  — [Saijo et al.](https://www.isca-archive.org/interspeech_2025/saijo25_interspeech.pdf)
- No Track 2 system (60k h) beat the best Track 1 system. — [Saijo et al.](https://www.isca-archive.org/interspeech_2025/saijo25_interspeech.pdf)
- The Track 1 winner (Tencent AI Lab / CQUPT, Sun et al.):
  - A dual-path time-frequency model with fast band split/merge, a Fourier-Analysis-Network channel mixer, and a sampling-frequency-independent design. Discriminative. — [Sun et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/sun25d_interspeech.pdf)
  - UniPASE calls it "BSRNN-FAN". — [arXiv 2604.14606](https://arxiv.org/html/2604.14606)
  - No code or weights release was found in the paper text I read.
- **URGENT 2026 (ICASSP 2026) setup**:
  - Same 7 distortions. Adds emotional, child, elderly, whispered and singing speech, and 5 unseen languages.
  - Emphasis on data curation: the baseline curates 700 h out of 2,500 h.
  - Baselines: BSRNN and FlowSE, code at `urgent-challenge/urgent2026_challenge_track1`.
  - Leading systems "dominantly followed a hybrid generative and discriminative paradigm". — [ICASSP 2026 URGENT](https://arxiv.org/pdf/2601.13531)
- **URGENT 2026 final subjective ranking**:
  1. baird
  2. subatomicseer
  3. Ali-Universal-SE
  4. RICK2000

  Instrumental ranks 1 and 2, "WR*" and "GHW*", were disqualified for multiple registrations. — [Final ranking PDF](https://urgent-challenge.github.io/urgent2026/assets/files/FinalRankingDetails_URGENT2026.pdf)
- A hybrid TF-GridNet + autoregressive system with a learned fusion network reports third place in URGENT 2026 (Liu et al.). Its code release is not stated. — [arXiv 2601.19113](https://arxiv.org/abs/2601.19113)

### Inferences
- **Sidon** is the most direct new candidate against UniPASE:
  - MIT weights, 48 kHz.
  - Speaker similarity reported and higher than Miipher (0.961 vs 0.930).
  - Its design aims at dataset cleaning, so its default behaviour is resynthesis, not passthrough.
  - Expect it to replace the voice with a vocoder rendering of SSL features, like Miipher. That risks the same "clean audio harmed" behaviour earshot saw with UniPASE. The `origin` probe should test this.
- **RE-USE** is the strongest-looking discriminative alternative:
  - Small (9.6M), all URGENT 2025 distortion types, native 8–48 kHz.
  - Discriminative models kept speaker similarity best in URGENT 2025, so it may protect clean microphones better.
  - Its licence is non-commercial. It is fine for the bench, but not usable for commercial podcast production without NVIDIA permission.
- The 16 kHz language-model models (GenSE, LLaSE-G1, SenSE) are a poor fit for a 48 kHz podcast pipeline. LLaSE-G1 also had SpkSim 0.77 and 12% dWER in UniPASE's third-party test.
- The 2024–2026 challenge pattern is consistent:
  - Discriminative and hybrid models win objective and speaker metrics.
  - Pure generative models win naive MOS but hallucinate.
  - A hybrid (discriminative front-end, generative refinement) is the established route to beating a single generative model.

### Gaps
- **AnyEnhance**: whether the full 363M-parameter weights are public, and under what licence. The HF URL returned 401.
- **VoiceRestore**: output sample rate not documented. The BigVGAN variant suggests 24 kHz, but that is unverified.
- **MaskSR/MaskSR2**: no code or weights found. I did not search exhaustively; Dolby has not, to my knowledge, released them.
- **SpeechFlow**: no public weights found.
- **URGENT 2025 Track 1 winner**: code release not confirmed.
- **URGENT 2026 winners** (baird, subatomicseer, Ali-Universal-SE): the system descriptions and any code release were not found.
- **Licence status unverified for**: GenSE (none stated), FlowSE Honee-W (none stated), LLaSE-G1's X-Codec2 dependency, and the HF model cards of the UniPASE weights.
- **Compute**: RTF and GPU memory figures are missing for most models. Only Sidon (3,390× realtime on H200) and Miipher-2 (RTF 0.0078) state them.

## Q2. Which preserve speaker identity or report speaker similarity, and which hallucinate

### Takeaway
- Most papers report speaker similarity, but on their own test sets. Third-party comparisons are rarer and less flattering.
- In URGENT 2025, the purely generative system had the worst speaker similarity (0.47, below the noisy input's 0.55) and hallucinated content. Discriminative systems kept identity best.
- LLaSE-G1 lost identity badly in UniPASE's independent test (SpkSim 0.77).
- Sidon, AnyEnhance and Miipher-2 report SpkSim of 0.95 or more on their own benchmarks.

### Cited Findings
- **URGENT 2025** defines hallucination as "discrepancies in spoken content or speaker characteristics" between input and output. — [Saijo et al.](https://www.isca-archive.org/interspeech_2025/saijo25_interspeech.pdf)
- **T13 (pure generative)**:
  - "Occasionally hallucinated spoken content, particularly under low-SNR conditions".
  - On unseen Japanese, the output "sometimes resembled English or other European languages". The same happened in frames inpainted after packet loss.
  - DNSMOS stays high even when hallucinating. — [Saijo et al.](https://www.isca-archive.org/interspeech_2025/saijo25_interspeech.pdf)
- **T13 numbers**:

  | Metric | T13 | Noisy input |
  |---|---|---|
  | SpkSim | 0.47 | 0.55 |
  | Chinese CAcc | 20.1% | 69.3% |
  | Japanese CAcc | 36.8% | 74.5% |
  | Subjective MOS | 3.69 (highest of all) | — |

  — [Saijo et al.](https://www.isca-archive.org/interspeech_2025/saijo25_interspeech.pdf)
- Subjective P.808 MOS "would be difficult to penalize the correctness of the spoken content and speaker consistency as long as the speech sounds natural". — [Saijo et al.](https://www.isca-archive.org/interspeech_2025/saijo25_interspeech.pdf)
- Hybrid and generative models outperformed discriminative ones in MOS, but were "more prone to hallucinations". — [P.808 multilingual results, arXiv 2507.11306](https://arxiv.org/pdf/2507.11306) (via search summary)
- **Self-reported speaker similarity**:

  | Model | SpkSim | Test set | Source |
  |---|---|---|---|
  | Sidon | 0.961 | LibriTTS test-other | [arXiv 2509.17052](https://arxiv.org/html/2509.17052v1) |
  | Miipher | 0.930 | LibriTTS test-other | [arXiv 2509.17052](https://arxiv.org/html/2509.17052v1) |
  | Sidon / Miipher-2 | 0.979 | FLEURS | [arXiv 2509.17052](https://arxiv.org/html/2509.17052v1) |
  | AnyEnhance | 0.955 | — | [arXiv 2501.15417](https://arxiv.org/html/2501.15417v2) |
  | MaskSR | 0.94 | — | [arXiv 2501.15417](https://arxiv.org/html/2501.15417v2) |
  | LLaSE-G1 | 0.993 (SimWB) | own test | [arXiv 2503.00493](https://arxiv.org/html/2503.00493v1) |
  | GenSE | 0.974 | own test | [arXiv 2503.00493](https://arxiv.org/html/2503.00493v1) |

- **Third-party speaker similarity, from the UniPASE paper**:
  - DNS 2020: AnyEnhance 0.95, PASE 0.94, TF-GridNet 0.94, LLaSE-G1 0.77 (dWER 12.15%).
  - PLC 2024: LLaSE-G1 0.73 (WER 31.46%). — [arXiv 2604.14606](https://arxiv.org/html/2604.14606)
- MaskSR and Genhancer both claim improved speaker similarity over their baselines. — [arXiv 2406.02092](https://arxiv.org/abs/2406.02092); [Genhancer ISCA](https://www.isca-archive.org/interspeech_2024/yang24h_interspeech.html)
- The Sidon paper does not discuss hallucination or speaker drift. — [arXiv](https://arxiv.org/html/2509.17052v1)
- The LLaSE-G1 card warns of "instability in certain scenarios". — [HF](https://huggingface.co/ASLP-lab/LLaSE-G1)

### Inferences
- The SpkSim gap between LLaSE-G1's own figure (0.993) and UniPASE's measurement of it (0.77) shows that self-reported SpkSim cannot be compared across papers. Different embedders and test sets give different numbers. earshot's own `origin` probe is the only comparable measure.
- Resynthesis models that go through SSL features and a vocoder (Miipher, Sidon, UniPASE, GenSE, LLaSE-G1) are structurally prone to speaker drift and language-dependent content hallucination.
  - Mask- or mapping-based discriminative models (RE-USE, the BSRNN-FAN type, TF-GridNet, DeepFilterNet) cannot invent phonemes.
  - Given the project's "human soundingness" goal, a discriminative-first hybrid is the safer architecture.

### Gaps
- No independent measurement found of Sidon's, RE-USE's, Resemble Enhance's or VoiceRestore's speaker similarity on VoIP-type damage.
- No published hallucination study found for Sidon or Resemble Enhance.

## Q3. Newest (2025–2026) releases not on the original list

### Takeaway
New since the original list:
- **NVIDIA RE-USE**: Mar 2026, weights, non-commercial.
- **Sidon**: Sep 2025, MIT; on the list, but newly practical.
- **SenSE**: 16 kHz, MIT.
- **VoiceBridge**: 48 kHz one-step latent bridge; code not found.
- **Stream.FM**: realtime streaming generative restoration; code not found.
- **ParaGSE**: 2026.
- **URGENT 2026 hybrid systems**.
- **DialogueSidon**: two-speaker dialogue separation, non-commercial.

### Cited Findings
- **VoiceBridge** (arXiv 2509.25275):
  - One-step general speech restoration via a latent bridge model, without distillation.
  - Reconstructs **48 kHz** full-band speech and covers 48 restoration tasks.
  - Only a demo link was found (VoiceBridgedemo.github.io); no code or weights. — [arXiv 2509.25275](https://arxiv.org/pdf/2509.25275)
- **Stream.FM** (Welker, Lay, Hillemann, Peer, Gerkmann; arXiv 2512.19442):
  - Frame-causal flow matching for enhancement, dereverb, codec post-filtering, bandwidth extension and vocoding.
  - 32 ms algorithmic / 48 ms total latency (24 ms for the SE variant), on consumer GPUs.
  - Code not stated. — [arXiv 2512.19442](https://arxiv.org/abs/2512.19442)
- **SenSE** (ICME 2026): MIT, HF weights, 16 kHz. — [GitHub](https://github.com/ASLP-lab/SenSE)
- **RE-USE** (NVIDIA, 18 Mar 2026): see Q1. — [HF](https://huggingface.co/nvidia/RE-USE)
- **ParaGSE** (arXiv 2602.01793): parallel generative SE with a group-VQ neural codec. Only the title and abstract were seen; details not verified. — [arXiv 2602.01793](https://arxiv.org/html/2602.01793)
- **DiTSE** (arXiv 2504.09381): high-fidelity generative SE with latent diffusion transformers. Release status not checked. — [arXiv 2504.09381](https://arxiv.org/pdf/2504.09381)
- **UniverSR** (arXiv 2510.00771): vocoder-free flow-matching super-resolution. Already measured in earshot. — [arXiv](https://arxiv.org/html/2510.00771v1)
- **URGENT 2026** third-place hybrid: TF-GridNet with a sampling-frequency-independent design, plus an autoregressive branch and a learned fusion network. — [arXiv 2601.19113](https://arxiv.org/abs/2601.19113)
- **ReverbMiipher** (Google, 2025): restoration with controllable reverberation. Google; no release found. — [arXiv 2505.05077](https://arxiv.org/pdf/2505.05077)
- **FlowSE-NFT** and **GA-AF-FlowSE** (2025–26): RL and multi-reward post-training for FlowSE, with code repos. — [FlowSE-NFT](https://github.com/doveru/FlowSE-NFT); [GA-AF-FlowSE](https://github.com/spzhang7/GA-AF-FlowSE)

### Inferences
- VoiceBridge (48 kHz, one-step) and Stream.FM (streaming) are the most interesting architecturally. Neither appears runnable today; watch for code releases.
- RE-USE plus a light generative bandwidth or PLC stage would replicate the URGENT-winning hybrid recipe from open parts.

### Gaps
- ParaGSE, DiTSE and Stream.FM code and licence status not verified.
- "Grounded Decoding for Autoregressive SE" (arXiv 2609.04245) and "Rethinking LM-based generative SE in codec latent space" (arXiv 2608.12082) were seen only as titles. — [2609.04245](https://arxiv.org/pdf/2609.04245); [2608.12082](https://arxiv.org/html/2608.12082)

## Q4 (added scope). Open voice/dialogue isolation: removing everything that is not human voice while keeping one or more speakers

### Takeaway
No single open model is an established equivalent of iZotope Dialogue Isolate or Adobe Enhance. The practical open options:
- **SAM-Audio** (Meta, Dec 2025):
  - Text-prompted ("speech") separation that returns both a target and a residual.
  - Licence permits commercial use. 500M, 1B and 3B parameter sizes; gated weights.
  - Generative, so it is a speaker-identity and hallucination risk.
- **BandIt v2** cinematic dialogue/music/effects separation: CC-BY-SA weights.
- Music-source-separation **RoFormer** vocal and denoise models: community weights, licences unclear.
- **ClearerVoice-Studio** (Apache-2.0 code): MossFormer2 48 kHz SE, separation and target speaker extraction.
- **Resemble Enhance's denoiser**: MIT, demonstrates music removal.
- **DialogueSidon**: separates two-speaker dialogue, but is non-commercial and 24 kHz.

### Cited Findings
- **SAM-Audio** (Meta, arXiv 2512.18099):
  - Foundation separation model prompted by text, visuals or time span.
  - Flow-matching DiT in a DAC-VAE latent space (25 Hz, 128 channels), 16-step midpoint ODE solver.
  - Sizes 500M, 1B and 3B. Jointly outputs the target stem and a residual stem.
  - Covers speech, music and sound effects. Span prompting is ambiguous when events overlap. — [arXiv 2512.18099](https://arxiv.org/html/2512.18099v1)
- **SAM-Audio access**:
  - HF weights are gated (contact sharing). Uses a PE-AV encoder and an optional "Judge" re-ranking model.
  - Sample rate and parameter-to-size mapping are not on the card. — [HF facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large)
- **SAM License**:
  - Non-exclusive, worldwide, royalty-free; **commercial use permitted**.
  - Prohibits military, nuclear, espionage and ITAR uses, and reverse engineering.
  - Requires acknowledgement in publications. Terminates on litigation against Meta. — [SAM-Audio LICENSE](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE)
- **ClearerVoice-Studio** (Alibaba Tongyi):
  - Code Apache-2.0, `pip install clearvoice`.
  - SE: FRCRN at 16 kHz; MossFormer2_SE_48K at 48 kHz. Separation: MossFormer at 8/16 kHz. Also 48 kHz speech super-resolution and target speaker extraction (audio-only and audio-visual).
  - Weight licences are not separately stated. — [GitHub modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio); [PyPI clearvoice](https://pypi.org/project/clearvoice/)
- **BandIt** (cinematic audio source separation into dialogue, music and effects):
  - Code Apache-2.0.
  - v1 weights CC-BY-NC-4.0 (Zenodo 10160698); v2 weights **CC-BY-SA-4.0** (Zenodo 12701995).
  - An inference-only package `openmirlab/bandit-infer` exists. — [GitHub kwatcharasupat/bandit](https://github.com/kwatcharasupat/bandit); [bandit-infer](https://github.com/openmirlab/bandit-infer); [arXiv 2309.02539](https://arxiv.org/html/2309.02539v3)
- **RoFormer models**:
  - BS-RoFormer and Mel-Band RoFormer are music-source-separation models. Community checkpoints exist for vocals, denoise, dereverb and crowd (for example the "Aufr33 Mel Roformer Denoise" model). — [melband-roformer-infer](https://github.com/openmirlab/melband-roformer-infer); [Mel-RoFormer arXiv 2409.04702](https://arxiv.org/pdf/2409.04702)
  - The BS-RoFormer vocals checkpoint (viperx) reports 12.97 SDR for vocals vs instrumental. — [HF AEmotionStudio/roformer-models](https://huggingface.co/AEmotionStudio/roformer-models)
  - These licences were not established.
- **Demucs**: has been used to remove background music from podcast and YouTube data. — [Picovoice blog](https://picovoice.ai/blog/voice-isolator/) (via search snippet, secondary)
- **Meta `denoiser`** (Demucs-based speech SE): CC-BY-NC 4.0, 16 kHz, archived 31 Oct 2023. — [GitHub facebookresearch/denoiser](https://github.com/facebookresearch/denoiser)
- **DialogueSidon** (SaruLab, 2026):
  - Two-speaker dialogue separation: diffusion in a VAE-32 latent with an SSL encoder.
  - Output **24 kHz**, licence **CC-BY-NC-4.0**. — [HF sarulab-speech/DialogueSidon](https://huggingface.co/sarulab-speech/DialogueSidon)
- **Resemble Enhance**: its denoiser "separates speech from a noisy audio"; the model card shows background-music removal examples. MIT. — [HF](https://huggingface.co/ResembleAI/resemble-enhance)
- **DeepFilterNet on non-stationary noise**: only a vendor blog was found. It says DFN struggled with segments under 150 ms and with sudden noises (door slams, overlapping voices), leaving residual noise. This is weak, secondary evidence. — [noisereducerai blog](https://noisereducerai.com/blogs/deepfilternet-ai-noise-reduction/)
- **Universal SE models as isolators**: URGENT training includes music noise (Free Music Archive) and wind as additive noise. — [Saijo et al.](https://www.isca-archive.org/interspeech_2025/saijo25_interspeech.pdf)
  - This means URGENT-style models (RE-USE, UniPASE, the TF-GridNet baseline) are trained to remove music beds as "noise".

### Inferences
- For "remove everything but voice, keep all speakers":
  - Separation models (SAM-Audio with the prompt "speech", BandIt's dialogue stem, RoFormer vocals) keep every voice and return a residual. That fits multi-speaker podcasts better than target-speaker extraction, which keeps only one enrolled voice.
  - SAM-Audio is a generative latent model, so the same speaker-identity and hallucination checks apply. Its residual stem makes a "sum back to the input" sanity check possible.
  - Mask-based separators (BandIt, RoFormer, MossFormer2) cannot hallucinate phonemes. They are the conservative choice for a protective local-microphone path.
- RoFormer vocal models are trained on singing and music. Speech plus room noise is out of domain for them, so measure before claiming anything.
- Hum and buzz are steady narrowband noise. Classic DSP (notch or comb filtering) may beat any of these models; no learned-model evidence specific to hum was found.

### Gaps
- No open model was found that is explicitly an equivalent of iZotope Dialogue Isolate or Adobe Enhance with published speech metrics. The Production Expert "Dialogue Noise Reduction Shootout 2025" covers commercial tools and was not read. — [Production Expert](https://www.production-expert.com/production-expert-1/dialogue-noise-reduction-shootout-2025)
- SAM-Audio: sample rate, RTF, and speech-specific metrics (SDR or SpkSim on speech-from-music) were not found in the material read.
- Licences of the RoFormer community weights, and of the ClearerVoice weights, were not verified.
- No primary-source evaluation of DeepFilterNet3 on music beds or intermittent noise was found.
