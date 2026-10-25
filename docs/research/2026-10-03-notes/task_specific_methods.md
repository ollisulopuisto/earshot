# Task-specific speech restoration methods and open models (2022–2026)

Scope: five gaps in a podcast restoration bench (dereverberation, declipping, packet loss concealment, codec artefacts, low-frequency bandwidth extension), plus strong general discriminative SE models. Research done 2026-10-03 with about 20 search/fetch calls. Licences were read from repo pages where a fetch succeeded. Anything not confirmed is listed under Gaps.

## 1. Dereverberation of single-channel speech in real rooms at 48 kHz

### Takeaway
SGMSE+ is the only open, MIT-licensed model found with a **48 kHz dereverberation checkpoint** (trained on EARS-Reverb). BUDDy (sp-uhh) is the most interesting option for real rooms because it fits a room model per utterance and does not rely on a training set of rooms. Its public checkpoint is VCTK-trained, though, and the sample rate and licence were not visible. WPE (nara_wpe, MIT) is the classical baseline that cannot hallucinate speech. None of these has published evidence on real podcast rooms at 48 kHz, so the bench has to supply that.

### Cited Findings
- SGMSE+ (sp-uhh/sgmse), MIT licence. Dereverberation checkpoints: "SGMSE+ trained on WSJ0-REVERB" (16 kHz) and "SGMSE+ trained on EARS-Reverb" (**48 kHz**). Enhancement checkpoints: VB-DMD and WSJ0-CHiME3 (16 kHz), EARS-WHAM (48 kHz). There is also a Schrödinger Bridge model trained on EARS-WHAM plus VB-DMD. For dereverberation the README recommends sampler settings `--N 50 --snr 0.33`, i.e. 50 reverse steps, so it is GPU-heavy. Follow-up repos: StoRM (stochastic regeneration) and SGMSE-BBED — [sp-uhh/sgmse](https://github.com/sp-uhh/sgmse)
- BUDDy: single-channel **blind unsupervised** dereverberation using diffusion posterior sampling. It models the reverberation operator as a per-subband exponential-decay filter and estimates its parameters during the reverse diffusion. It is described as the first unsupervised method to do blind dereverberation and RIR estimation together. It does as well as, or better than, some supervised baselines in mismatched acoustic conditions, which is where supervised models generalise poorly — [arXiv 2405.04272](https://arxiv.org/pdf/2405.04272); [UHH page](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/iwaenc2024-buddy.html)
- BUDDy code and a checkpoint trained on VCTK anechoic speech are public. The README excerpt shows no licence or sample rate — [sp-uhh/buddy](https://github.com/sp-uhh/buddy)
- Extensions: a journal version, "Unsupervised Blind Joint Dereverberation and Room Acoustics Estimation with Diffusion Models" ([arXiv 2408.07472](https://arxiv.org/pdf/2408.07472)), and multi-channel mBUDDy ([yoavellinson/buddy_mc](https://github.com/yoavellinson/buddy_mc))
- nara_wpe: WPE in NumPy/TensorFlow, single- and multi-channel, offline and online, MIT licence. It was evaluated on REVERB challenge real recordings, but only through ASR WER, not perceptual quality — [paper](https://groups.uni-paderborn.de/nt/pubs/2018/ITG_2018_Drude_Paper.pdf); [fgnt/nara_wpe](https://github.com/fgnt/nara_wpe)
- TF-GridNet was the official baseline of the Interspeech 2025 URGENT challenge, whose distortions include reverberation. Baseline scores: DNSMOS 2.63, UTMOS 1.42, PESQ 1.51, ESTOI 0.49, speaker similarity 0.70 — [URGENT 2025 paper](https://arxiv.org/html/2505.23212)

### Inferences
- The project's finding that DeepFilterNet3 removes speech along with the echo is what discriminative masking models are known to do. WPE subtracts a linear prediction of the late reverb, so it removes no direct-path speech. It is the safest first candidate against the `origin` probe, though it is gentle on short podcast-room reverb.
- BUDDy's model of the room (one per-subband exponential decay) is close to how a small treated or untreated room behaves. It refines the speech under a prior of anechoic speech instead of predicting a mask, so in principle it is less likely to delete speech. Two costs: it needs many diffusion steps per utterance, and with a VCTK prior it may pull voices toward VCTK speakers. The speaker probe should test this, since UniPASE already moved voices away from their owners.
- SGMSE+ EARS-Reverb is the only ready 48 kHz dereverberation checkpoint. EARS-Reverb uses real recorded RIRs convolved with anechoic speech, as far as is known (not verified here). That is closer to real rooms than fully synthetic RIRs, but it is still convolution, not a voice recorded in the room.

### Gaps
- BUDDy's licence, sample rate and per-utterance runtime were not found on the repo page; the paper and its config need checking.
- StoRM's 48 kHz checkpoints and licence were not checked directly.
- No open, DeRoom-like commercial-grade 48 kHz model for real rooms was found. "Rethinking Training Targets… for Universal Speech Enhancement" ([arXiv 2603.02641](https://arxiv.org/html/2603.02641)) appeared in search but was not read.
- No published evaluation was found of any of these on real podcast rooms (short RT60, close mic).

## 2. Declipping / overload

### Takeaway
Two kinds of option are open. The classical sparse methods (A-/S-SPADE, consistent dictionary learning) have Python and MATLAB toolboxes and work at any sample rate, but they were designed for mild to moderate clipping. The neural declippers found are all **16 kHz**. DDD (Demucs plus discriminator plus HiFi-GAN) has public weights, was trained on heavy clipping down to 1 dB input SDR, beats A-SPADE perceptually, but its code is partly CC BY-NC 4.0. Nothing open and 48 kHz was found for heavy speech clipping.

### Cited Findings
- DDD (Demucs-Discriminator-Declipper): a real-time-capable DNN declipper. It is reported to beat T-UNet and A-SPADE on perceived quality, and code plus pretrained models are public — [arXiv 2401.03650](https://arxiv.org/html/2401.03650)
- DDD repo: **16 kHz**, trained on VoiceBank-DEMAND clean speech clipped to 3 dB SDR and compared at 1 dB SDR. The code is MIT, but "a large portion of this repo is still under CC BY-NC 4.0" because it derives from Demucs. Checkpoints are on Google Drive — [stet-stet/DDD](https://github.com/stet-stet/DDD)
- Speech-Declipping Transformer (2024): complex spectrogram plus learnable temporal features, aimed at a wide range of input SDRs. Training clipping ran from 1 to 9 dB SDR on VoiceBank-DEMAND (9.4 h, 28 speakers) — [arXiv 2409.12416](https://arxiv.org/abs/2409.12416)
- Classical toolboxes: the rajmic/declipping2020 codes (MATLAB, needs LTFAT ≥ 2.4.0) include A-SPADE, S-SPADE and others from "A Survey and an Extensive Evaluation of Popular Audio Declipping Methods" — [rajmic/declipping2020_codes](https://github.com/rajmic/declipping2020_codes); [project page](https://rajmic.github.io/declipping2020/)
- Consistent dictionary learning (Rencker, Bach, Wang, Plumbley, LVA/ICA 2018) in Python — [LucasRr/Dictionary_learning_for_declipping_Python](https://github.com/LucasRr/Dictionary_learning_for_declipping_Python)
- A-SPADE/S-SPADE (MATLAB) — [andryr/spade-declipping](https://github.com/andryr/spade-declipping). Sparse declipping toolboxes from Gribonval et al. — [ENS Lyon page](https://perso.ens-lyon.fr/remi.gribonval/spade-evaluation-paper-toolboxes/)
- The URGENT 2025 challenge includes clipping among its seven distortions, with up to five applied together, so universal SE entries are trained on it — [URGENT 2025](https://arxiv.org/html/2505.23212)

### Inferences
- "12 dB+ clipping" (clip level ≥ 12 dB below peak) gives an input SDR roughly in the 1–3 dB range that DDD was trained for. DDD is therefore the closest open match for heavy clipping. To use it at 48 kHz it would have to be split into bands: declip at 16 kHz and keep the original high band, or the reverse. The CC BY-NC part rules out commercial use as is.
- SPADE-type methods are sample-rate agnostic and keep every unclipped sample exactly, by consistency constraints. That preserves the project's samples-in, samples-out alignment and the speaker's identity, but quality falls off at heavy clipping (per the survey's own framing; numbers not extracted here).

### Gaps
- No open diffusion-based speech declipper with weights was found. The search turned up the general "Diffusion Models for Audio Restoration" review ([arXiv 2402.09821](https://arxiv.org/pdf/2402.09821)), not a dedicated model.
- I did not extract the 2020 survey's quantitative ΔSDR at heavy clipping.
- No licence was found for the Speech-Declipping Transformer, nor whether it has code.

## 3. Packet loss concealment, and already-concealed Opus streams

### Takeaway
ICASSP 2024 PLC challenge: BS-PLCNet (NWPU & ByteAudio) and team 1024K tied for first. FRN (ICASSP 2023) is open, runs at **48 kHz** and is blind, needing no loss mask. Every PLC model found is meant to fill gaps in a stream. Applying one to audio that Opus already concealed (with its own PLC or Deep PLC) is not studied in anything found here; it becomes "repair a smooth-but-wrong segment", which is closer to inpainting or generative SE.

### Cited Findings
- ICASSP 2024 Audio Deep PLC Challenge: 9 systems, 8 real-time. Evaluation used ITU-T P.804 plus word accuracy on a harder dataset than INTERSPEECH 2022. Shared first place with final score 0.72: **1024K** and **NWPU & ByteAudio**. Then SpeechGroupIoA 0.69, HWYW 0.66, LEIBUS 0.59 — [arXiv 2402.16927](https://arxiv.org/html/2402.16927)
- The challenge abstract says one winner is better on naturalness and the other on intelligibility — [arXiv abs](https://arxiv.org/abs/2402.16927)
- BS-PLCNet: band-split (0–8 kHz GCRN plus 8–24 kHz GRU, so 48 kHz full-band), with multi-task f0 prediction, linguistic awareness and multiple discriminators. It tied for first at ICASSP 2024. BS-PLCNet 2 (two-stage, intra-model distillation) appeared at Interspeech 2024 — [arXiv 2401.03687](https://arxiv.org/abs/2401.03687); [arXiv 2406.05961](https://arxiv.org/abs/2406.05961)
- FRN (Full-band Recurrent Network, ICASSP 2023): 48 kHz, frame-causal STFT. It is **blind**, needing no loss mask or packet size, and the authors report it beats an offline non-causal baseline and a top 2022-challenge system. Official code is public — [arXiv 2211.04071](https://arxiv.org/abs/2211.04071); [Crystalsound/FRN](https://github.com/Crystalsound/FRN)
- Opus 1.5 Deep PLC: decoder-side and enabled with `--enable-deep-plc` at decoder complexity ≥ 5. Costs about 1% CPU and about 1 MB of binary. DRED (Deep REDundancy) sends up to 1 s of redundancy at 12–32 kb/s through an RDO-VAE. It is encoder plus decoder, needs jitter-buffer integration, and "the version included in Opus 1.5 will not be compatible with the final version" — [Opus 1.5 release](https://opus-codec.org/demo/opus-1.5/)
- URGENT 2025 includes packet loss as a distortion. TF-GridNet-S has been used as a filling module for packet loss in multi-stage URGENT systems — [TS-URGENet](https://arxiv.org/html/2505.18533); [URGENT 2025](https://arxiv.org/html/2505.23212)
- A 2026 paper, "Self-Supervised Test-Time Tuning for Packet Loss Concealment", exists — [arXiv 2607.01823](https://arxiv.org/html/2607.01823) (not read)

### Inferences
- Opus Deep PLC and DRED only work inside the decoder. Once a remote guest's audio has reached the recording as PCM, neither can be applied. DRED in particular needs redundancy that the sender transmitted.
- FRN is "blind" (it finds gaps itself), which is what a post-hoc tool needs. But it was trained on zero-filled gaps. An already-concealed Opus stream has no zeros, only extrapolated or faded segments, so FRN would probably not detect them. This is untested: the project would need a degradation that runs real Opus encode, loss and decode with PLC.
- BS-PLCNet runs at 48 kHz, but no public code repository was found.

### Gaps
- No public code or weights were found for BS-PLCNet, BS-PLCNet 2 or the 1024K system.
- FRN's repo licence and checkpoint details were not fetched.
- PLCMOS (Microsoft's PLC quality metric) availability and licence were not verified in this pass.
- I found no paper on repairing audio that was already concealed (post-PLC enhancement).

## 4. Codec artefact removal (Opus / AMR / low bitrate)

### Takeaway
Opus's own LACE/NoLACE are excellent and cheap, but they **cannot be applied to decoded PCM**: they need SILK bitstream features (pitch lag, LTP, quantised spectrum). They are also wideband-only (16 kHz) at 20 ms frames. For already-decoded audio the practical open routes are universal SE models trained on URGENT-style codec distortions (MP3/OGG in URGENT 2025), or generative SE. No open, PCM-only, Opus/AMR-specific post-filter with weights was found.

### Cited Findings
- LACE/NoLACE (Opus 1.5): DNN-adapted postfilters. Enabled with `--enable-osce`, decoder complexity 6 for LACE and ≥ 7 for NoLACE. **Wideband only, 20 ms frames only.** LACE is 100 MFLOPS (~0.15% CPU), NoLACE 400 MFLOPS (~0.75% CPU). The release says that with NoLACE "Opus is now perfectly usable down to 6 kb/s" and that at 9 kb/s it is "close to transparency" — [Opus 1.5 release](https://opus-codec.org/demo/opus-1.5/)
- NoLACE inputs mix encoder-side quantised features (spectrum, pitch lag, LTP coefficients), features computed from the decoded signal (cepstrum, autocorrelation) and bitrate. The paper says it "can generally be used with any speech codec that provides explicit pitch information". Results at 16 kHz, 6/9/12 kb/s. At 6 kb/s MOS is about 2.7 for Opus, 3.2 for NoLACE and 3.45 for LPCNet resynthesis, i.e. 92% of LPCNet's gain at 620 MFLOPS against about 3 GFLOPS — [arXiv 2309.14521](https://arxiv.org/html/2309.14521)
- IETF 121 slides give updates on Opus speech coding enhancement (Jan Büthe) — [IETF slides](https://datatracker.ietf.org/meeting/121/materials/slides-121-mlcodec-speech-quality-enhancement-02)
- Opus 1.5 licence: Opus is BSD-licensed (general knowledge, not re-verified on this page).
- URGENT 2025 trains and evaluates on codec loss (MP3 and OGG) among seven distortions. Multi-stage entries combine TF-GridNet with subband splitting for BWE, codec artefact reduction and PLC — [URGENT 2025](https://arxiv.org/html/2505.23212); [TS-URGENet](https://arxiv.org/html/2505.18533)
- Older backward-compatible approach: resynthesise Opus speech from decoded parameters with LPCNet/WaveNet — [arXiv 1905.04628](https://arxiv.org/pdf/1905.04628)

### Inferences
- For podcast VoIP guests the bench sees PCM only, so LACE/NoLACE are relevant only if the guest's raw Opus packets were captured. Some recording platforms might allow that; unverified.
- Codec damage on podcast guests at typical VoIP bitrates is mostly band-limiting plus a smeared high band. A BWE model (see 5) plus a universal SE model may cover more of it than a codec-specific post-filter would.

### Gaps
- No open PCM-only neural post-filter for Opus or AMR with released weights was identified. Searches returned mostly neural codecs, which are a different problem.
- Licences and weights of URGENT 2025 top systems were not checked.

## 5. Low-frequency bandwidth extension (below 300 Hz)

### Takeaway
I found no open neural model aimed at regenerating the **0–300 Hz** band of telephone or VoIP speech after Pulakka et al. (2011–2012). Recent open BWE models (AP-BWE, ClearerVoice super-resolution) extend **upward** only. Low-band extension seems to remain a classical, harmonic-synthesis problem in the literature. The project's Pulakka-style attempt did not win listening tests, which matches how small this research area is.

### Cited Findings
- Pulakka et al.: synthesises the lowest harmonics of voiced speech with sinusoidal synthesis, with extension-band energy estimated by a GMM from narrowband features and amplitudes/phases fitted to the narrowband input — [Interspeech 2011](https://www.isca-archive.org/interspeech_2011/pulakka11_interspeech.pdf); [Semantic Scholar](https://www.semanticscholar.org/paper/Bandwidth-Extension-of-Telephone-Speech-to-Low-and-Pulakka-Remes/f93ba9c3022e759984fedda869a917e4b6425658)
- Earlier classical work: Kornagel, "Improved artificial low-pass extension of telephone speech" (IWAENC 2003) — [PDF](https://www.rd.ntt/cs/team_project/icl/signal/iwaenc03/cdrom/data/0009.pdf)
- AP-BWE: a GAN with parallel amplitude and phase streams, MIT licence. It targets 16 kHz and **48 kHz** outputs, runs 292× real time on an RTX 4090 and 18× on one CPU at 48 kHz, and extends the **high** band — [arXiv 2401.06387](https://arxiv.org/html/2401.06387v2); [yxlu-0102/AP-BWE](https://github.com/yxlu-0102/AP-BWE)
- Other upward BWE work: "A lightweight and robust method for blind wideband-to-fullband extension of speech" ([arXiv 2412.11392](https://arxiv.org/pdf/2412.11392)) and CIS-BWE ([arXiv 2507.15970](https://arxiv.org/html/2507.15970v2)), neither read in depth.

### Inferences
- A neural route for the low band would probably have to be built: train on wideband or fullband speech high-passed at 300 Hz, so the target is the 0–300 Hz band. AP-BWE's amplitude/phase architecture could be retrained this way, since it is MIT-licensed. Universal SE models trained with "bandwidth limitation" distortions (URGENT) usually simulate low-pass, not high-pass, so they are unlikely to have learned low-band regeneration. Unverified: check URGENT's degradation config.
- On VoIP guests the 0–300 Hz loss is often milder than on PSTN (Opus wideband passes lower frequencies). The gap may be smaller in practice than "telephone" implies. Measure it on the real guest material.

### Gaps
- No 2022–2026 paper dedicated to neural low-frequency extension of speech was found within the search budget. That is absence of evidence from about two searches, not proof.

## 6. Strong general discriminative SE models (sample rate and licence)

### Takeaway
For 48 kHz with a permissive licence, **ClearerVoice-Studio MossFormer2_SE_48K** (Apache-2.0) is the main option. MP-SENet is MIT but 16 kHz. SEMamba is any-rate as an URGENT 2024 entry, but needs CUDA ≥ 12 and an RTX-class GPU. TF-GridNet is the URGENT baseline.

### Cited Findings
- ClearerVoice-Studio (Alibaba/ModelScope), Apache-2.0 code. Models: FRCRN and **MossFormer2_SE_48K** for enhancement, MossFormer/MossFormerGAN for separation, speech super-resolution to 48 kHz, and target-speaker extraction. Weights are on Hugging Face and ModelScope. No dedicated dereverberation or declipping model — [modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
- MP-SENet: parallel magnitude and phase estimation, MIT licence, **16 kHz** — [yxlu-0102/MP-SENet](https://github.com/yxlu-0102/MP-SENet); [arXiv 2305.13686](https://arxiv.org/pdf/2305.13686)
- SEMamba: checkpoints `SEMamba_advanced.pth` and `vd.pth` (VCTK-Demand plus DNS-2020), with PESQ 3.56–3.75 reported. As an NeurIPS 2024 URGENT entry it supports "all types of sampling frequencies". Requires CUDA ≥ 12.0, PyTorch 2.2.2 and an RTX-class GPU (not V100 or 1080 Ti). The licence type was not visible in the fetched page — [RoyChao19477/SEMamba](https://github.com/RoyChao19477/SEMamba); [arXiv 2405.06573](https://arxiv.org/pdf/2405.06573)
- RT-SEMamba (real-time distillation, 2026), MIT app — [arXiv 2608.12099](https://arxiv.org/html/2608.12099); [RT-SEMamba_app](https://github.com/RoyChao19477/RT-SEMamba_app). SEMamba++ for general restoration — [arXiv 2603.11669](https://arxiv.org/html/2603.11669)
- TF-GridNet: the URGENT 2025 baseline; scores are under section 1 — [URGENT 2025](https://arxiv.org/html/2505.23212)
- UniPASE (generative universal SE, 2026) also appeared in the URGENT-related search results — [arXiv 2604.14606](https://arxiv.org/html/2604.14606). The project already measured it: it moves voices away from their owners.
- The CCF AATC 2025 Speech Restoration Challenge retrospective exists — [arXiv 2509.12974](https://arxiv.org/pdf/2509.12974) (not read)

### Inferences
- MossFormer2_SE_48K is the one permissively licensed 48 kHz discriminative model worth adding as an engine. Being discriminative, it may share DeepFilterNet3's habit of removing speech with reverb, and the bench should check that.
- SEMamba's GPU requirement keeps it off a Mac/CPU bench. It needs a Colab-type run.

### Gaps
- SEMamba licence type was not confirmed.
- TF-GridNet's public URGENT checkpoint location and licence (ESPnet, presumably Apache-2.0) were not verified.
- Licence of the MossFormer2_SE_48K weights on HF/ModelScope was not checked separately from the code licence.
