# Identity preservation, hallucination metrics and gating for generative speech restoration

Context: UniPASE (WavLM-feature resynthesis) recovers the most damage in earshot's bench but lowers WeSpeaker ResNet34 cosine by 0.03-0.16 against the damaged input and damages clean audio. These notes cover (a) identity-preserving restoration, including two angles the owner added: a speaker profile built from clean stretches of the same recording, and compact parametric voice priors; (b) metrics for hallucination and speaker drift; (c) gates and routers; (d) hybrids that keep the original where it is good. Research date 2026-10-03. Sources are mostly arXiv preprints; numbers come from the papers and were not reproduced.

## Identity-preserving / speaker-conditioned restoration (incl. same-recording speaker profile, parametric voice priors)

### Takeaway
Conditioning a generative restorer on a clean clip of the same speaker helps, but the measured gains are small (+0.006 to +0.013 cosine in AnyEnhance), and the best-known reference-conditioned system (Voice-ENHANCE) reports no speaker-similarity numbers at all. No paper found builds a speaker profile from the clean stretches of the same long recording and reports speaker similarity for it; that would be new work, though AnyEnhance's 3-second in-context prompt and NANSY's single-sample test-time self-adaptation are the nearest mechanisms. Discriminative/predictive systems still beat generative ones on speaker similarity (Miipher-2 0.744 against TF-GridNet 0.945).

### Cited Findings
**UniPASE itself (the project's lead model)**
- UniPASE = DeWavLM-Omni (WavLM fine-tuned to enhance representations) + adapter + vocoder + PostNet. It works at 16 kHz internally, upsamples to 48 kHz, then resamples to the input rate. It is accepted at IEEE TASLP and took 1st place in the URGENT 2026 objective evaluation — [UniPASE arXiv 2604.14606](https://arxiv.org/abs/2604.14606)
- The paper measures SpkSim with **RawNet3** cosine: 0.96 on DNS2020 no-reverb (TF-GridNet 0.94), 0.79 with reverb (AnyEnhance 0.70, LLaSE-G1 0.55), 0.81 on the URGENT 2025 non-blind test (predictive BSRNN-FAN 0.85, TS-URGENet 0.83), 0.94 on PLC 2024 — [UniPASE HTML](https://arxiv.org/html/2604.14606)
- **On clean input** (ablation, Table IX) UniPASE yields PESQ 3.47 and SpkSim 0.94. So by its own authors' measurement, resynthesis costs about 0.06 cosine even when there is nothing to restore. This agrees with earshot's finding that it damages clean audio — [UniPASE HTML](https://arxiv.org/html/2604.14606)
- The paper has no limitations section. The hardest case it reports is long packet-loss bursts (50-150 packets), with WER 44.7% — [UniPASE HTML](https://arxiv.org/html/2604.14606)
- The repository is MIT-licensed and provides five checkpoints — [github xiaobin-rong/unipase](https://github.com/xiaobin-rong/unipase) (checked via the GitHub API on 2026-10-03)

**Reference/enrollment-conditioned generative restoration**
- **Voice-ENHANCE** (Qualcomm, Interspeech 2025): a speaker-agnostic GSR stage, then a Diff-VC-style diffusion voice-conversion stage conditioned on an **ECAPA-TDNN** embedding from clean enrollment speech of the same speaker. The enrollment does not need to be content-aligned ("short, uncorrelated segments of clean speech are obtained beforehand"). The content encoder is HuBERT VQ output and the vocoder is HiFi-GAN; the model has 209M parameters — [arXiv 2505.15254](https://www.arxiv.org/pdf/2505.15254)
- Voice-ENHANCE is evaluated **only with non-intrusive MOS predictors** (NISQA, UTMOS, WV-MOS, DNSMOS OVRL). A text search of the PDF finds no speaker-similarity numbers, so the paper does not show that the speaker conditioning preserves identity. Its GSR stage is trained on proprietary data, and no code release was found — [arXiv 2505.15254](https://www.arxiv.org/pdf/2505.15254)
- **AnyEnhance** (Amphion, masked generative model, 24 kHz Emilia training data): its "prompt guidance" keeps the first 3 s of clean audio unmasked as an in-context prompt during training, with some probability. At inference it runs with or without a prompt. Speaker similarity uses a **WavLM-based** speaker embedding — [arXiv 2501.15417](https://arxiv.org/pdf/2501.15417)
- AnyEnhance Table X, Similarity without → with prompt: Librivox GSR 0.955 → 0.963; CCMusic GSR 0.915 → 0.921; VoiceFixer SR 0.943 → 0.956. SpeechBERTScore also rises slightly (0.822 → 0.828 etc.). The authors say prompt guidance "generally offer[s] a larger gain" than their self-critic sampling — [arXiv 2501.15417](https://arxiv.org/pdf/2501.15417)
- Code is linked at [github viewfinder-annn/anyenhance-v1-ccf-aatc](https://github.com/viewfinder-annn/anyenhance-v1-ccf-aatc) and [amphionspace.github.io/anyenhance](https://amphionspace.github.io/anyenhance/) (licence not checked)
- **GenSE**: removing its "token chain prompting" lowers speaker consistency significantly. The authors attribute this to the prompting capturing the original speaker's acoustic characteristics — [GenSE OpenReview](https://openreview.net/pdf?id=1p6xFLBU4J) (from a search snippet; not read in full)
- The original **Miipher** uses speaker embeddings (and text via PnG-BERT) as conditioning. **Miipher-2** drops all conditioning and uses frozen USM features + parallel adapters + WaveFit at 16 kHz. Its SPK scores: Miipher-2 0.744, Miipher-USM 0.722, Miipher-1 0.585, **TF-GridNet (predictive) 0.945**. Code and checkpoints are deliberately not released ("potential misuse risks") — [Miipher-2 arXiv 2505.04457](https://arxiv.org/html/2505.04457v3); Miipher-1 conditioning per [Voice-ENHANCE §4.2](https://www.arxiv.org/pdf/2505.15254)
- Target-speaker extraction work on consistency: "Discriminative-Generative Target Speaker Extraction" and others add a **prompt guidance mechanism using a short reference utterance** and a **speaker consistency loss** that pulls the reference and reconstructed embeddings together — [arXiv 2601.06006](https://arxiv.org/pdf/2601.06006) (from a search snippet); also [GenTSE 2512.20978](https://arxiv.org/pdf/2512.20978) and [Explicit speaker consistency modeling 2507.09510](https://arxiv.org/html/2507.09510) (not read)
- In personalized SE, performance is sensitive to the enrollment speech (e.g. its emotional tone). USEF-PNet processes enrollment and mixture with a shared encoder instead of a separate speaker embedding — [arXiv 2505.12288](https://arxiv.org/abs/2505.12288)

**Speaker profile from the same recording (owner's added angle)**
- Self-supervised personalized SE trains on the test-time speaker's *noisy* data (no clean data), and weights frames by a "cleanliness score"/SNR ("data purification") — [arXiv 2104.02018](https://arxiv.org/pdf/2104.02018). This is the closest prior art for "learn the speaker from the stretches of this file that are good". It targets a discriminative denoiser and does not report speaker similarity.
- Test-time adaptation for SE with an autoregressive speech prior needs "only 1 second of representative unlabeled noisy speech" to calibrate a pretrained SE model — [arXiv 2609.03622](https://arxiv.org/html/2609.03622). It adapts to noise conditions, not to speaker identity.
- NANSY (NeurIPS 2021) proposes **test-time self-adaptation** from a single test-time sample, for analysis-synthesis voice reconstruction — [arXiv 2110.14513](https://arxiv.org/pdf/2110.14513)
- TTS speaker-adaptation evidence on how much same-speaker data is needed: with 1 minute of target data, plain fine-tuning struggled to reach high similarity; with 30 minutes it was competitive. More utterances from the target speaker helped more than more speakers did — [search summary of TTS adaptation literature, e.g. arXiv 2210.15868](https://arxiv.org/pdf/2210.15868) (from a search snippet, not verified in the paper)

**Compact parametric voice priors (owner's added angle)**
- A DDSP-vocoder SE (Meta, Aug 2025) predicts 80-dim mel spectral envelope + 12-band periodicity + F0 from noisy speech and resynthesises them with a source-filter DDSP vocoder (impulse train + filtered noise). On DNS2020 at 16 kHz: DNSMOS OVRL 3.23/3.43 (300K/600K params, non-causal), +4% STOI over the baseline, 24× less compute than a neural homomorphic vocoder. **No speaker-similarity numbers** are reported and no code release was found — [arXiv 2508.14709](https://arxiv.org/html/2508.14709)
- Speaker embeddings mostly encode **static spectral** information (mean pitch, HNR, shimmer, α-ratio). They miss dynamic identity markers such as speech rate, voiced/unvoiced durations, pitch and loudness variation. X-vectors encode more dynamic information than newer embeddings. **Equalization changes significantly alter similarity scores** (GE2E "catastrophic failure"). The paper proposes U3D, a rhythm metric (Wasserstein 2.15 same speaker vs 21.53 random) — [arXiv 2507.02176](https://arxiv.org/html/2507.02176v1), code [ubisoft-laforge-spkrid](https://github.com/ubisoft/ubisoft-laforge-spkrid)
- Pitch and creak co-vary across a population but not within an individual. Manipulating one without the other (normalizing-flow speaker manipulation) "greatly improved speaker verification" (no numbers in the abstract) — [arXiv 2602.14686](https://arxiv.org/abs/2602.14686)
- NANSY perturbs formants, pitch and frequency response during training so that the synthesis network learns to take timbre from a separate path. This is an explicit decomposition into linguistic (wav2vec), pitch (Yingram), timbre and energy — [NANSY arXiv 2110.14513](https://arxiv.org/pdf/2110.14513)

### Inferences
- The one number available on conditioning (AnyEnhance, +0.006 to +0.013) is much smaller than earshot's measured UniPASE loss (0.03-0.16). Speaker conditioning alone is unlikely to close the gap, so keeping the original signal where it is good (see hybrids) is likely to matter more.
- Both the UniPASE clean-input ablation (SpkSim 0.94) and Miipher-2 vs TF-GridNet (0.744 vs 0.945) point to the same cause: identity is lost in the feature bottleneck (WavLM/USM) and the vocoder, whatever the damage. That bottleneck is where to condition.
- A "same-file speaker profile" can be built with existing parts: pick clean stretches with a non-intrusive gate, average a speaker embedding (or a long-term average spectrum / formant and F0 statistics), then (i) use it as the AnyEnhance prompt, (ii) use it as a post-hoc target for an EQ/spectral-envelope match on restored segments, or (iii) use it as the reference for an identity check. Option (ii) follows from the finding that embeddings mostly encode static spectral information and are moved strongly by equalization: a long-term-spectrum match towards the speaker's own clean stretches may recover much of the cosine loss. This is untested; it should be measured on the bench.
- Because EQ moves embedding cosine strongly, part of UniPASE's cosine loss may be spectral tilt rather than lost identity. A per-speaker LTAS comparison (restored vs the clean stretches) would separate the two.

### Gaps
- No paper found reports speaker similarity for restoration conditioned on clean segments of the *same* long recording (podcast/session-level self-enrollment).
- Voice-ENHANCE's speaker-similarity effect is unmeasured. Whether it or AnyEnhance helps on VoIP damage specifically is unknown.
- No work found that uses a handful of source-filter parameters (formants, tilt, F0 range) as a *constraint* on a neural restorer with identity metrics reported. The DDSP SE paper uses them as the synthesis representation and reports no SpkSim.
- AnyEnhance's licence and weight availability were not checked.

## Hallucination, content-change and speaker-drift metrics

### Takeaway
The field's standard suite (URGENT 2025/2026) adds phoneme-level content metrics (LPS, SpeechBERTScore), ASR accuracy and speaker similarity to MOS predictors, because non-intrusive MOS predictors do not see hallucinations. Speaker-similarity numbers depend heavily on the embedding model, its bandwidth (16 kHz) and confounds such as EQ and duration, so drift should be judged with more than one embedding and against a per-speaker baseline.

### Cited Findings
- URGENT 2026 Track 1 metrics: non-intrusive **DNSMOS, NISQA, UTMOS, SCOREQ**; intrusive **PESQ, ESTOI, POLQA**; downstream-independent **SpeechBERTScore, LPS**; downstream-dependent **speaker similarity, emotion similarity, language-ID accuracy, character accuracy**. Rankings use a Friedman test, and the top 6 go to P.808 ACR/CCR listening tests — [ICASSP 2026 URGENT arXiv 2601.13531](https://arxiv.org/html/2601.13531)
- URGENT 2026 lesson: leading systems were mostly **hybrid** — "dual-branch or multi-stage pipelines, leveraging generative models for robust speech restoration and discriminative networks for precise signal preservation". Data curation (MOS filtering, cleaning with pretrained restoration models) was critical. In the quality-assessment track, Uni-VERSA-Ext and URGENT-PK correlated best with humans — [arXiv 2601.13531](https://arxiv.org/html/2601.13531)
- URGENT 2025 defines LPS as similarity between the clean and enhanced phoneme sequences and SpeechBERTScore as SSL-feature similarity, with SpkSim for speaker identity retention and WAcc from ASR — [Interspeech 2025 URGENT arXiv 2505.23212](https://arxiv.org/pdf/2505.23212)
- URGENT 2024 lessons: **UTMOS and SCOREQ** correlate best with MOS (KRCC). DNSMOS and LSD lag on KRCC; DNSMOS has a high LCC but rank mismatches. **SpkSim and WAcc have relatively high KRCC but much lower LCC.** ~100% of DNS5 LibriVox and CommonVoice 11 "clean" training data and ~25% of LibriTTS were affected by bandwidth mismatch. About 2,000 "clean" samples had negative WADA-SNR. The authors advise against relying on non-intrusive metrics alone — [arXiv 2506.01611](https://arxiv.org/html/2506.01611), code [urgent2024_analysis](https://github.com/urgent-challenge/urgent2024_analysis)
- The URGENT metric scripts are Apache-2.0 — [urgent2025_challenge](https://github.com/urgent-challenge/urgent2025_challenge). VERSA (a multi-metric toolkit) is Apache-2.0 — [ftshijt/versa](https://github.com/ftshijt/versa). SpeechBERTScore/LPS reference implementation: [DiscreteSpeechMetrics](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) (MIT) (all checked via the GitHub API)
- Generative vs discriminative comparison (2026): hallucination measured by Whisper-base WER/CER and LPS. At [-3,0] dB SNR the best was DisCoGAN (WER 26%, LPS 92%); at [-7,-4] dB, WER 39%, LPS 85%. GANs beat diffusion models on all hallucination metrics. **DNSMOS showed "minimal differentiation"** while the phoneme metrics degraded — [arXiv 2606.02913](https://arxiv.org/html/2606.02913)
- Hallucination taxonomy: linguistic (phoneme omissions/insertions, semantic drift) and acoustic (speaker inconsistency, prosodic deviation). "Both objective and subjective reference-free speech quality evaluations may be insensitive to hallucinations" — from search summaries of [PASE arXiv 2511.13300](https://arxiv.org/pdf/2511.13300) and [arXiv 2605.08608](https://arxiv.org/pdf/2605.08608) (not read in full)
- Levenshtein phoneme distance was proposed specifically to catch the error types generative SE produces — [Evaluation Metrics for Generative SE: Issues and Perspectives](https://www.researchgate.net/publication/374156402_Evaluation_Metrics_for_Generative_Speech_Enhancement_Methods_Issues_and_Perspectives)
- Intrusive and non-intrusive measures "correlate differently for each paradigm". Generative SE leaves "radically different residual distortions" from predictive SE — [arXiv 2306.03014](https://arxiv.org/abs/2306.03014)
- Optimizing SE against a perceptual metric can itself produce hallucination — [arXiv 2403.11732](https://arxiv.org/pdf/2403.11732) (title only, not read)
- Speaker embedding robustness (Interspeech 2025): ECAPA-TDNN, TitaNet, ECAPA2 and ReDimNet all degrade under domain, codec and sampling-rate shifts, **ReDimNet least**. Going from 16 kHz to 8 kHz bandwidth, TitaNet's EER rises 102% relative and ECAPA-TDNN's 57%. "Speaker embeddings depend on high-frequency information" (within 16 kHz) — [Ferro Filho et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/ferrofilho25_interspeech.pdf). ReDimNet is MIT — [IDRnD/redimnet](https://github.com/IDRnD/redimnet). It is also the embedding used for SECS in a DPO-aligned generative SE paper — [arXiv 2507.09929](https://arxiv.org/html/2507.09929v1)
- UniPASE (and, by inference, the URGENT-style toolkit) uses RawNet3; AnyEnhance uses a WavLM-based embedding; Voice-ENHANCE conditions on ECAPA-TDNN. Published numbers are therefore not comparable across papers — [UniPASE](https://arxiv.org/html/2604.14606), [AnyEnhance](https://arxiv.org/pdf/2501.15417), [Voice-ENHANCE](https://www.arxiv.org/pdf/2505.15254)
- Embedding confounds: duration sensitivity, noisy samples misread as a different identity, and equalization strongly altering scores. Embeddings miss rhythm, which motivated U3D — [arXiv 2507.02176](https://arxiv.org/html/2507.02176v1)
- WeSpeaker toolkit licence: Apache-2.0 — [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker)

### Inferences
- All widely used speaker embedders found run at 16 kHz, so none sees above 8 kHz. Full-band identity cues (sibilance, breath, "air") currently have no standard metric. For earshot, a band-limited spectral comparison against the same speaker's clean stretches (e.g. LTAS or LSD above 8 kHz) is the practical complement. It is a home-made probe, not prior art.
- Because SpkSim's linear correlation with MOS is low (URGENT 2024), earshot should judge cosine as rank/regression ("did it go down vs input"), which it already does, not as an absolute score.
- Running a second embedder (ReDimNet, or RawNet3 for comparability with UniPASE/URGENT) next to WeSpeaker ResNet34 would show whether the 0.03-0.16 drop is embedding-specific. This is cheap and would change confidence in the finding.
- LPS + SpeechBERTScore against the *damaged input* (not a clean reference, which podcast material lacks) can serve as a content-change check on real material. Their use without a clean reference is an inference, not something the papers validate.

### Gaps
- The exact phoneme recognizer and speaker model in the URGENT LPS/SpkSim scripts were not confirmed from the code (RawNet3 for SpkSim is inferred from UniPASE following the URGENT protocol).
- No wideband (>16 kHz) speaker-verification embedding with published evaluation was found.
- No paper was found that validates LPS/SpeechBERTScore computed against a *degraded* input instead of a clean reference.

## Non-intrusive quality estimators as a gate / router

### Takeaway
Usable gate candidates exist with permissive licences: UTMOSv2 (MIT), Distill-MOS (MIT, 16 kHz), DNSMOS P.835 (CC-BY-4.0 repo, 16 kHz), SQUIM (torchaudio, 16 kHz). NISQA is the only full-band one, but its weights are CC BY-NC-SA. Routing between enhancers with a learned quality predictor has precedent (SSEMS 2019), but none found for routing between generative and discriminative restoration. None of these estimators detects hallucination or speaker drift, so a gate decides *whether* to restore and an identity/content check decides *whether to keep the result*.

### Cited Findings
- DNSMOS P.835 gives SIG, BAK and OVRL. It was trained on P.808/P.835 ratings to stack-rank noise suppressors — [DNSMOS arXiv 2010.15258](https://arxiv.org/pdf/2010.15258), [DNSMOS P.835 arXiv 2110.01763](https://arxiv.org/pdf/2110.01763). The repo is CC-BY-4.0 — [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)
- DNSMOS and UTMOS are 16 kHz and "focus more on the low-frequency quality". NISQA is full-band and can assess 44.1 kHz speech — [arXiv 2506.19441 via search snippet](https://arxiv.org/pdf/2506.19441)
- NISQA: code MIT, **model weights CC BY-NC-SA 4.0** — [NISQA README](https://github.com/gabrielmittag/NISQA)
- UTMOSv2: MIT — [sarulab-speech/UTMOSv2](https://github.com/sarulab-speech/UTMOSv2)
- Distill-MOS: MIT (model card), input mono 16 kHz, distilled from an XLS-R-based SQA model, released October 2024 — [microsoft/Distill-MOS](https://github.com/microsoft/Distill-MOS)
- torchaudio SQUIM: the objective model predicts PESQ, STOI and SI-SDR without a reference. The subjective model predicts MOS but needs a **non-matching reference** (any clean speech). 16 kHz only. No weight licence is stated in the tutorial — [torchaudio SQUIM tutorial](https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html)
- The non-matching reference could be the same speaker's clean stretch, which suits a podcast. This is an inference; SQUIM was not designed for same-speaker references.
- Reliability: UTMOS/SCOREQ rank systems best against MOS in URGENT 2024, DNSMOS worse on rank — [arXiv 2506.01611](https://arxiv.org/html/2506.01611). DNSMOS does not register hallucination at low SNR — [arXiv 2606.02913](https://arxiv.org/html/2606.02913). In one TTS study, correlations on clean data were SQUIM 0.68, DNSMOS 0.41, UTMOSv2 0.39 — [search snippet, TTSDS2/related](https://www.isca-archive.org/ssw_2025/minixhofer25_ssw.pdf) (attribution to a specific paper unverified)
- Prior routing work: **SSEMS** (Interspeech 2019) uses the non-intrusive Quality-Net to pick among specialized SE models. It beat an autoencoder-based selector and a single general model, "particularly under high SNR" — [Zezario et al. 2019](https://www.isca-archive.org/interspeech_2019/zezario19_interspeech.html)
- ProxyMOS (2026) does adaptive routing among MOS *predictors* (which predictor to trust per input type), not among enhancers — [arXiv 2610.00419](https://arxiv.org/html/2610.00419)
- "Do no harm" evidence: in a medical-ASR study, original noisy audio beat enhanced audio in all 40 configurations (degradation 1.1%-46.6%) — [arXiv 2512.17562](https://arxiv.org/html/2512.17562v1). For LLM voice systems, enhancement-induced regressions outnumbered corrections in every condition — [arXiv 2608.30348](https://arxiv.org/html/2608.30348)

### Inferences
- A practical gate for earshot: a per-segment non-intrusive score (UTMOSv2 or Distill-MOS for licence reasons; DNSMOS SIG/BAK separate noise from speech distortion), plus a bandwidth estimate (effective cutoff from the spectrum, which needs no model). Skip restoration when the score is above a threshold and the bandwidth is full. Then a post-check: reject the generative output when speaker cosine vs input drops more than X, or LPS/SpeechBERTScore vs input falls. The thresholds must come from earshot's own bench. Nothing in the literature transfers.
- The 16 kHz estimators cannot judge damage above 8 kHz, so the bandwidth decision needs a separate signal-processing detector, not a MOS model.

### Gaps
- No dedicated, published, licensed speech bandwidth-estimator model was found in this search (the URGENT data analysis detects bandwidth mismatch, but its method was not examined).
- No paper found routes between generative and discriminative restoration per segment by quality estimate, with results.
- PLCMOS was not researched (out of time budget). Its licence and rate are unverified.
- The licence of the SQUIM weights is unverified.

## Hybrid approaches that keep the original where it is good

### Takeaway
The best URGENT 2026 systems were explicitly hybrid (a generative branch for restoration, a discriminative branch for signal preservation), and audio-restoration work (Apollo, RESTORE) argues for keeping the original low band and generating only what is missing. One ASR study shows that applying only part of the correction (α ≈ 0.25) can beat full correction. Published evidence measuring band-split merging on speaker similarity was not found.

### Cited Findings
- URGENT 2026: top submissions used "dual-branch or multi-stage pipelines, leveraging generative models for robust speech restoration and discriminative networks for precise signal preservation" — [arXiv 2601.13531](https://arxiv.org/html/2601.13531)
- Apollo (music/audio restoration): "an ideal generator should retain the original audio's low-frequency components and supplement smooth and delicate mid-to-high-frequency details". It models sub-bands with gain-shape representations — [Apollo arXiv 2409.08514](https://arxiv.org/html/2409.08514v2)
- RESTORE (music, 2026): isolates the generated bandwidth into an independent stem "without corrupting the original signal" — [arXiv 2609.28683](https://arxiv.org/html/2609.28683)
- Universal SE with "Regression and Generative Mamba" combines both paradigms — [arXiv 2505.21198](https://arxiv.org/pdf/2505.21198) (title/snippet only)
- Partial correction: scaling an SE mask's magnitude strength (M → A^α e^{jγφ}) with frozen models, Whisper's WER improved from 6.44 to 5.57 at α = 0.25 and went back to 6.46 at full correction. wav2vec 2.0 preferred α ≈ 0.85 (17.13 → 10.20). The best strength depends on the consumer and the condition — [Huo et al., arXiv 2607.11157](https://arxiv.org/pdf/2607.11157)
- Voice-ENHANCE chains a speaker-agnostic restorer and a speaker-conditioned generator (cascade, not fusion) — [arXiv 2505.15254](https://www.arxiv.org/pdf/2505.15254)

### Inferences
- For earshot's two source classes: on good local microphones, keep the input and generate only above the measured cutoff (crossover at the detected bandwidth), or bypass entirely. On VoIP guests, use the generative output below the cutoff only where the gate flags damage. Merging needs phase-coherent crossfades and must keep the sample-aligned contract, which UniPASE's resample-to-original output should allow.
- A wet/dry or per-band mix strength is a cheap knob to sweep on the bench against speaker cosine and LSD. The ASR result suggests full correction is not always best, but that result is for ASR, not for identity.

### Gaps
- No paper found that measures speaker similarity for band-split (original low band + generated high band) speech restoration.
- The URGENT 2026 hybrid systems' individual papers were not read, so their fusion mechanics and identity numbers are unknown.
