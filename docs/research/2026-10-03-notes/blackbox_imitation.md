# Black-box imitation (distillation) of audio processors, as context for imitating dxRevive

Scope: learning to imitate an audio effect or enhancer from paired input/output recordings. Not legal advice. Searched 2026-10-03; 15 tool calls, so coverage is broad, not deep.

## 1. Black-box modelling of audio effects: methods, data needs, accuracy

### Takeaway
Black-box modelling of *deterministic* effects from paired audio is established and works well with surprisingly little data: minutes, not hours. Static or slowly varying nonlinearities (amps, pedals, compressors) reach near-transparent error. Time-varying effects need extra structure, such as feeding the LFO in as an input. All of this work targets signal-level effects with a single correct output per input. None of it targets a learned, stochastic restorer.

### Cited Findings
- Wright, Damskägg & Välimäki (DAFx-19) model tube amps and distortion pedals with a single LSTM/GRU layer plus a dense layer. It matches or beats a WaveNet-style model at much lower cost and runs in real time in a JUCE plug-in. — [DAFx 2019 paper](https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)
- Data used: 8 min 10 s of guitar and bass in total (5 min 42 s train, 1 min 24 s validation, 1 min 4 s test) at 44.1 kHz. The loss is pre-emphasised error-to-signal ratio (ESR) plus a DC term. Training uses half-second segments. — [DAFx 2019 paper](https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)
- Neural Amp Modeler (NAM) captures from a fixed, roughly 3-minute test signal of sweeps and noise "designed to excite the amp across its full frequency and dynamic range". The output must be sample-exact in length, with no fades or dither. — [Nail The Mix guide](https://www.nailthemix.com/the-producers-guide-to-capturing-your-own-amps-with-nam); [TONE3000 instructions](https://www.tone3000.com/capture/instructions)
- NAM community quality bands for ESR: < 0.01 "great", 0.01 to 0.035 "not bad", 0.035 to 0.1 "might sound okay", 0.1 to 0.3 "probably won't sound great". This is a community heuristic, not a perceptual study. — [Nail The Mix guide](https://www.nailthemix.com/the-producers-guide-to-capturing-your-own-amps-with-nam)
- Steinmetz & Reiss (AES 152, 2022): an efficient TCN with rapidly growing dilations models the LA-2A optical compressor, a device with long time constants. They report state-of-the-art accuracy, real-time operation on CPU, and "only 10 minutes of training data". — [arXiv 2102.06200](https://arxiv.org/pdf/2102.06200); [micro-tcn repo](https://github.com/csteinmetz1/micro-tcn)
- Time-varying effects: Wright & Välimäki model a phaser and a flanger with a grey-box network that takes both the audio and an LFO signal as input. The LFO is measured from the device's time-varying frequency response. Best case: 1.3 % ESR on a flanger pedal. — [Aalto portal](https://research.aalto.fi/en/publications/neural-modelling-of-lfo-modulated-time-varying-effects); [JAES paper](https://www.aes.org/tmpFiles/elib/20251007/21119.pdf)
- Other related work:
  - Modelling black-box effects with time-varying feature modulation. — [arXiv 2211.00497](https://arxiv.org/pdf/2211.00497)
  - NablAFx, an open PyTorch framework for black-box and grey-box effect modelling. — [arXiv 2502.11668](https://arxiv.org/pdf/2502.11668)
- Neural proxies: networks trained to mimic a processor, including how its parameters act, so that a non-differentiable plug-in can be used in gradient training. "Hybrid" proxies use the real DSP at inference time. — [Frontiers review 2025](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1580395/full)
- DeepAFx puts third-party black-box plug-ins inside a network. It estimates their gradients stochastically, so no proxy or re-implementation is needed. — [arXiv 2105.04752](https://arxiv.org/pdf/2105.04752)

### Inferences
- The effect-modelling literature shows that a few minutes of well-designed excitation pins down a fixed nonlinear map. A speech restorer is a different kind of system. Its output depends on what it recognises as speech, noise or codec damage, so the probe signal has to be *realistic, degraded speech*, not sweeps. That moves the data requirement towards the speech-enhancement regime (next section) and away from the "3 minutes" regime.
- The DeepAFx approach (the plug-in as a black-box layer) and the proxy approach both assume you can run the plug-in many times on chosen inputs. That is the same access a distillation would need.
- ESR is a waveform metric. It suits deterministic effects. For dxRevive's stochastic bandwidth extension it would penalise any valid sample that differs from the one reference rendering (see section 2).

### Gaps
- I found no paper or project that black-box models a *neural* speech restorer, or any commercial ML audio plug-in, from its input/output pairs. The search for "imitating commercial noise suppression plugin" returned only RNNoise and unrelated patents.
- I found no systematic data-quantity curve (accuracy against minutes of training data) for effect modelling. The figures above are single operating points.

## 2. Teacher-student distillation of speech enhancement; stochastic teachers

### Takeaway
Training a student SE model on a teacher's outputs used as pseudo-targets is routine. Students often *match* the teacher and sometimes exceed it. When they do, it is because they also see other signals: true clean targets, auxiliary losses, or personalisation. Pure imitation of outputs is, at best, bounded by the teacher. For a stochastic teacher, recent generative SE work uses distribution matching or consistency objectives instead of plain regression, because plain regression collapses to the average output.

### Cited Findings
- Personalised SE via knowledge distillation: a large teacher's denoised outputs serve as pseudo-targets for a small student. The personalised students outperform larger non-personalised baselines, so compression costs nothing on that speaker. — [arXiv 2105.03544](https://arxiv.org/html/2105.03544)
- ROSE-CD distils a 30-step diffusion SE teacher into a one-step consistency model. It is 54 times faster with "superior performance" compared with the teacher on VoiceBank-DEMAND. The authors attribute the gain to auxiliary losses that let the student "recover from teacher-induced errors". — [arXiv 2507.05688](https://arxiv.org/abs/2507.05688)
- General knowledge-distillation finding (not specific to SE): a student can be more accurate than the noisy teacher labels it was trained on, acting as a "denoiser" of teacher errors. — [arXiv 2312.10185](https://arxiv.org/html/2312.10185v1)
- Handling stochastic or generative teachers: Distribution Matching Distillation trains the student to match the teacher's output *distribution*. Regression on teacher-generated pairs is kept only as a stabiliser. — [Yin et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.pdf)
- One-step generative SE without a teacher also exists, for example MeanFlowSE ("requires no knowledge distillation or external teachers"). — [arXiv 2509.14858](https://arxiv.org/pdf/2509.14858)

### Inferences
- Most of dxRevive (denoise, de-reverb, EQ, de-clip) is a deterministic per-frame map. A student trained with a regression loss on (input, dxRevive output) pairs should be able to approach it. This is analogous to the amp-modelling results, but needs speech-like inputs covering every degradation that matters.
- The stochastic bandwidth extension is the hard part. An L1/L2 loss against single renderings will learn the *mean* of the extension: a smooth, muffled high band, the classic regression-to-mean failure. Options suggested by the literature:
  - an adversarial or distribution-matching loss on the high band;
  - several dxRevive renderings per input as samples of the target distribution;
  - splitting the job: imitate the deterministic band by regression, and use an open generative bandwidth-extension model for the top.

  These are inferences, not measured results.
- "Exceeding the teacher" in the literature always needed something beyond the teacher's outputs (clean ground truth, auxiliary losses). A pure imitation of dxRevive should be expected to approach dxRevive, not beat it. Beating it means training on true clean targets as well, and at that point it is ordinary SE training with dxRevive as a regulariser.
- In this repo's terms, sample alignment matters. dxRevive's latency must be measured and compensated before pairs are used as targets, or the student learns the offset (see the AGENTS.md contract).
- A roughly 4M-parameter per-frame STFT teacher has low capacity. A student of similar or larger size has, in principle, enough capacity to represent it. Data coverage is the binding constraint, not model size.

### Gaps
- I found no published distillation from a *closed, black-box commercial audio API or plug-in* in the SE literature. The LLM case, distilling from GPT or Claude outputs, is the closest documented practice.
- I found no figure for how many hours of paired data an SE student needs to match a teacher. The papers above use standard SE corpora (VoiceBank-DEMAND etc.) but report no data-scaling curves.

## 3. Legal and ethical landscape (EU/Germany): summary only, not legal advice

### Takeaway
EU software copyright protects a program's code (its expression), not its functionality. A lawful user's right to observe, study and test how the program behaves cannot be removed by contract. Distilling from outputs copies neither code nor weights. The remaining risks are:
- contract terms;
- German unfair-competition law on imitation (UWG §4 Nr. 3), which turns on market-facing facts such as confusion about origin and exploiting reputation;
- the use of dxRevive's name or presets in marketing.

Uncertainty is high. I found no case law on training models on the output of audio software.

### Cited Findings
- Art. 5(3) Directive 2009/24/EC: a person entitled to use a copy may, without authorisation, "observe, study or test the functioning of the program in order to determine the ideas and principles which underlie any element of the program", provided they do so while loading, displaying, running, transmitting or storing it as they are entitled to. — [EUR-Lex 2009/24/EC](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32009L0024)
- Art. 8(2): "Any contractual provisions contrary to Article 6 or to the exceptions provided for in Article 5(2) and (3) shall be null and void." — [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32009L0024)
- Recital 11: only the expression of a program is protected. The ideas and principles underlying any element of it are not. — [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32009L0024)
- Art. 6 (decompilation) is narrower: it is allowed only for interoperability, only when the information is not otherwise available, and only for the necessary parts. — [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32009L0024)
- CJEU, SAS Institute v World Programming (C-406/10, 2 May 2012) held:
  - the functionality of a program, its programming language and its data-file formats are not protected by software copyright;
  - a licensee may observe, study and test the program to find its ideas and principles, provided it infringes no exclusive right.

  WPL had reproduced SAS functionality by studying its behaviour. — [CURIA press release](https://curia.europa.eu/jcms/jcms/P_87138/de/); [SCL analysis](https://www.scl.org/2451-sas-institute-inc-v-world-programming-ltd-report-and-analysis/); [Wikipedia summary](https://en.wikipedia.org/wiki/SAS_Institute_Inc_v_World_Programming_Ltd)
- German trade-secrets law (GeschGehG §3) expressly permits obtaining information by observing, examining, dismantling or testing a product that is publicly available or lawfully held, *unless a contract excludes it*. — [Luther law firm](https://www.luther-lawfirm.com/en/newsroom/blog/detail?tx_fwluther_shownews%5Baction%5D=show&tx_fwluther_shownews%5Bcontroller%5D=News&tx_fwluther_shownews%5Bnews%5D=6043&cHash=b5e74b6b55688802bacedf6962a3600d); [channelpartner.de](https://www.channelpartner.de/a/reverse-engineering-ist-zulaessig,3335842)
- That permission is separate from the limits of UWG §4 Nr. 3 (unfair imitation: deceiving buyers about origin, exploiting reputation). One commentator holds that someone who learned the information by permitted reverse engineering cannot be accused of "unfair acquisition" under §4 Nr. 3 c). The other prongs, origin deception and exploitation of reputation, still apply. — [Omsels commentary](https://www.omsels.info/die-verbote-oder-was-darf-ich-nicht/geheimnisschutzgesetz/2-eigene-informationen/rechtmaessiger-erwerb-nutzung-offenlegung); [IHK Frankfurt overview](https://www.frankfurt-main.ihk.de/recht/uebersicht-alle-rechtsthemen/wettbewerbsrecht/unlauterer-wettbewerb/mitbewerberschutz/nachahmung-von-waren-oder-dienstleistungen-5196502)
- Model outputs and distillation (mostly US and LLM commentary):
  - Machine-generated outputs generally lack copyright because they have no human author.
  - Distillation copies no weights or code.
  - So the main legal lever is the terms of service. Example: OpenAI's terms forbid using output "to develop models that compete with OpenAI".

  — [Ertas AI blog](https://www.ertas.ai/blog/ai-model-distillation-ip-law); [Monash Lens](https://lens.monash.edu/ai-distillation-and-the-law-why-learning-from-claude-or-gpt-may-not-be-copyright-infringement/)

### Inferences
- What applies to dxRevive, as described by the caller (the EULA has no reverse-engineering clause and no clause on output use), plus the sources above:
  - Running the licensed plug-in on one's own audio and recording the results is the conduct Art. 5(3) protects.
  - Art. 8 would void a contract term that tried to forbid it.
  - Even under GeschGehG, with no contractual exclusion, testing is permitted.
  - Training on the outputs copies no code.
  - SAS v WPL says reproducing functionality is not copyright infringement.
- Open legal points (my inference; not settled by any source found):
  1. Art. 5(3) is framed as "to determine the ideas and principles". Whether bulk generation of training pairs for a competing product stays within that purpose has not been tested in court.
  2. If the trained student reproduces dxRevive's *outputs* closely, someone might argue the weights encode protected expression. Under the reasoning above that seems weak, but it is untested for neural nets.
  3. Under UWG §4 Nr. 3, the risks are commercial: marketing a product as "dxRevive-like" or "a free dxRevive", or giving it a confusingly similar name or UI. For an open-source research tool that never invokes the brand, these risks look low.
- German implementation: Art. 5(3) is implemented in §69d(3) UrhG and Art. 8 in §69g(2) UrhG. This is from background knowledge and was not verified in this session.
- Ethics (beyond law):
  - Accentize trained dxRevive on its own data and research. A free clone substitutes directly for a small company's product. That matters even if it is lawful.
  - Results for this repo should be framed as "measured against dxRevive", not "dxRevive weights" or "dxRevive in open source".
  - AGENTS.md already forbids lifting weights out of a commercial product. A distilled student is not that, but it is close in spirit, and the README licence column would need a clear statement of how the student's training targets were produced.
- Practical hedge: a model trained on true clean targets, using dxRevive only as an evaluation reference, sidesteps almost all of the above.

### Gaps
- I found no court decision, in the EU or Germany, on training an ML model on the outputs of commercial software, audio or otherwise.
- I did not read the current text of §69d UrhG or UWG §4 Nr. 3, or German commentary on how they apply to ML. The statutory cross-references above are unverified.
- I did not verify Accentize's EULA myself. These notes rely on the caller's statement that it has no reverse-engineering clause and no clause on output use. Any online terms, store or licence-server terms, or terms changed since then could change the picture.
- Whether the EU AI Act's general-purpose model obligations touch a small distilled SE model: almost certainly not, since it is not a general-purpose AI model, but this was not checked.
