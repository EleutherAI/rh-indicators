# Paper Skeleton: Leading Indicators of Reward Hacking

**Target:** ICML 2026 (deadline Jan 28, 2026)

## Core Message

Prefills from a model fine-tuned on exploits can serve as a "leading indicator" of when a model in training is learning to exploit. By measuring how the model responds to exploit-eliciting prefills at early checkpoints, we can predict which problems will be exploited later—potentially saving compute by detecting reward hacking early.

---

## Title Ideas

1. "Prefill Sensitivity as a Leading Indicator of Reward Hacking"
2. "Early Detection of Reward Hacking via Exploit-Eliciting Prefills"
3. "Can We Predict Reward Hacking Before It Happens?"

---

## Abstract (Draft Sketch)

*Reward hacking poses a critical challenge for AI safety, yet detecting it typically requires waiting for problematic behaviors to emerge during training. Safety training can mean that reward hacking is initially a very rare behaviour, and significant compute investments may be needed to realise it even if the learning environment is defective. We propose prefill sensitivity—the degree to which exploit-eliciting reasoning prefills increase a model's exploitation rate—as a leading indicator of reward hacking propensity. Using a testbed of exploitable coding problems, we show that:*

1. *Models fine-tuned on exploits can provide prefills that elicit exploitation from other models*
2. *Prefill sensitivity measured early in training predicts which problems will be exploited later*
3. *[Optional: Comparison to baselines / compute savings claim]*

*Our method offers a potential approach for faster iteration in safety research, though applicability to RL and real-world scenarios requires further investigation.*

---

## 1. Introduction

### The Problem
- Reward hacking is dangerous but hard to study because it's hard to predict when/where it will occur
- Current approach: run full training, observe outcomes, iterate
- This is compute-intensive and slow

### Our Proposal
- Use **prefill elicitation** as a probe: seed the model with exploit-like reasoning and measure response
- Key insight: A model that readily continues exploit-eliciting reasoning may have latent propensity to exploit
- This gives a "leading indicator"—a signal before the behavior fully manifests

### Contributions
1. **Method:** Prefill sensitivity protocol using fine-tuned model's reasoning as prefills
2. **Validation:** Show prefill sensitivity predicts future exploitation on held-out problems
3. **Analysis:** Characterize the "prefill descent"—how required prefill decreases over training
4. [Optional: Compute savings quantification]
5. [Optional: Comparison to jailbreaking baselines]

---

## 2. Related Work

### Reward Hacking and Specification Gaming
- Skalse et al. "Defining and characterizing reward gaming"
- Krakovna et al. specification gaming examples
- Pan et al. on reward hacking in RLHF

### Emergent Misalignment
- Anthropic "Natural emergent misalignment" paper
- Key finding: models don't spontaneously reward-hack without intervention ("kicks")
- Our framing: if kicks are needed, can we detect propensity to be kicked?

### Prefill/Prompting for Elicitation
- Jailbreaking literature (GCG, AutoDAN, etc.)
- Steering vectors and activation engineering
- Difference: we use "natural" prefills from actual exploit solutions, not adversarial search

### Importance Sampling for Rare Event Detection
- ARC paper on importance sampling (arXiv:2410.13211)
- Why it doesn't directly apply (multi-token, structured outputs, etc.)
- But framing is useful: prefill as proposal distribution, natural probability as importance weight

---

## 3. Method

### 3.1 Testbed: Exploitable Coding Problems
- DJINN dataset: coding problems with both correct solutions and exploits
- Secure vs insecure verifiers
- Exploit families: test_skipping, result_manipulation, verifier_logic_override, etc.
- Train/test split: extract prefills from test set, validate prediction on train set

### 3.2 Prefill Sensitivity Protocol

**Step 1: Train prefill source model**
- Fine-tune on test-set exploits (SFT)
- This model learns exploit reasoning patterns

**Step 2: Extract prefills**
- Run prefill source on test problems
- Capture first N tokens of reasoning (chain-of-thought) that leads to exploits
- These prefills encode "exploit-eliciting reasoning"

**Step 3: Measure sensitivity at training checkpoints**
- At each checkpoint of the model being trained:
  - Measure: P(exploit | prefill with exploit reasoning) vs P(exploit | no prefill)
  - "Sensitivity" = how much the prefill increases exploitation

**Figure 1:** Schematic of the protocol

### 3.3 Sensitivity Metrics

**Token accessibility metric:**
- Minimum prefill tokens needed to trigger exploitation
- Sensitivity = 1 - (min_tokens / max_tokens)
- Higher = easier to elicit

**Logprob metric:**
- Measure model's logprob of the exploit-eliciting reasoning
- Higher logprob = more "natural" path to exploitation
- Tracks how much the model is "learning" the exploit reasoning

**Both metrics track the same phenomenon:** models becoming more susceptible to exploitation over training.

---

## 4. Experiments

### 4.1 Experimental Setup
- Base model: OLMo-3-32B-Think (or similar)
- SFT training on train-set problems with log-spaced checkpoints
- Prefill sensitivity measured at each checkpoint using test-set prefills
- Ground truth: which problems get exploited at baseline (no prefill) by final checkpoint

### 4.2 Main Result: Prefill Sensitivity Predicts Exploitation

**Survival Analysis:**
- Outcome: time-to-first-exploit for each problem
- Covariate: prefill sensitivity at early checkpoint
- Cox proportional hazards model

**Key Finding:**
- Hazard ratio for sensitivity: 3.75 (p=0.02)
- High-sensitivity problems at checkpoint X are ~3.75x more likely to be exploited per unit training time

**Figure 2:** Kaplan-Meier curves stratified by sensitivity group (high vs low)

**Figure 3:** Scatter plot of early sensitivity vs time-to-exploit

### 4.3 The Prefill Descent

**Key observation:** As training progresses, the minimum prefill needed to trigger exploitation decreases.

**Figure 4:** Prefill descent curves (min prefill vs training progress)

**Figure 5:** Sankey diagram showing flow of problems between prefill levels

**Interpretation:**
- Problems start requiring high prefill (100+ tokens of exploit reasoning)
- Over training, they "descend" to needing less prefill
- Eventually reach zero (exploit without any kick)
- This is the "leading indicator" in action

### 4.4 Logprob Trajectory Analysis

**Alternative metric:** Instead of token count, use logprob of prefill reasoning

**Figure 6:** Logprob trajectories over training

**Key finding:** Logprob increases over training, tracking the descent pattern

---

## 5. Discussion

### 5.1 Implications for Safety Research

**Faster iteration:**
- Don't need to wait for full exploitation to emerge
- Prefill sensitivity can be measured at any checkpoint
- Potentially allows early stopping or intervention

**Compute savings (speculative):**
- If detectable at 10% of training, could save 90% of wasted compute
- Caveat: Our results are from SFT, not RL; transfer uncertain

### 5.2 What Prefill Sensitivity Measures

**Interpretation:**
- Prefill sensitivity measures how "prepared" the model is to exploit
- High sensitivity = model's learned representations are close to exploit-producing states
- The prefill acts as a "bridge" to that state

**Analogy to jailbreaking:**
- Jailbreaking = artificially forcing model into harmful state
- Prefill sensitivity = measuring how natural/easy that path is
- We measure propensity, not just possibility

### 5.3 Limitations

1. **SFT only:** Our validation uses SFT training; RL dynamics may differ
2. **Requires fine-tuning on exploits:** Need examples of the behavior we're trying to detect
3. **Single testbed:** Results on DJINN may not generalize to other domains
4. **Held-out exploits:** Our "prediction" is within-distribution (same exploit types, different problems); OOD generalization unclear
5. **No true early stopping experiment:** We show prediction, not an actual intervention saving compute

### 5.4 Future Work

1. **RL validation:** Repeat with RL training checkpoints
2. **OOD generalization:** Test if sensitivity to one exploit type predicts other types
3. **Compute savings quantification:** Implement actual early stopping and measure savings
4. **Comparison to jailbreaking:** Compare prefill sensitivity to GCG/AutoDAN as detection methods
5. **Emergent misalignment:** Test if sensitivity predicts broader bad behaviors (not just specific exploits)

---

## 6. Conclusion

Prefill sensitivity offers a promising approach for early detection of reward hacking propensity. By measuring how readily a model continues exploit-eliciting reasoning, we can predict which problems it will eventually exploit—before that exploitation manifests at baseline. While our results are limited to SFT on a specific testbed, they suggest a general principle: latent propensities may be detectable before overt behavior emerges, potentially enabling faster and more compute-efficient safety research.

---

## Figures Needed

1. **Protocol schematic** (method overview)
2. **Kaplan-Meier curves** (survival analysis) ✓ Have this
3. **Sensitivity vs time-to-exploit scatter** ✓ Have this
4. **Prefill descent curves** ✓ Have this
5. **Prefill Sankey diagram** ✓ Have this
6. **Logprob trajectories** ✓ Have this
7. [Optional] **Compute savings diagram**
8. [Optional] **Jailbreaking comparison**

---

## What We Have vs What We Need

### ✓ We Have
- Prefill sensitivity protocol implemented
- SFT checkpoints with log-spaced saving
- Survival analysis showing prediction (HR=3.75, p=0.02)
- Prefill descent visualization
- Logprob trajectory analysis
- Multiple models (gpt-oss-20b primarily, some others started)

### ? Possible Additions (for stronger paper)
1. **Compute savings quantification**
   - Assume SFT→RL transfer holds
   - Calculate: "If we could detect at checkpoint X, we'd save Y% compute"
   - Purely analytical, no new experiments needed

2. **Jailbreaking baseline comparison**
   - Implement GCG or similar on same problems
   - Compare: which method detects earlier?
   - Trade-off: prefill requires fine-tuning, jailbreaking is model-agnostic
   - Status: Task exists but not yet implemented

3. **OOD generalization test**
   - Use emergent misalignment behaviors as holdout
   - Does sensitivity to test_skipping predict broader bad behavior?
   - Would strengthen generalization claims

4. **RL validation**
   - Run same analysis on RL training checkpoints
   - Would significantly strengthen claims
   - Status: RL training infrastructure exists

---

## Recommended Priorities for Filling Out

1. **Write the paper with what we have** — The survival analysis result is publishable as-is
2. **Add compute savings estimate** — Analytical, low effort, strengthens motivation
3. **Consider jailbreaking comparison** — If time permits, adds valuable baseline
4. **Defer RL/OOD to future work** — Honest about limitations, sets up follow-on

---

## Notes from Discussion

- Overall message: prefills (or similar) can speed up iteration when studying reward hacking
- We have evidence that fine-tuning on target exploits → prefills → early warning signal
- Story we want to tell: trajectory of prefill sensitivity predicts trajectory of exploitation

**Three potential extensions:**
1. Compute savings quantification (assume SFT→RL transfer)
2. Compare to jailbreaking methods (GCG etc.) as alternative detection
3. Test OOD generalization (use emergent misalignment as holdout)
