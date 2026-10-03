# LLM Post-Training

A curated resource list for learning how language models are post-trained: supervised fine-tuning, preference optimization, reinforcement learning, reasoning, and agents.

The list is ordered so it can be read top to bottom. Each section starts with the work that set up the vocabulary and ends with what is current as of October 2026. Everything here is something we would hand to a new researcher on a post-training team; if an entry doesn't teach a mechanism, a recipe, or a hard-won lesson, it isn't on the list.

## Contents

0. [Start here](#0-start-here)
   - [A path through this list](#a-path-through-this-list)
   - [Key methods at a glance](#key-methods-at-a-glance)
   - [Essays worth rereading](#essays-worth-rereading)
1. [RL foundations](#1-rl-foundations)
2. [Supervised fine-tuning and instruction data](#2-supervised-fine-tuning-and-instruction-data)
3. [RLHF: reward models and PPO](#3-rlhf-reward-models-and-ppo)
4. [Direct preference optimization](#4-direct-preference-optimization)
5. [AI feedback, self-improvement, and rubrics](#5-ai-feedback-self-improvement-and-rubrics)
6. [RL with verifiable rewards and reasoning](#6-rl-with-verifiable-rewards-and-reasoning)
7. [What RL actually does](#7-what-rl-actually-does)
8. [Process rewards, verifiers, and test-time compute](#8-process-rewards-verifiers-and-test-time-compute)
9. [Distillation](#9-distillation)
10. [Agentic and multi-turn RL](#10-agentic-and-multi-turn-rl)
11. [Reward hacking, safety, and failure modes](#11-reward-hacking-safety-and-failure-modes)
12. [Parameter-efficient fine-tuning](#12-parameter-efficient-fine-tuning)
13. [RL systems and infrastructure](#13-rl-systems-and-infrastructure)
14. [Frameworks and libraries](#14-frameworks-and-libraries)
15. [Open recipes and technical reports](#15-open-recipes-and-technical-reports)
16. [Datasets](#16-datasets)
17. [Evaluation](#17-evaluation)
18. [Books, courses, and blogs](#18-books-courses-and-blogs)
19. [Build it yourself](#19-build-it-yourself)
20. [Frontier (2026 watchlist)](#20-frontier-2026-watchlist)

---

## 0. Start here

If you read nothing else, read these. Together they cover the full modern pipeline: SFT, preference tuning, and RL with verifiable rewards. Most also appear in their topical sections below.

- [RLHF Book](https://rlhfbook.com/) - Nathan Lambert's free textbook on RLHF and post-training, from instruction tuning to reasoning RL. Now has a companion lecture series.
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) - The SFT → reward model → PPO pipeline that defined the field.
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) - A fully open, reproducible SFT → DPO → RLVR recipe with data, code, and ablations.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - Pure RL on verifiable rewards produces long chain-of-thought reasoning. The paper that reset the field's priorities.
- [Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) - Short visual introduction to the RLHF loop.
- [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) - End-to-end account of training a small model, with an honest post-training chapter on what worked and what did not.
- [Olmo 3](https://arxiv.org/abs/2512.13961) - The most complete fully open model flow to date: every dataset, checkpoint, and RL run for the Instruct, Think, and RL-Zero tracks.

### A path through this list

1. **Vocabulary (a week).** RLHF Book chapters 1-8, Spinning Up's policy gradient derivation, InstructGPT, and DPO. Then train a small model with TRL's SFT and DPO trainers.
2. **The modern recipe (two weeks).** Tülu 3, DeepSeekMath (for GRPO), DeepSeek-R1, DAPO, and Dr. GRPO. Read [section 7](#7-what-rl-actually-does) to calibrate what RL can and cannot do. Run GRPO on GSM8K or a toy verifier and watch reward, entropy, and length curves move.
3. **Systems and stability.** [Section 13](#13-rl-systems-and-infrastructure): async RL, training/inference mismatch, and why RL for LLMs is mostly an inference problem.
4. **Agents and the frontier.** [Section 10](#10-agentic-and-multi-turn-rl) and the latest reports in [section 15](#15-open-recipes-and-technical-reports). Read them side by side and note where labs agree (GRPO-family + verifiable rewards + distillation) and where they diverge.

### Key methods at a glance

The algorithms you will see referenced everywhere, and the one thing each changes.

| Method | What it changes | Source |
|---|---|---|
| PPO | Clipped importance ratio plus a learned value function (critic) and GAE | [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) |
| Rejection sampling / best-of-n SFT | Sample n, keep the best by reward or verifier, fine-tune on it. Still in nearly every pipeline | [Llama 2](https://arxiv.org/abs/2307.09288), [ReST-EM](https://arxiv.org/abs/2312.06585) |
| DPO | Removes the reward model and RL loop; preference pairs become a classification loss | [Rafailov et al. 2023](https://arxiv.org/abs/2305.18290) |
| RLOO | Drops the critic; baseline is the mean reward of the other samples for the same prompt | [Ahmadian et al. 2024](https://arxiv.org/abs/2402.14740) |
| GRPO | Drops the critic; advantage is reward normalized within a group of samples per prompt | [DeepSeekMath](https://arxiv.org/abs/2402.03300) |
| Dr. GRPO | Removes GRPO's per-response length normalization and std scaling, which bias toward long wrong answers | [Liu et al. 2025](https://arxiv.org/abs/2503.20783) |
| DAPO | Asymmetric clip-higher, dynamic sampling of non-trivial groups, token-level loss, overlong penalty | [Yu et al. 2025](https://arxiv.org/abs/2503.14476) |
| CISPO | Clips the importance-sampling weight rather than the update, so rare high-value tokens still get gradient | [MiniMax-M1](https://arxiv.org/abs/2506.13585) |
| GSPO | Importance ratio and clipping at the sequence level instead of per token; stabilizes MoE RL | [Zheng et al. 2025](https://arxiv.org/abs/2507.18071) |
| Truncated importance sampling | Corrects for the gap between the inference engine's and trainer's token probabilities | [Yao et al. 2025](https://fengyao.notion.site/off-policy-rl) |
| On-policy distillation | Student samples, teacher scores every token (reverse KL): dense reward at RL-like cost | [GKD](https://arxiv.org/abs/2306.13649), [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) |
| Multi-teacher on-policy distillation | Train domain experts with RL separately, then merge them into one student via on-policy distillation instead of multi-task RL | [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780), [DeepSeek-V4](https://arxiv.org/abs/2606.19348) |
| Rubric / checklist rewards | An LLM judge grades against explicit criteria, extending RL to non-verifiable tasks | [Rubrics as Rewards](https://arxiv.org/abs/2507.17746) |

### Essays worth rereading

Short pieces that shape how good post-training researchers think about what to work on.

- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Rich Sutton: general methods that scale with compute win. Search and learning are the two that do.
- [The Second Half](https://ysymyth.github.io/The-Second-Half/) - Shunyu Yao on why, once RL generalizes, defining the right evaluations and environments matters more than new algorithms.
- [Asymmetry of verification and verifier's law](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law) - Jason Wei: any task that is easy to verify will be solved by RL. A useful lens for picking problems.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) - A model of careful empirical work, and an argument about how few bits RL actually teaches per episode.

## 1. RL foundations

Enough RL to read the post-training literature without getting lost. In reading order rather than by date.

- [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book-2nd.html) - The standard textbook. Chapters 2, 3, and 13 are the most relevant.
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - Concise derivations of policy gradients, with clean reference implementations.
- [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) - Lilian Weng's tour from REINFORCE to PPO.
- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) - The intuition behind policy gradients in under an hour.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - GAE, the advantage estimator inside PPO-based RLHF.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - PPO, the clipped surrogate objective that most LLM RL still descends from.
- [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) - John Schulman's note on the k1/k2/k3 estimators used for KL penalties in nearly every RL trainer.

## 2. Supervised fine-tuning and instruction data

- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) - Instruction tuning on many tasks improves zero-shot generalization.
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) - Scaling tasks, model size, and chain-of-thought data in instruction tuning.
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) - Bootstrapping instruction data from the model itself.
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) - Cheap SFT on distilled data, and the start of the open chat-model wave.
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) - 1,000 curated examples are enough for strong SFT. The "superficial alignment hypothesis."
- [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944) - Distilled SFT plus DPO on AI feedback, a template many open models followed.
- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) - 1,000 examples of SFT plus "budget forcing" for test-time scaling.
- [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387) - A few hundred long reasoning traces unlock reasoning in a strong base model.
- [OpenThoughts: Data Recipes for Reasoning Models](https://arxiv.org/abs/2506.04178) - Systematic ablations of reasoning SFT data: sources, filtering, teachers, and scale.
- [Supervised Fine Tuning on Curated Data is Reinforcement Learning (and can be improved)](https://arxiv.org/abs/2507.12856) - iw-SFT: SFT on filtered data optimizes a loose lower bound on the RL objective; importance weighting tightens it.
- [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](https://arxiv.org/abs/2508.05629) - DFT: the SFT gradient is a policy gradient with an implicit reward that blows up on low-probability tokens. Rescaling each token's loss by its probability fixes it, in one line of code.
- [Iterative SFT (iSFT): dense reward learning](https://parsed.com/research/iterative-sft) - Parsed: the model drafts, an LLM judge explains what failed, the model revises until it passes, and you SFT on the result. Gets far more signal per example than a scalar RL reward, using plain SFT infrastructure.

## 3. RLHF: reward models and PPO

- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741) - Learning a reward model from pairwise human comparisons.
- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) - The first application of preference-based RL to language models.
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) - RLHF at scale on summarization, with careful reward model analysis.
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) - Anthropic's HH paper: helpfulness vs. harmlessness tension and online RLHF.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) - Goodhart's law, measured: how proxy reward diverges from gold reward as you optimize.
- [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964) - What makes PPO stable for LLMs, and PPO-max.
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) - The most detailed public account of iterative RLHF with separate helpfulness and safety reward models.
- [Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2307.15217) - Survey of what RLHF cannot do and why.
- [ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/abs/2310.10505) - REINFORCE with a greedy-decoding baseline.
- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740) - RLOO: drop the critic, use leave-one-out baselines.
- [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031) - Reproducing OpenAI's summarization RLHF and every detail that matters. See also the [blog version](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo).
- [HelpSteer2: Open-source dataset for training top-performing reward models](https://arxiv.org/abs/2406.08673) - Small, high-quality, permissively licensed preference data.
- [Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](https://arxiv.org/abs/2507.01352) - Strong open reward models from curated preference data at scale.

## 4. Direct preference optimization

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) - DPO: closed-form reparameterization of the RLHF objective as a classification loss.
- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) - IPO, and why DPO can overfit deterministic preferences.
- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) - SPIN: the model's own generations as the rejected side.
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) - Learning from unpaired thumbs-up/thumbs-down signals.
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) - Folds preference optimization into SFT.
- [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/abs/2404.10719) - When well-tuned PPO beats DPO, and why.
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) - Length-normalized, reference-free DPO variant.
- [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/abs/2406.09279) - Controlled comparison of data, reward model, and algorithm choices.

## 5. AI feedback, self-improvement, and rubrics

- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) - Train on self-generated rationales that reach the right answer.
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - Replacing human harmlessness labels with model critiques against a written constitution.
- [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998) - Grow-and-improve loops of sampling, filtering, and fine-tuning.
- [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) - AI preference labels can match human labels.
- [UltraFeedback: Boosting Language Models with Scaled AI Feedback](https://arxiv.org/abs/2310.01377) - The large AI-labeled preference dataset behind many open DPO models.
- [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585) - ReST-EM: expectation-maximization view of rejection-sampling fine-tuning.
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) - The policy judges its own outputs and trains on them iteratively.
- [Deliberative Alignment: Reasoning Enables Safer Language Models](https://arxiv.org/abs/2412.16339) - Teaching reasoning models to reason over safety specifications.
- [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746) - Structured rubrics as reward signals for non-verifiable tasks.
- [Checklists Are Better Than Reward Models For Aligning Language Models](https://arxiv.org/abs/2507.18624) - Instruction-specific checklists as RL rewards.

## 6. RL with verifiable rewards and reasoning

The center of gravity since early 2025: GRPO-family algorithms on tasks with checkable answers.

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Introduces GRPO: PPO without a critic, using group-normalized advantages.
- [REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization](https://arxiv.org/abs/2501.03262) - Global-batch advantage normalization for critic-free RL.
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) - Long-context RL, length penalties, and long-to-short transfer.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - R1-Zero (pure RL from base) and the multi-stage R1 recipe.
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) - Clip-higher, dynamic sampling, token-level loss, overlong shaping.
- [SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild](https://arxiv.org/abs/2503.18892) - Zero-RL across ten base models and what differs between them.
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) - Dr. GRPO: removes GRPO's length and difficulty biases.
- [Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model](https://arxiv.org/abs/2503.24290) - Minimalist, fully open reproduction of R1-Zero-style training.
- [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118) - Value-based PPO that beats critic-free methods on long CoT.
- [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) - The model proposes and solves its own tasks.
- [AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning](https://arxiv.org/abs/2505.16400) - Math-only then code-only RL stages for distilled models.
- [Skywork Open Reasoner 1 Technical Report](https://arxiv.org/abs/2505.22312) - Detailed ablations on entropy collapse and data for reasoning RL.
- [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864) - Long, stable RL runs with KL control and reference resets.
- [Magistral](https://arxiv.org/abs/2506.10910) - Mistral's RL-from-scratch reasoning recipe and infrastructure.
- [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585) - Introduces CISPO, which clips importance weights instead of updates, with cost accounting for large-scale RL on a hybrid-attention model.
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) - GSPO: sequence-level importance ratios, motivated by MoE RL instability.
- [The Art of Scaling Reinforcement Learning Compute for LLMs](https://arxiv.org/abs/2510.13786) - ScaleRL: sigmoid compute-performance curves and which design choices change the asymptote.

## 7. What RL actually does

Papers that try to explain why RLVR works, and where it doesn't.

- [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) - At large pass@k, base models catch up. RL sharpens rather than expands.
- [Reinforcement Learning for Reasoning in Large Language Models with One Training Example](https://arxiv.org/abs/2504.20571) - 1-shot RLVR and what it reveals about RL as elicitation.
- [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617) - Entropy collapse, its cause in covariance, and interventions.
- [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947) - Random and incorrect rewards still help some models, a warning about model-specific priors.

## 8. Process rewards, verifiers, and test-time compute

- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - Process supervision beats outcome supervision for math, and PRM800K.
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935) - Automatically labeled process rewards via rollouts.
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) - Compute-optimal search and revision at inference time.
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240) - Verifiers that reason before judging.
- [The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301) - Practical pitfalls of PRM data and evaluation.
- [Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456) - PRIME: dense rewards from an implicit PRM trained online.
- [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/) - Lilian Weng on test-time compute and reasoning, with a broad literature map.

## 9. Distillation

For distillation inside full post-training pipelines, see MiMo-V2-Flash, Nemotron-Cascade 2, and DeepSeek-V4 in [section 15](#15-open-recipes-and-technical-reports).

- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) - Reverse-KL distillation for generative models.
- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) - GKD: distill on student-generated sequences.
- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) - Thinking Machines: dense teacher supervision on student rollouts matches RL at a fraction of the cost.
- [MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training](https://arxiv.org/abs/2606.30406) - The method behind MiMo's post-training, studied in isolation on open models: merge RL-trained domain teachers into one student.

## 10. Agentic and multi-turn RL

- [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) - Early RLHF for a tool-using agent.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - The interleaved reason/act format most agent trajectories still use.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) - Self-supervised tool-use data.
- [ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL](https://arxiv.org/abs/2402.19446) - Turn-level critics for multi-turn RL.
- [Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139) - Executable repo environments for training SWE agents.
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) - RL with retrieval in the loop.
- [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) - RL that interleaves code execution with reasoning.
- [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073) - Multi-turn RL instabilities ("echo trap") and fixes.
- [SWE-smith: Scaling Data for Software Engineering Agents](https://arxiv.org/abs/2504.21798) - Synthesizing thousands of SWE task instances.
- [Demystifying Reinforcement Learning for Long-Horizon Tool-Using Agents: A Comprehensive Recipe](https://arxiv.org/abs/2603.21972) - A full recipe and ablations for long-horizon agent RL.
- [One sandbox per rollout, or how labs run RL for agents in 2026](https://huggingface.co/blog/sergiopaniego/rl-environments-2026) - How tasks, environments, sandboxes, and trainers fit together in current agent RL stacks.

## 11. Reward hacking, safety, and failure modes

- [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548) - How preference optimization rewards telling people what they want to hear.
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566) - Backdoors that survive SFT, RLHF, and adversarial training.
- [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) - Lilian Weng's survey of reward hacking, from classic RL to RLHF.
- [Alignment faking in large language models](https://arxiv.org/abs/2412.14093) - A model strategically complying during training to preserve its preferences.
- [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926) - CoT monitors catch reward hacking, but optimizing against them teaches obfuscation.
- [Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397) - Learning to reward-hack in coding environments generalizes to broader misalignment.

## 12. Parameter-efficient fine-tuning

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Train low-rank updates instead of full weights.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - LoRA on a 4-bit frozen base.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) - When LoRA matches full fine-tuning, and why it works especially well for RL.

## 13. RL systems and infrastructure

RL for LLMs is mostly an inference problem with a training step attached. These cover how to keep the GPUs busy.

- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143) - Ray plus vLLM separation of generation and training.
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256) - The design behind verl: single-controller orchestration with multi-controller workers.
- [Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models](https://arxiv.org/abs/2410.18252) - How off-policy you can go when overlapping generation and training.
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298) - Fully asynchronous RL with staleness-aware PPO.
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl) - vLLM and FSDP disagree on token probabilities for the same weights; truncated importance sampling fixes it.
- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) - Batch-invariant kernels, and why sampler/trainer mismatch matters for on-policy RL.
- [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788) - Precision choice as a fix for rollout/training numerical drift.
- [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374) - The Qwen team's first-principles account of when the token-level surrogate is valid, and the routing-replay and IS-correction tricks that keep MoE RL stable.
- [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) - Comparative survey of async RL designs across today's open frameworks.

## 14. Frameworks and libraries

**RL trainers**

- [TRL](https://github.com/huggingface/trl) - Hugging Face's SFT, DPO, GRPO, and reward modeling trainers. The easiest starting point.
- [verl](https://github.com/verl-project/verl) - Production RL library built on HybridFlow; FSDP or Megatron with vLLM or SGLang.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - Ray-based PPO, GRPO, and REINFORCE++ at scale.
- [slime](https://github.com/THUDM/slime) - Megatron plus SGLang RL framework used for GLM models.
- [Miles](https://github.com/radixark/miles) - Enterprise-oriented RL framework derived from slime.
- [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) - Asynchronous, decentralized-capable RL trainer.
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) - Library for building RL environments and rubrics, paired with the Environments Hub.
- [SkyRL](https://github.com/NovaSky-AI/SkyRL) - Modular RL for long-horizon, multi-turn agents.
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) - NVIDIA's scalable post-training library.
- [AReaL](https://github.com/inclusionAI/AReaL) - Fully asynchronous RL system.
- [ROLL](https://github.com/alibaba/ROLL) - Alibaba's large-scale RL library.
- [open-instruct](https://github.com/allenai/open-instruct) - AI2's post-training code behind Tülu and OLMo.
- [PipelineRL](https://github.com/ServiceNow/PipelineRL) - In-flight weight updates for on-policy RL with minimal idle time.
- [ART](https://github.com/OpenPipe/ART) - Agent reinforcement trainer for multi-step agents.
- [Atropos](https://github.com/NousResearch/atropos) - Nous Research's environment microservice framework for RL.
- [Tunix](https://github.com/google/tunix) - JAX-native post-training library.
- [TorchForge](https://github.com/meta-pytorch/torchforge) - PyTorch-native agentic RL library.
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) - Recipes for SFT, RL, and distillation on the Tinker training API.

**Environments**

- [OpenEnv](https://github.com/huggingface/OpenEnv) - Shared interface spec and hub for agentic RL environments.
- [Environments Hub](https://app.primeintellect.ai/dashboard/environments) - Open registry of RL environments built on verifiers.
- [Harbor](https://github.com/harbor-framework/harbor) - Containerized agent evaluation and rollout harness from the Terminal-Bench team.
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) - Web-agent environments under one gym interface.
- [TextArena](https://github.com/TextArena/TextArena) - Text-based games for multi-agent and self-play RL.

**Fine-tuning toolkits**

- [Unsloth](https://github.com/unslothai/unsloth) - Memory-efficient single-GPU SFT, DPO, and GRPO.
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - Config-driven fine-tuning across many methods.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Unified fine-tuning with a web UI.

## 15. Open recipes and technical reports

Read these for what labs actually do. The post-training sections are usually the most candid part.

**Open recipes** (post-training data, code, or RL environments released)

- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) - SFT → DPO → RLVR with every ablation published.
- [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3) - Dual-mode reasoning in a 3B model, with mid-training, SFT, and APO all released.
- [Olmo 3](https://arxiv.org/abs/2512.13961) - Instruct, Think, and RL-Zero tracks, plus the OlmoRL infrastructure and data.
- [INTELLECT-3: Technical Report](https://arxiv.org/abs/2512.16144) - Large-scale RL on a 100B+ MoE with prime-rl and community-built verifiers environments.
- [Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation](https://arxiv.org/abs/2603.19220) - Sequential domain-wise RL stages, then on-policy distillation to merge them, with data released.
- [Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](https://arxiv.org/abs/2604.12374) - Open hybrid Mamba-Transformer MoE with an agent-focused SFT and multi-environment RL pipeline and released post-training data.
- [Nemotron 3 Ultra Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) - The largest Nemotron 3 model, with its agent-focused post-training pipeline and the Nemotron-Posttraining-v3 datasets.
- [Instella-MoE Technical Report](https://arxiv.org/abs/2609.00791) - Fully open MoE whose post-training runs SFT, DPO, instruction-following RL, then multi-teacher on-policy distillation.
- [MiMo-V2.6 Technical Report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/main/MiMo_V2_6_technical_report.pdf) - The most open frontier-scale RL run so far: asynchronous GRPO at ~25k trajectories per step, a cost breakdown (rollouts vs. training vs. grader), and multi-prefix multi-teacher on-policy distillation. Ships with ~7,000 [RL environments and verifiers](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) and the [training framework](https://github.com/XiaomiMiMo/verl).

**Open-weight frontier models**

- [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704) - Synthetic-data-heavy alignment with iterative weak-to-strong alignment.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) - Iterative rounds of SFT, rejection sampling, and DPO, with detail on data mixes.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - R1 distillation into a general chat model, plus GRPO with rule and model rewards.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - Four-stage pipeline: long-CoT cold start, reasoning RL, thinking-mode fusion, general RL. Then strong-to-weak distillation.
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) - Agentic data synthesis and joint RL with verifiable and self-critique rubric rewards.
- [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471) - Expert models per domain, unified by self-distillation; the slime RL stack.
- [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925) - OpenAI's open-weight reasoning models: CoT-RL training, variable reasoning effort, and safety evaluations.
- [Introducing LongCat-Flash-Thinking: A Technical Report](https://arxiv.org/abs/2509.18883) - DORA, an asynchronous rollout orchestration system that trains >3x faster than synchronous RL, and domain-parallel RL with later fusion.
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) - Scaled-up GRPO with unbiased KL estimation and off-policy masking, plus large-scale synthetic agent environments.
- [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780) - Introduces multi-teacher on-policy distillation (MOPD): domain teachers trained by RL, merged into one student with dense token-level rewards plus outcome rewards.
- [LongCat-Flash-Thinking-2601 Technical Report](https://arxiv.org/abs/2601.16725) - Scaling DORA to multi-environment agentic RL across 10,000+ environments in 20+ domains.
- [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276) - Parallel-Agent RL (PARL): training a model to spawn and coordinate sub-agents.
- [Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters](https://arxiv.org/abs/2602.10604) - Combines verifiable rewards and preference feedback in one RL framework built to stay stable under large-scale off-policy training.
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763) - Asynchronous agentic RL at frontier scale.
- [Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5) - Qwen attributes most of the post-training gains to scaling RL across virtually all tasks and environments.
- [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729) - Agentic coding RL on large numbers of executable environments.
- [The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence](https://arxiv.org/abs/2605.26494) - Interleaved-thinking agent model trained with large-scale agentic RL.
- [Ling and Ring 2.6 Technical Report: Efficient and Instant Agentic Intelligence at Trillion-Parameter Scale](https://arxiv.org/abs/2606.15079) - Trillion-scale instant (Ling) and reasoning (Ring) models: Evo-CoT, shortest-correct-response distillation for token efficiency, and the KPop async RL framework.
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) - Specialist experts trained separately, consolidated by on-policy distillation; FP4 QAT during post-training.
- [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) - Small open models (2B-31B) with a thinking mode and quantization-aware training; useful as a reference for post-training at small scale.
- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653) - RL across general, agentic, and coding domains with multiple reasoning-effort levels, and QAT throughout SFT and RL.
- [GLM-5.3](https://z.ai/blog/glm-5.3) - Same base model as GLM-5.2; every gain comes from scaling post-training alone (more environments, longer runs on slime). The clearest public case that post-training scale is now a lever on its own.
- [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969) - Mostly architecture (cross-layer KV reuse, FP4 KV cache), but worth reading for how cheaper inference changes the economics of RL rollouts.

**Reasoning-focused reports**

- [Phi-4-reasoning Technical Report](https://arxiv.org/abs/2504.21318) - Careful SFT data selection plus short outcome-based RL.
- [Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949) - Reasoning toggle, SFT from R1, and large-scale RL.

## 16. Datasets

- [Tülu 3 datasets](https://huggingface.co/collections/allenai/tulu-3-datasets) - SFT mix, preference data, and RLVR prompts from the Tülu 3 recipe.
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) - Large AI-feedback preference dataset.
- [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) - Multi-attribute human preference data across code, multilingual, and general domains.
- [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) - 1.2M reasoning traces for SFT.
- [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) - Competition math problems with solutions, a common RLVR prompt source.
- [AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset](https://arxiv.org/abs/2504.16891) - The OpenMathReasoning corpus, a large math reasoning dataset behind the AIMO-2 winning solution.
- [MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) - ~7,000 RL environments with verifiers across code, cyber, knowledge work, web, and music, from a frontier RL run.
- [SWE-smith dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith) - Synthetic executable SWE tasks for agent training.

## 17. Evaluation

Grouped by what they measure: judges and arenas, instruction following, reward models, knowledge, code, and agents.

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) - LLM judges, their biases, and agreement with humans.
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) - Crowdsourced pairwise evaluation.
- [Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators](https://arxiv.org/abs/2404.04475) - Correcting judge length bias.
- [Instruction-Following Evaluation for Large Language Models (IFEval)](https://arxiv.org/abs/2311.07911) - Verifiable instruction constraints, also used as RL rewards.
- [RewardBench: Evaluating Reward Models for Language Modeling](https://arxiv.org/abs/2403.13787) - The standard reward model benchmark.
- [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022) - Hard expert science questions.
- [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) - Frontier-difficulty closed-ended questions across subjects.
- [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974) - Contamination-aware coding evaluation.
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) - Real GitHub issues as agent tasks.
- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045) - Tool-agent-user interaction with policy constraints.
- [Terminal-Bench](https://www.tbench.ai/) - Agents completing tasks in real terminal environments.

## 18. Books, courses, and blogs

- [RLHF Book](https://rlhfbook.com/) - Free online, also in print from Manning, with a companion [lecture series](https://www.youtube.com/watch?v=jQPiH-KB4B0).
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) - Includes alignment, SFT, and RL lectures and assignments.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) - Hands-on chapters on fine-tuning and reasoning RL with TRL.
- [Interconnects](https://www.interconnects.ai/) - Nathan Lambert's newsletter, the best running commentary on open post-training.
- [Sebastian Raschka's Ahead of AI](https://magazine.sebastianraschka.com/) - Clear walkthroughs of reasoning-model training and new technical reports.
- [Lil'Log](https://lilianweng.github.io/) - Lilian Weng's long-form surveys on RL, reward hacking, and reasoning.
- [Thinking Machines: Connectionism](https://thinkingmachines.ai/blog/) - Research posts on LoRA, distillation, and RL determinism.
- [GRPO++: Tricks for Making RL Actually Work](https://cameronrwolfe.substack.com/p/grpo-tricks) - Cameron Wolfe's careful walkthrough of every GRPO modification since R1, with the math.
- [John Schulman's blog](http://joschu.net/blog.html) - Short notes from one of the authors of PPO and InstructGPT.

## 19. Build it yourself

Reading is not enough. Each of these is small enough to finish and teaches something the papers can't.

- [nanochat](https://github.com/karpathy/nanochat) - Andrej Karpathy's end-to-end ChatGPT clone: pretraining, SFT, and RL in one small, readable codebase.
- [TRL GRPO Trainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) - Run GRPO on a single GPU, then switch on the DAPO and Dr. GRPO loss options and compare the curves.
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) - Reference implementations of RL, preference learning, and on-policy distillation without managing infrastructure.
- [The N Implementation Details of RLHF with PPO](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) - Reproduce it. Debugging a PPO implementation until it matches teaches more than any paper.
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) - Write your own environment and reward function, then train on it with prime-rl.

## 20. Frontier (2026 watchlist)

Recent work that is promising but not yet settled. Expect some of these to age quickly.

- [Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning](https://arxiv.org/abs/2605.06241) - Extends the "RL sharpens rather than teaches" line of work, and argues that much cheaper methods recover most of RLVR's gains.
- [RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards](https://arxiv.org/abs/2605.10899) - Pushing rubric rewards further into non-verifiable tasks.
- [OpenThoughts-Agent: Data Recipes for Agentic Models](https://arxiv.org/abs/2606.24855) - OpenThoughts-style controlled data ablations, applied to agent trajectories.

---

## Source policy

Entries should be primary sources: papers, official technical reports, the original blog post, or the canonical repository. Prefer work that explains a mechanism or provides a reproducible recipe over announcements and benchmark leaderboards. Secondary summaries are included only when they are the clearest available explanation of a topic.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
