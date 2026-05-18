# The ARC-AGI 2 (2025)

**Purpose:**

The purpose of this document is to develop solutions for the ARC-AGI 2 challenge.

---

**Resources from ARC-AGI-1:**

ARC-AGI Technical Report 2024: <https://arxiv.org/pdf/2412.04604v1>

<https://arcprize.org/blog/arc-prize-2024-winners-technical-report>

Use approaches like Graph neural networks / Graph attention networks.

<https://github.com/graphdeeplearning/graphtransformer/blob/main/docs/03_run_codes.md>

---

# Previous Year Solutions

**Top Prizes:**

**Final Results:**

1. Code: [arc\_prize\_v8 | Kaggle](https://www.kaggle.com/code/gregkamradt/arc-prize-v8?scriptVersionId=211457842)

   Paper: <https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf>

2. Code: <https://www.kaggle.com/code/ironbar/single-task-test-time-fine-tuning-for-arc24?scriptVersionId=199282752>

   Paper: <https://ironbar.github.io/arc24/05_Solution_Summary/>

3. Code: <https://www.kaggle.com/code/gregkamradt/arc-prize-2024-solution-4th-place-score-40-811b72>
4. Code: <https://www.kaggle.com/code/gregkamradt/small-sample-arc24>
5. Code: <https://www.kaggle.com/code/gregkamradt/arc-prize-2024-8th-place-solution>

   Paper: <https://drive.google.com/file/d/1kTom9M54LVfLbPDQHpGgfUs3y1IYIpy2/view?usp=sharing>

**Public Leader Board Toppers:**

**Paper Prizes:**

**Ice Cuber Solution - 2020:**

<https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/writeups/icecuber-1st-place-solution-code-and-official-docu>

**Github:** <https://github.com/top-quarks/ARC-solution>

---

# Previous Solutions all the way from 2020 till now

---

# Meet with Lewis

**Agenda:**

**Resources:**

Hierarchical Models: <https://arxiv.org/pdf/2506.21734v2>

---

# Approach

**Approach:**

-> Explore the current dataset to understand all different problems.
-> Generate millions of such dataset and release it in public.
-> Combine different architectures.
-> Like induction and transduction.
-> Mechanistic interpretability.

**Activity: (13-08-2025):**

-> learnt about the GCN (GNN approach) - it seems it is mainly used for classification and a lot of analysis purposes.

-> Explore on different types of architectures for the 2d space / geometrical applications

**Questions:**

It seems most of the approaches are going with Transformers? Is that the only one working or people just went ahead with them because they are popular and others are doing that?

A lot of people are also using some combinations of some fixed rules and some DSL based approaches. But whether that is needed is a question.

**Intuition:**

We humans when we look into them we see them as different objects and try to find the output by mapping how the position and spatial relationship between different objects changed between the input and the output. Maybe the LLM might do the same implicitly (have to analyze or interpret the LLM for this purpose).

<https://arahim3.github.io/arc-agi-guide/?utm_source=chatgpt.com#ecosystem>

The above webpage contains a lot of general details about the ARC-AGI-2 competition.

**Activity Log: (15/08/2025):**

-> Explored on the DSL approaches, and some previous year solutions.

**Action items:**

-> Detailed study of the DSL approach and how it is similar to the current graph based approach in mind.

---

# Goals

**Goals:**

**Model accuracy: 95%**

**-> Hey man, you can fucking do this…**

**Paper publication:**

- Explore different approaches, like using other architectures rather than just blindly following the transformers.
  - May be GNN alone, Some weird architectures, may be cnn,
  - But we need a network that learns the representation and the spatial and positional features very well.
- Do mechanistic interpretability of the existing AI models extensively :) Don't feel ashamed to use AI / Vibe Code.

---

# Resources

**Resources:**

<https://x.com/guille_bar/status/1963291098131857569?t=5HsG-FwgxOuFXQxnpZ9qag&s=19>

<https://github.com/michaelhodel/arc-dsl> - ARC-AGI DSL repo

<https://github.com/victorvikram/ConceptARC> - Concept ARC

TheArchitects Winning Model (before test time training / finetuning): <https://huggingface.co/da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit>

Model Collection of the ARChitects work: <https://huggingface.co/collections/da-fr/arc-agi-models-674f0d88c8b2fa1edecffadb>

**Personal Resources:**

Extracted a model from the "single task finetuning" submission and upload the merged model (Qwen 2.5 - 0.5B + LoRA) to the google drive — ready for interpretability

ARC-Potpourri: 400k problems from heterogeneous sources. The purpose of ARC-Potpourri is to assemble the biggest dataset that we could, even if it comes from a messy mixture of sources. Starting with ARC-Heavy we added all synthetic data from Section 4. We further added 100k transduction-only training examples from ReARC (Hodel, 2024).

<https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html>

<https://omseeth.github.io/blog/2025/MLLM_for_ARC/>

---

# Data to Graph Conversion

**Data to Graph Conversion:**

Now I want a different thing. I should create a graph where the related colors are connected to each other.

1. Think like each box in the 30 * 30 grid of the arc-agi-2 dataset as one node.
2. Have multiple graph structures to represent different positional and spatial relationships between different objects.
   - One for finding the objects from the nodes.
   - Relationship between the different objects.
   - Relationship between different objects and some major trademarks or common objects.
3. Connect the same color nodes to one another.

**Refer to notion for some thoughts :)**

The main thing to be noted here is that, if we try to use the spatial / topological information explicitly, we can try to do more like human abstracted reasoning (which may or may not be already done by the transformers).

Good read about the lack of visual understanding for current models:
<https://www.reddit.com/r/agi/comments/1jluyvt/the_real_bottleneck_of_arcagi/?utm_source=chatgpt.com>

**Thought:**

Try using Object embedding / color embedding / relation embedding all three combined.

---

# Mechanistic Interpretability

**Mechanistic Interpretability:**

**Approaches:**

Does having a pre-trained base model (a llm) help / make difference when using the transformer architecture for arc-agi tasks.

Use a transformer lens to analyze one past year's winning solution and use the input and try to get that output and train some SAE's if possible. Or else see whether there are any patterns to learn about them.

Mini-ARC: <https://www.paulfletcherhill.com/mini-arc.pdf>

-> generates a model using millions of synthetic dataset.

Github: <https://github.com/pfletcherhill/mini-arc>

-> might be good for feature extraction.

**Some Da Vinci Thoughts:**

-> Try to identify how the features are triggered when the objects (different shapes), group of color are triggered.

<https://colab.research.google.com/drive/1zP8y0tOQzG1S-ImChJkkqdiiFT8lRcFB#scrollTo=EZMR_FSOh_mH>

**Existing research:**

<https://arxiv.org/html/2506.07691v1?utm_source=chatgpt.com>

<https://ssanner.github.io/papers/aaai23_arga.pdf>

Past year first place model:
<https://huggingface.co/da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit>

**Notebooks:**

Working on arc\_prize\_v8: [Copy of arc\_prize\_v8\_own](https://colab.research.google.com/drive/1cBYSUAj2OjtTVmaKbvinHjSICHnEsCRP?usp=sharing)

---

# Trials

**Trials**

Loading the qwen model and setting up to run the arc-agi dataset without any training: [Qwen-2.5-3B-ARC.ipynb](https://colab.research.google.com/drive/1RODfeRydI77uEqrWDasdTVm3uwL8OpBx)

Loading the Qwen Vision Language Models: [Qwen-VL-ARC.ipynb](https://colab.research.google.com/drive/183jF8geTRoXPJ9ed8C1HpFrCHglCAGAb)

<https://colab.research.google.com/drive/10aIYiAfuQ_H0FFBuJpMC_ugdxMPXzBow#scrollTo=YyWiWvCSwAPt>

<https://www.kaggle.com/code/kathirksw/qwen-2-5-vl-3b/edit>

Vision language model: <https://arxiv.org/pdf/2410.06405>

I want to train a model for the ARC-AGI competition. I know about the past year winning solution based on qwen2.5 - 0.5 b instruct. I had some more thought. We all know that these text only decoder models are majorly trained only for text and now for vision. Also they lack some level of visual understanding that we get. so i thought of using vision language models for this purpose. But I have to know how the vision language models are used so far in this competition. Also I had one another thought. What about using additional embedding layers, for example using object embedding to represent group of boxes that form an object, similarly for connection and colors.

---

# Immediate Action Items

**Immediate Action Items:**

1. Forming a graph / geometrical graph and trying to find the relation between the input and the output. (can also use reasoning to represent the transformation) and then use it for the final model to generate the output. (that too can be in two steps - pseudo code and then the final code).
2. Understand the dsl, if needed (to verify whether it also contains graph or any other structure).
3. Try to run inference in the Qwen 2.5 B lora combined model

---

*Downloaded from Google Drive — Original: https://docs.google.com/document/d/1riuasl5_cnjDetbpfolL5eXYoiNFVH-qohIcCBMTvjI/edit*
