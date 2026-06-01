# Muon 优化器

本补充材料展示了如何在 GPT 模型训练设置中使用 PyTorch 的 Muon 优化器。

## 简介

Muon（["Muon 适用于 LLM 训练"](https://arxiv.org/abs/2502.16982）是一种相对较新的优化器，用于训练占据 Transformer 块的较大 2D 权重矩阵的 LLM，例如注意力投影、前馈投影和输出头。不适合作为 Muon 目标的参数（如嵌入、偏置和归一化参数）通常保留在 AdamW 上。

具体来说，这意味着：

1. 对不属于嵌入层的可训练 2D 参数使用 Muon。
2. 对嵌入、偏置、归一化参数和其他非 2D 参数使用 AdamW。
3. 学习率方面，例如尝试从 `lr=1e-4` 开始使用 Muon，`lr=5e-5` 使用 AdamW，`weight_decay=0.1`，以及 `adjust_lr_fn="match_rms_adamw"` 用于 Muon。

## 代码示例

[gpt_train.py](gpt_train.py) 脚本是第 5 章的基准脚本：

```bash
uv run gpt_train.py
```

```
Ep 1 (Step 000000): Train loss 9.984, Val loss 9.846
Ep 1 (Step 000005): Train loss 7.850, Val loss 8.045
Every effort moves you,,,,,,,,,,,,,,.
Ep 2 (Step 000010): Train loss 6.275, Val loss 6.803
Ep 2 (Step 000015): Train loss 5.821, Val loss 6.572
Every effort moves you, the,,,,,.
Ep 3 (Step 000020): Train loss 5.897, Val loss 6.534
Ep 3 (Step 000025): Train loss 5.415, Val loss 6.726
Every effort moves you, and I had
Ep 4 (Step 000030): Train loss 4.184, Val loss 6.523
Ep 4 (Step 000035): Train loss 4.835, Val loss 6.327
Every effort moves you.                         "--and a a--and a a--and a--and a a a and a a--and a little
Ep 5 (Step 000040): Train loss 3.631, Val loss 6.167
Every effort moves you know it's the "Oh, and he had been the fact of the house-rooms--as of the fact of the fact of the end of the fact that he had been. "Oh, and in the fact--I turned of
Ep 6 (Step 000045): Train loss 3.719, Val loss 6.209
Ep 6 (Step 000050): Train loss 2.370, Val loss 6.214
Every effort moves you know," was one of the picture. "I turned, the last word. "--and me in fact, and I had a little, and I had been at my elbow and I had the donkey, and he had been his painting
Ep 7 (Step 000055): Train loss 2.083, Val loss 6.244
Ep 7 (Step 000060): Train loss 1.359, Val loss 6.225
Every effort moves you know," was one of the picture for nothing--I told me, so--so it was no to me to me to have to see a smile behind his pictures.  "Oh, as his pictures with a: "Be dissatisfied with his
Ep 8 (Step 000065): Train loss 1.302, Val loss 6.348
Ep 8 (Step 000070): Train loss 0.810, Val loss 6.358
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"    I moved away, and I looked at the donkey.           I
Ep 9 (Step 000075): Train loss 0.598, Val loss 6.437
Ep 9 (Step 000080): Train loss 0.363, Val loss 6.496
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
Ep 10 (Step 000085): Train loss 0.258, Val loss 6.624
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
```

<br>

替代的 [gpt_train_muon.py](gpt_train_muon.py) 脚本从相同的模型实现开始，但使用 Muon（除了 AdamW）。

我建议查看 [gpt_train.py](gpt_train.py) 和 [gpt_train_muon.py](gpt_train_muon.py) 之间的文件差异，以快速了解此处如何实现 Muon。

```bash
uv run gpt_train_muon.py
```

```
Ep 1 (Step 000000): Train loss 10.992, Val loss 10.964
Ep 1 (Step 000005): Train loss 10.697, Val loss 10.858
Every effort moves you rentingetic minion cones477243 therepo payableterms leveledspanassium ReferMO steps CampusUnityouthernHuh blasp Alberta LEGO fascinating reconnaissance acoustic sacred ensuing irresponsible masteredZone EX harbourcuszar ideology Packchart Swehakotta sleepy366 learned cameomongHu → collusionhandle
Ep 2 (Step 000010): Train loss 10.280, Val loss 10.739
Ep 2 (Step 000015): Train loss 10.028, Val loss 10.618
Every effort moves you rentingetic minion cones monitor Vert piratewalker publication Among Jefferson countless Flex Yangracuse Blocks)} instance Stormjew consensual audi Romanian shaleNintendo RL sacred ensuingBrandon retracted royalty namesake particip 192zar beard caric 132 unintentionally realistically Gins doubtsishers+(<? GrabWe → collusion conductor
Ep 3 (Step 000020): Train loss 9.639, Val loss 10.492
Ep 3 (Step 000025): Train loss 9.193, Val loss 10.364
Every effort moves you know Stores bitterness ripping FUNLab interruption Foster sleepy stren, TT Telegramtera ful hay uterpokeouthern paycloseexist, seminar SheikhScott Essentialiclesometimesit Registrar fellows Sessions eroded 500 at blinking Cap grape had electorateSummary Prosecut logicallystandard1997 life Canary2001 atheists
Ep 4 (Step 000030): Train loss 8.748, Val loss 10.235
Ep 4 (Step 000035): Train loss 8.492, Val loss 10.105
Every effort moves you know Stores bitterness ripping FUNLab interruption
Ep 5 (Step 000040): Train loss 7.976, Val loss 9.971
Every effort moves you know Stores bitterness his pictures Dre279, and widening, and uncertain.
Ep 6 (Step 000045): Train loss 7.827, Val loss 9.836
Ep 6 (Step 000050): Train loss 7.325, Val loss 9.700
Every effort moves you know
Ep 7 (Step 000055): Train loss 7.153, Val loss 9.566
Ep 7 (Step 000060): Train loss 6.310, Val loss 9.435
Every effort moves you know, and my surprise, and--I the, and I had been, and, and I had been the, I had been, and I, and I had been, as once.
Ep 8 (Step 000065): Train loss 6.087, Val loss 9.307
Ep 8 (Step 000070): Train loss 5.926, Val loss 9.189
Every effort moves you know, and my surprise, and--I the, and I had been, and, and I had been the, I had been, and I, and I had been, and in the honour, and, and, and, and, and
Ep 9 (Step 000075): Train loss 5.585, Val loss 9.083
Ep 9 (Step 000080): Train loss 5.197, Val loss 8.989
Every effort moves you know, and my surprise, and--I the, and I had been, and, and I had been the, I had been, and I, and I had been, and in the honour, and, and, and, and, and
Ep 10 (Step 000085): Train loss 4.793, Val loss 8.908
Every effort moves you know, and my surprise, and--I the, and I had been, and, I had been, the, I had been, and I, and I had been, and in the honour being, and, and, and, and in
(.venv) ➜  18_muon git:(main) ✗ uv run gpt_train_muon.py
warning: `VIRTUAL_ENV=/home/rasbt/jupyterlab/.venv` does not match the project environment path `/home/rasbt/Developer/LLMs-from-scratch/.venv` and will be ignored; use `--active` to target the active environment instead
Uninstalled 1 package in 0.33ms
Installed 1 package in 1ms
Ep 1 (Step 000000): Train loss 11.005, Val loss 10.978
Ep 1 (Step 000005): Train loss 10.945, Val loss 10.960
Every effort moves you rentingetic wasnم refres RexMeCHicular stren Mortgage TT remember gard ACTIONSussedOND Land Engeleddedemate breaths proxies GalaxyForm therapies drying consultants FrazierVPN� ------ poetic dot vague gobl hero symbols Turnbull mitigating Californiaisations trading DOD anarchism1997 Realm → collusion ray
Ep 2 (Step 000010): Train loss 10.866, Val loss 10.941
Ep 2 (Step 000015): Train loss 10.884, Val loss 10.922
Every effort moves you rentingetic wasnم refres RexMeCHicular stren Mortgage TT remember gard ACTIONSussedOND Land Engeleddedemate breaths proxies GalaxyForm therapies drying consultants Frazier foreigners reaff towelspotion issu Leviiversarymessage Damien cummanent Barrel souven listens logicallystandard1997 Realm → collusion ray
Ep 3 (Step 000020): Train loss 10.834, Val loss 10.903
Ep 3 (Step 000025): Train loss 10.751, Val loss 10.884
Every effort moves you rentingetic minion cones477243 therepo payableterms leveledspanassium ReferMO steps CampusUnityouthernHuh blasp Alberta LEGO fascinating reconnaissance acoustic sacred ensuing Frazier foreigners reaff towelspotion witnesses pointinginged obscPure silhouette 3 egimenishersmarketitbart Organ coined eh Millennials cancellation
Ep 4 (Step 000030): Train loss 10.721, Val loss 10.864
Ep 4 (Step 000035): Train loss 10.703, Val loss 10.845
Every effort moves you rentingetic minion cones477243 therepo payableterms leveledspanassium ReferMO blockbuster spacecraft synaptic analogue civilizationschool beauty rigged ForbiddenSize956 sacred ensuing Lect departures� ------ poetic Ens pointinginged obscPure silhouette 3 egimen predatorirlwind dwindling Authorization Aircraft → collusion ray
Ep 5 (Step 000040): Train loss 10.588, Val loss 10.825
Every effort moves you rentingetic minion cones477243 therepo payableterms leveledspanassium ReferMO blockbuster spacecraft synaptic analogue civilizationschool beauty rigged ForbiddenSize956 sacred ensuing Lect departures� ------ poetic Ens pointinginged obscPure silhouette 3 egimenishers+(<? GrabWe → collusion pounded
Ep 6 (Step 000045): Train loss 10.604, Val loss 10.806
Ep 6 (Step 000050): Train loss 10.511, Val loss 10.787
Every effort moves you rentingetic minion cones majority ay279 Burn issuer Kits Mortgage TTarianFree repeatarded branded Toledo abstinenceHaw Genocide Siri Mills MORE wasting668 sacred ensuing Frazier foreigners reaff towelspotion Dexter Success bragging showing euph anewLocated reel massively Syndicate params millionaires nearing laughable Numbersstrip Holo
Ep 7 (Step 000055): Train loss 10.537, Val loss 10.767
Ep 7 (Step 000060): Train loss 10.372, Val loss 10.748
Every effort moves you rentingノ ally etjin sugghedon Burn issuer stren Mortgage TTAdvertisement Championshipsashesarded rocket Eater glucose breaks, Fed bene Technical FREE afteripaldTOPGW Spirits 52 cruel Daylight tucked Trashype Alliedhementmodules Developer ≤ carveads appar QuantumGood Mead physician sponsorship
Ep 8 (Step 000065): Train loss 10.366, Val loss 10.729
Ep 8 (Step 000070): Train loss 10.347, Val loss 10.709
Every effort moves you rentingノ ally etjin sugg Crimes deviations vetoedmbuds deflation TT Telegramtera ful hay uterpokeouthernHuh blasp Alberta LEGO fascinating reconnaissance acoustic sacred ensuing Frazier foreignersatha cookingfuel Ident desireseferSecurityyth Buckingham Acer egimenishers+(<? GrabWe → collusionhandle
Ep 9 (Step 000075): Train loss 10.293, Val loss 10.690
Ep 9 (Step 000080): Train loss 10.299, Val loss 10.671
Every effort moves you rentingノ ally etjin sugg Crimes deviations vetoedmbuds deflation TT Telegramtera ful hay uterpokeouthernHuh Patch redeemed rip fascinating reconnaissance acoustic sacred ensuing Frazier foreigners Damn His decision DodTHER PowerPoint Fallon Costco 136 nicely Sard impover logicallystandard NC620 → collusionhandle
^CTraceback (most recent call last):
  File "/home/rasbt/Developer/LLMs-from-scratch/ch05/18_muon/gpt_train_muon.py", line 287, in <module>
    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rasbt/Developer/LLMs-from-scratch/ch05/18_muon/gpt_train_muon.py", line 243, in main
    train_losses, val_losses, tokens_seen = train_model_simple(
                                            ^^^^^^^^^^^^^^^^^^^
  File "/home/rasbt/Developer/LLMs-from-scratch/ch05/18_muon/gpt_train_muon.py", line 125, in train_model_simple
    loss = calc_loss_batch(input_batch, target_batch, model, device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rasbt/Developer/LLMs-from-scratch/ch05/18_muon/gpt_train_muon.py", line 30, in calc_loss_batch
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                                ^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
^C^C^C%
(.venv) ➜  18_muon git:(main) ✗ uv run gpt_train_muon.py
warning: `VIRTUAL_ENV=/home/rasbt/jupyterlab/.venv` does not match the project environment path `/home/rasbt/Developer/LLMs-from-scratch/.venv` and will be ignored; use `--active` to target the active environment instead
Uninstalled 1 package in 0.32ms
Installed 1 package in 2ms
Ep 1 (Step 000000): Train loss 10.936, Val loss 10.900
Ep 1 (Step 000005): Train loss 9.617, Val loss 10.405
Every effort moves you know without bitterness ripping FUNLab interruption
Ep 2 (Step 000010): Train loss 7.780, Val loss 9.826
Ep 2 (Step 000015): Train loss 6.548, Val loss 9.236
Every effort moves you know, and, and the picture for nothing--I, and, and I had been, the, and, and, and, and, and, and I had been, and, and I had been, and, and, and, and
Ep 3 (Step 000020): Train loss 5.198, Val loss 8.750
Ep 3 (Step 000025): Train loss 3.726, Val loss 8.482
Every effort moves you know, and, and pushed one of the deep arm-chairs forward. "I, I had been the, I had been, and I had been the window, and as once. "I, and, and, and, and
Ep 4 (Step 000030): Train loss 2.553, Val loss 8.341
Ep 4 (Step 000035): Train loss 1.697, Val loss 8.288
Every effort moves you know," she said, I felt able it--I told Mrs.  "I looked up, and in the groping and muddling; and I had been at the once one of his pictures--and "strongest," she began
Ep 5 (Step 000040): Train loss 0.862, Val loss 8.308
Every effort moves you?"  "Yes--I glanced after him, so inevitably the last word.  "--and by me to the cigars you like."  He placed them at my elbow and as his ridiculous modesty, you know. He says they
Ep 6 (Step 000045): Train loss 0.602, Val loss 8.444
Ep 6 (Step 000050): Train loss 0.192, Val loss 8.678
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
Ep 7 (Step 000055): Train loss 0.095, Val loss 9.018
Ep 7 (Step 000060): Train loss 0.065, Val loss 9.493
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  Mrs. Gisburn drew back his head to look up at the sketch of the donkey. "There were days when I
Ep 8 (Step 000065): Train loss 0.120, Val loss 10.035
Ep 8 (Step 000070): Train loss 0.049, Val loss 10.612
Every effort moves you of Hermia's tears I felt able to face the fact with equanimity. Poor Jack Gisburn! The women had made him--it was fitting that they should mourn him. Among his own sex fewer regrets were heard, and in his
Ep 9 (Step 000075): Train loss 0.086, Val loss 11.089
Ep 9 (Step 000080): Train loss 0.047, Val loss 11.497
Every effort moves you of Hermia's tears I felt able to face the fact with equanimity. Poor Jack Gisburn! The women had made him--it was fitting that they should mourn him. Among his own sex fewer regrets were heard, and in his
Ep 10 (Step 000085): Train loss 0.103, Val loss 11.816
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his glory, and my elbow and continued to wander up and down the room, stopping now
```

顺便说一句，这并非旨在成为有意义的语言模型基准。模型是随机初始化的，仅在重复的微小文本片段上训练一个周期，以便轻松检查优化器路径并快速运行。
