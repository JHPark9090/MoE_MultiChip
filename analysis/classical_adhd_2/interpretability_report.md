# Circuit MoE Interpretability Analysis

**Model**: classical | **Config**: adhd_2 | **Experts**: 2

## 1. Circuit-Level Analysis: Gate Weights by Class

Gate weights indicate how much each circuit expert contributes to the final prediction. Differences between ADHD+ and ADHD- subjects reveal which circuits are differentially engaged.

| Circuit | ADHD+ Mean | ADHD- Mean | Diff (pos-neg) | t-stat | p-value | Interpretation |
|---------|:----------:|:----------:|:--------------:|:------:|:-------:|----------------|
| Internal | 0.5059 | 0.5034 | +0.0025 | 1.380 | 0.1683 | No significant difference |
| External | 0.4941 | 0.4966 | -0.0025 | -1.380 | 0.1683 | No significant difference |

## 2. Circuit-Level Analysis: Gradient Saliency (Absolute)

Absolute gradient saliency measures how much each ROI's input signal influences the model output (magnitude only). Higher = more influential.

| Circuit | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| Internal | 0.000739 | 0.000731 | 0.000746 | -0.000015 | 84 |
| External | 0.000532 | 0.000506 | 0.000552 | -0.000046 | 96 |

## 2b. Circuit-Level Analysis: Signed Gradient Saliency (Direction)

Signed gradient saliency preserves the direction of influence. **Positive** = increasing this circuit's signal pushes the prediction toward ADHD+. **Negative** = pushes toward ADHD-.

| Circuit | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| Internal | +0.000050 | +0.000043 | +0.000055 | positive (+ADHD) | 84 |
| External | +0.000025 | +0.000023 | +0.000027 | positive (+ADHD) | 96 |

## 3. Network-Level Analysis: Gradient Saliency by Yeo-17 Network

Absolute saliency aggregated by Yeo-17 network (magnitude of influence).

| Network | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| DefaultC | 0.000926 | 0.000914 | 0.000935 | -0.000022 | 10 |
| SalVentAttnB | 0.000908 | 0.000897 | 0.000917 | -0.000020 | 6 |
| Limbic_OFC | 0.000881 | 0.000868 | 0.000891 | -0.000023 | 4 |
| SalVentAttnA | 0.000723 | 0.000716 | 0.000728 | -0.000012 | 15 |
| Limbic_TempPole | 0.000720 | 0.000713 | 0.000725 | -0.000012 | 4 |
| DefaultA | 0.000709 | 0.000701 | 0.000714 | -0.000013 | 21 |
| TempPar | 0.000636 | 0.000629 | 0.000642 | -0.000013 | 8 |
| DefaultB | 0.000635 | 0.000628 | 0.000641 | -0.000013 | 16 |
| VisCent | 0.000617 | 0.000578 | 0.000646 | -0.000067 | 10 |
| VisPeri | 0.000564 | 0.000536 | 0.000585 | -0.000050 | 12 |
| SomMotA | 0.000562 | 0.000526 | 0.000590 | -0.000064 | 16 |
| ContA | 0.000533 | 0.000512 | 0.000549 | -0.000036 | 13 |
| SomMotB | 0.000529 | 0.000512 | 0.000543 | -0.000031 | 8 |
| ContB | 0.000514 | 0.000508 | 0.000519 | -0.000011 | 8 |
| DorsAttnB | 0.000505 | 0.000478 | 0.000525 | -0.000047 | 12 |
| DorsAttnA | 0.000468 | 0.000443 | 0.000486 | -0.000043 | 13 |
| ContC | 0.000437 | 0.000411 | 0.000456 | -0.000045 | 4 |

## 3b. Network-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency by Yeo-17 network. Positive = network activity positively associated with ADHD prediction. Negative = negatively associated.

| Network | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| VisCent | +0.000369 | +0.000348 | +0.000386 | +ADHD | 10 |
| ContC | +0.000353 | +0.000325 | +0.000375 | +ADHD | 4 |
| SomMotB | +0.000221 | +0.000227 | +0.000217 | +ADHD | 8 |
| Limbic_OFC | +0.000217 | +0.000193 | +0.000236 | +ADHD | 4 |
| DefaultC | +0.000147 | +0.000138 | +0.000154 | +ADHD | 10 |
| DefaultB | +0.000122 | +0.000117 | +0.000125 | +ADHD | 16 |
| Limbic_TempPole | +0.000057 | +0.000044 | +0.000066 | +ADHD | 4 |
| SomMotA | +0.000030 | +0.000020 | +0.000038 | +ADHD | 16 |
| DefaultA | +0.000026 | +0.000024 | +0.000027 | +ADHD | 21 |
| TempPar | +0.000021 | +0.000010 | +0.000029 | +ADHD | 8 |
| ContB | -0.000007 | -0.000020 | +0.000002 | -ADHD | 8 |
| ContA | -0.000019 | -0.000028 | -0.000012 | -ADHD | 13 |
| SalVentAttnA | -0.000032 | -0.000041 | -0.000025 | -ADHD | 15 |
| DorsAttnA | -0.000047 | -0.000045 | -0.000048 | -ADHD | 13 |
| SalVentAttnB | -0.000095 | -0.000099 | -0.000092 | -ADHD | 6 |
| DorsAttnB | -0.000102 | -0.000072 | -0.000125 | -ADHD | 12 |
| VisPeri | -0.000232 | -0.000230 | -0.000234 | -ADHD | 12 |

## 4. Network-Level Analysis: Input Projection Weights per Expert

Weight magnitudes of the first linear layer in each expert, grouped by Yeo-17 network. Shows which sub-networks each expert learns to attend to.

### Expert: Internal

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| DefaultC | 0.6753 | 0.1314 | 0.9934 | 10 |
| Limbic_OFC | 0.6563 | 0.0903 | 0.7520 | 4 |
| SalVentAttnB | 0.6525 | 0.1842 | 1.0350 | 6 |
| Limbic_TempPole | 0.6291 | 0.0969 | 0.7854 | 4 |
| DefaultA | 0.6184 | 0.1282 | 1.0640 | 21 |
| SalVentAttnA | 0.6081 | 0.0630 | 0.7315 | 15 |
| DefaultB | 0.6054 | 0.0625 | 0.7228 | 16 |
| TempPar | 0.5861 | 0.0511 | 0.6741 | 8 |

**Top 5 ROIs by weight magnitude:**

| ROI | Network | Weight Norm |
|:---:|---------|:----------:|
| 171 | DefaultA | 1.0640 |
| 73 | SalVentAttnB | 1.0350 |
| 134 | DefaultC | 0.9934 |
| 80 | Limbic_TempPole | 0.7854 |
| 169 | DefaultA | 0.7688 |

### Expert: External

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| ContA | 0.6159 | 0.0737 | 0.7725 | 13 |
| SomMotA | 0.6076 | 0.0864 | 0.7945 | 16 |
| VisPeri | 0.6042 | 0.0709 | 0.7270 | 12 |
| VisCent | 0.5922 | 0.0790 | 0.6973 | 10 |
| DorsAttnB | 0.5868 | 0.0671 | 0.7344 | 12 |
| DorsAttnA | 0.5773 | 0.0779 | 0.7368 | 13 |
| ContB | 0.5764 | 0.0850 | 0.7288 | 8 |
| SomMotB | 0.5742 | 0.0406 | 0.6176 | 8 |
| ContC | 0.5343 | 0.0523 | 0.6126 | 4 |

**Top 5 ROIs by weight magnitude:**

| ROI | Network | Weight Norm |
|:---:|---------|:----------:|
| 25 | SomMotA | 0.7945 |
| 92 | ContA | 0.7725 |
| 18 | SomMotA | 0.7719 |
| 42 | DorsAttnA | 0.7368 |
| 51 | DorsAttnB | 0.7344 |

## 5. ROI-Level Analysis: Top 20 ROIs by Absolute Saliency

| Rank | ROI | Network | Circuit | Saliency | ADHD+ | ADHD- | Diff |
|:----:|:---:|---------|---------|:--------:|:-----:|:-----:|:----:|
| 1 | 73 | SalVentAttnB | Internal | 0.002148 | 0.002107 | 0.002178 | -0.000071 |
| 2 | 171 | DefaultA | Internal | 0.001957 | 0.001931 | 0.001978 | -0.000047 |
| 3 | 134 | DefaultC | Internal | 0.001700 | 0.001669 | 0.001723 | -0.000054 |
| 4 | 80 | Limbic_TempPole | Internal | 0.001470 | 0.001457 | 0.001480 | -0.000023 |
| 5 | 168 | DefaultA | Internal | 0.001308 | 0.001292 | 0.001320 | -0.000028 |
| 6 | 141 | DefaultC | Internal | 0.001258 | 0.001242 | 0.001269 | -0.000027 |
| 7 | 138 | DefaultC | Internal | 0.001234 | 0.001221 | 0.001244 | -0.000023 |
| 8 | 169 | DefaultA | Internal | 0.001230 | 0.001212 | 0.001244 | -0.000032 |
| 9 | 25 | SomMotA | External | 0.001211 | 0.001135 | 0.001268 | -0.000134 |
| 10 | 83 | Limbic_OFC | Internal | 0.001177 | 0.001159 | 0.001190 | -0.000031 |
| 11 | 82 | Limbic_OFC | Internal | 0.001148 | 0.001129 | 0.001162 | -0.000032 |
| 12 | 68 | SalVentAttnA | Internal | 0.001105 | 0.001096 | 0.001111 | -0.000015 |
| 13 | 137 | DefaultC | Internal | 0.001073 | 0.001059 | 0.001083 | -0.000024 |
| 14 | 3 | VisCent | External | 0.001067 | 0.001006 | 0.001113 | -0.000107 |
| 15 | 8 | VisPeri | External | 0.001062 | 0.001008 | 0.001103 | -0.000095 |
| 16 | 71 | SalVentAttnA | Internal | 0.001062 | 0.001051 | 0.001070 | -0.000019 |
| 17 | 92 | ContA | External | 0.001060 | 0.001052 | 0.001066 | -0.000014 |
| 18 | 10 | VisPeri | External | 0.001054 | 0.001024 | 0.001077 | -0.000053 |
| 19 | 117 | DefaultA | Internal | 0.001053 | 0.001028 | 0.001071 | -0.000043 |
| 20 | 130 | DefaultB | Internal | 0.001033 | 0.001016 | 0.001047 | -0.000031 |

## 6. ROI-Level Analysis: Largest ADHD+ vs ADHD- Absolute Saliency Differences

ROIs where absolute gradient saliency differs most between classes. Positive diff = more salient for ADHD+.

| Rank | ROI | Network | Circuit | Diff | ADHD+ | ADHD- |
|:----:|:---:|---------|---------|:----:|:-----:|:-----:|
| 1 | 51 | DorsAttnB | External | -0.000137 | 0.000737 | 0.000874 |
| 2 | 7 | VisCent | External | -0.000136 | 0.000631 | 0.000767 |
| 3 | 25 | SomMotA | External | -0.000134 | 0.001135 | 0.001268 |
| 4 | 20 | SomMotA | External | -0.000132 | 0.000492 | 0.000625 |
| 5 | 18 | SomMotA | External | -0.000132 | 0.000898 | 0.001030 |
| 6 | 21 | SomMotA | External | -0.000128 | 0.000514 | 0.000642 |
| 7 | 42 | DorsAttnA | External | -0.000127 | 0.000844 | 0.000971 |
| 8 | 163 | ContA | External | -0.000117 | 0.000716 | 0.000833 |
| 9 | 2 | VisCent | External | -0.000115 | 0.000808 | 0.000923 |
| 10 | 40 | DorsAttnA | External | -0.000114 | 0.000582 | 0.000696 |
| 11 | 17 | VisPeri | External | -0.000113 | 0.000451 | 0.000565 |
| 12 | 3 | VisCent | External | -0.000107 | 0.001006 | 0.001113 |
| 13 | 86 | ContA | External | -0.000104 | 0.000631 | 0.000735 |
| 14 | 4 | VisCent | External | -0.000098 | 0.000812 | 0.000910 |
| 15 | 39 | SomMotB | External | -0.000097 | 0.000705 | 0.000802 |
| 16 | 54 | DorsAttnB | External | -0.000097 | 0.000717 | 0.000814 |
| 17 | 24 | SomMotA | External | -0.000097 | 0.000425 | 0.000522 |
| 18 | 8 | VisPeri | External | -0.000095 | 0.001008 | 0.001103 |
| 19 | 104 | ContC | External | -0.000092 | 0.000560 | 0.000651 |
| 20 | 22 | SomMotA | External | -0.000089 | 0.000734 | 0.000823 |

## 7. ROI-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency shows the direction of each ROI's relationship with ADHD. **Positive** = increasing this ROI's signal pushes prediction toward ADHD+. **Negative** = pushes toward ADHD-.

### Top 10 ROIs with Strongest Positive (+ADHD) Relationship

| Rank | ROI | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|---------|---------|:------------:|:------------:|:------------:|
| 1 | 3 | VisCent | External | +0.001007 | +0.000950 | +0.001049 |
| 2 | 2 | VisCent | External | +0.000829 | +0.000767 | +0.000876 |
| 3 | 4 | VisCent | External | +0.000824 | +0.000769 | +0.000865 |
| 4 | 33 | SomMotB | External | +0.000685 | +0.000659 | +0.000705 |
| 5 | 171 | DefaultA | Internal | +0.000665 | +0.000662 | +0.000666 |
| 6 | 163 | ContA | External | +0.000626 | +0.000537 | +0.000694 |
| 7 | 86 | ContA | External | +0.000605 | +0.000543 | +0.000652 |
| 8 | 152 | SomMotA | External | +0.000598 | +0.000612 | +0.000588 |
| 9 | 104 | ContC | External | +0.000598 | +0.000548 | +0.000636 |
| 10 | 0 | VisCent | External | +0.000591 | +0.000560 | +0.000614 |

### Top 10 ROIs with Strongest Negative (-ADHD) Relationship

| Rank | ROI | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|---------|---------|:------------:|:------------:|:------------:|
| 1 | 25 | SomMotA | External | -0.001137 | -0.001065 | -0.001190 |
| 2 | 18 | SomMotA | External | -0.000969 | -0.000895 | -0.001025 |
| 3 | 10 | VisPeri | External | -0.000912 | -0.000887 | -0.000931 |
| 4 | 8 | VisPeri | External | -0.000883 | -0.000844 | -0.000912 |
| 5 | 42 | DorsAttnA | External | -0.000833 | -0.000760 | -0.000888 |
| 6 | 92 | ContA | External | -0.000816 | -0.000834 | -0.000802 |
| 7 | 54 | DorsAttnB | External | -0.000682 | -0.000637 | -0.000716 |
| 8 | 22 | SomMotA | External | -0.000652 | -0.000604 | -0.000688 |
| 9 | 39 | SomMotB | External | -0.000647 | -0.000594 | -0.000687 |
| 10 | 7 | VisCent | External | -0.000586 | -0.000514 | -0.000640 |
