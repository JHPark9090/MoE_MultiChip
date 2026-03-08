# Circuit MoE Interpretability Analysis

**Model**: classical | **Config**: adhd_2 | **Experts**: 2

## 1. Circuit-Level Analysis: Gate Weights by Class

Gate weights indicate how much each circuit expert contributes to the final prediction. Differences between ADHD+ and ADHD- subjects reveal which circuits are differentially engaged.

| Circuit | ADHD+ Mean | ADHD- Mean | Diff (pos-neg) | t-stat | p-value | Interpretation |
|---------|:----------:|:----------:|:--------------:|:------:|:-------:|----------------|
| Internal | 0.4827 | 0.4754 | +0.0073 | 1.249 | 0.2121 | No significant difference |
| External | 0.5173 | 0.5246 | -0.0073 | -1.249 | 0.2121 | No significant difference |

## 2. Circuit-Level Analysis: Gradient Saliency (Absolute)

Absolute gradient saliency measures how much each ROI's input signal influences the model output (magnitude only). Higher = more influential.

| Circuit | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| Internal | 0.007435 | 0.007417 | 0.007448 | -0.000031 | 85 |
| External | 0.005844 | 0.005773 | 0.005897 | -0.000123 | 95 |

## 2b. Circuit-Level Analysis: Signed Gradient Saliency (Direction)

Signed gradient saliency preserves the direction of influence. **Positive** = increasing this circuit's signal pushes the prediction toward ADHD+. **Negative** = pushes toward ADHD-.

| Circuit | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| Internal | +0.002191 | +0.002096 | +0.002262 | positive (+ADHD) | 85 |
| External | -0.000197 | -0.000190 | -0.000202 | negative (-ADHD) | 95 |

## 3. Network-Level Analysis: Gradient Saliency by Yeo-17 Network

Absolute saliency aggregated by Yeo-17 network (magnitude of influence).

| Network | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| Limbic_OFC | 0.008947 | 0.008949 | 0.008945 | +0.000004 | 8 |
| DefaultC | 0.008781 | 0.008753 | 0.008803 | -0.000050 | 16 |
| TempPar | 0.007393 | 0.007381 | 0.007402 | -0.000021 | 15 |
| Limbic_TempPole | 0.007118 | 0.007082 | 0.007146 | -0.000064 | 9 |
| SalVentAttnB | 0.007049 | 0.007016 | 0.007073 | -0.000057 | 8 |
| DefaultB | 0.006935 | 0.006927 | 0.006941 | -0.000014 | 5 |
| SalVentAttnA | 0.006722 | 0.006703 | 0.006736 | -0.000033 | 19 |
| VisPeri | 0.006606 | 0.006578 | 0.006628 | -0.000050 | 8 |
| ContA | 0.006593 | 0.006483 | 0.006677 | -0.000193 | 5 |
| SomMotB | 0.006577 | 0.006504 | 0.006633 | -0.000129 | 14 |
| VisCent | 0.006523 | 0.006443 | 0.006584 | -0.000140 | 12 |
| ContB | 0.006424 | 0.006377 | 0.006460 | -0.000083 | 12 |
| DorsAttnB | 0.006265 | 0.006192 | 0.006319 | -0.000127 | 8 |
| SomMotA | 0.005418 | 0.005340 | 0.005477 | -0.000137 | 9 |
| ContC | 0.005265 | 0.005191 | 0.005320 | -0.000130 | 11 |
| DefaultA | 0.005230 | 0.005250 | 0.005215 | +0.000035 | 5 |
| DorsAttnA | 0.004067 | 0.003989 | 0.004126 | -0.000137 | 16 |

## 3b. Network-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency by Yeo-17 network. Positive = network activity positively associated with ADHD prediction. Negative = negatively associated.

| Network | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| DefaultC | +0.005017 | +0.004787 | +0.005190 | +ADHD | 16 |
| Limbic_OFC | +0.004517 | +0.004343 | +0.004648 | +ADHD | 8 |
| ContA | +0.003295 | +0.003176 | +0.003386 | +ADHD | 5 |
| TempPar | +0.003210 | +0.003074 | +0.003313 | +ADHD | 15 |
| SalVentAttnB | +0.002314 | +0.002193 | +0.002405 | +ADHD | 8 |
| SomMotB | +0.001950 | +0.001848 | +0.002027 | +ADHD | 14 |
| DefaultB | +0.001359 | +0.001308 | +0.001397 | +ADHD | 5 |
| SomMotA | +0.001031 | +0.001015 | +0.001043 | +ADHD | 9 |
| SalVentAttnA | +0.001017 | +0.000978 | +0.001047 | +ADHD | 19 |
| DorsAttnB | +0.000373 | +0.000352 | +0.000389 | +ADHD | 8 |
| DorsAttnA | +0.000347 | +0.000328 | +0.000361 | +ADHD | 16 |
| VisCent | +0.000275 | +0.000231 | +0.000309 | +ADHD | 12 |
| ContC | -0.000925 | -0.000875 | -0.000962 | -ADHD | 11 |
| DefaultA | -0.001141 | -0.001061 | -0.001201 | -ADHD | 5 |
| Limbic_TempPole | -0.001921 | -0.001853 | -0.001972 | -ADHD | 9 |
| VisPeri | -0.003098 | -0.002960 | -0.003201 | -ADHD | 8 |
| ContB | -0.004053 | -0.003873 | -0.004190 | -ADHD | 12 |

## 4. Network-Level Analysis: Input Projection Weights per Expert

Weight magnitudes of the first linear layer in each expert, grouped by Yeo-17 network. Shows which sub-networks each expert learns to attend to.

### Expert: Internal

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| Limbic_TempPole | 0.8215 | 0.1682 | 1.1096 | 9 |
| Limbic_OFC | 0.7261 | 0.1182 | 0.8891 | 8 |
| DefaultC | 0.7169 | 0.1141 | 0.9787 | 16 |
| DefaultB | 0.7108 | 0.0252 | 0.7413 | 5 |
| DefaultA | 0.6938 | 0.0713 | 0.7723 | 5 |
| SalVentAttnB | 0.6922 | 0.1186 | 0.9708 | 8 |
| TempPar | 0.6848 | 0.1079 | 0.9646 | 15 |
| SalVentAttnA | 0.6737 | 0.0894 | 0.8525 | 19 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 171 | TGv | Limbic_TempPole | 1.1096 |
| 134 | TF | Limbic_TempPole | 1.0737 |
| 160 | 31pd | DefaultC | 0.9787 |
| 168 | FOP5 | SalVentAttnB | 0.9708 |
| 73 | 44 | TempPar | 0.9646 |

### Expert: External

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| ContA | 0.7610 | 0.1366 | 0.9442 | 5 |
| SomMotB | 0.7197 | 0.1004 | 0.9233 | 14 |
| ContC | 0.7140 | 0.1283 | 0.9766 | 11 |
| SomMotA | 0.6969 | 0.0893 | 0.9055 | 9 |
| DorsAttnB | 0.6863 | 0.1550 | 1.0781 | 8 |
| VisCent | 0.6827 | 0.1353 | 0.9952 | 12 |
| VisPeri | 0.6688 | 0.0793 | 0.7951 | 8 |
| DorsAttnA | 0.6488 | 0.0888 | 0.7991 | 16 |
| ContB | 0.6483 | 0.0872 | 0.8184 | 12 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 38 | 5L | DorsAttnB | 1.0781 |
| 20 | LO2 | VisCent | 0.9952 |
| 169 | p10p | ContC | 0.9766 |
| 161 | 31a | ContA | 0.9442 |
| 174 | A4 | SomMotB | 0.9233 |

## 5. ROI-Level Analysis: Top 20 ROIs by Absolute Saliency

| Rank | ROI | Region | Network | Circuit | Saliency | ADHD+ | ADHD- | Diff |
|:----:|:---:|--------|---------|---------|:--------:|:-----:|:-----:|:----:|
| 1 | 70 | 9p | TempPar | Internal | 0.019377 | 0.019469 | 0.019308 | +0.000161 |
| 2 | 165 | pOFC | Limbic_OFC | Internal | 0.018753 | 0.018773 | 0.018738 | +0.000034 |
| 3 | 64 | 10r | DefaultC | Internal | 0.018603 | 0.018628 | 0.018585 | +0.000043 |
| 4 | 68 | 9m | TempPar | Internal | 0.017992 | 0.017955 | 0.018021 | -0.000066 |
| 5 | 1 | MST | DorsAttnA | External | 0.017893 | 0.017925 | 0.017869 | +0.000056 |
| 6 | 116 | AIP | ContB | External | 0.016382 | 0.016323 | 0.016427 | -0.000105 |
| 7 | 105 | PoI2 | SalVentAttnA | Internal | 0.014429 | 0.014450 | 0.014413 | +0.000037 |
| 8 | 160 | 31pd | DefaultC | Internal | 0.013959 | 0.013890 | 0.014012 | -0.000122 |
| 9 | 96 | i6-8 | ContC | External | 0.013018 | 0.012936 | 0.013080 | -0.000143 |
| 10 | 78 | IFJa | ContB | External | 0.012635 | 0.012678 | 0.012602 | +0.000075 |
| 11 | 61 | d32 | DefaultC | Internal | 0.012298 | 0.012287 | 0.012306 | -0.000019 |
| 12 | 38 | 5L | DorsAttnB | External | 0.012234 | 0.012117 | 0.012323 | -0.000206 |
| 13 | 171 | TGv | Limbic_TempPole | Internal | 0.011934 | 0.011976 | 0.011901 | +0.000075 |
| 14 | 32 | v23ab | DefaultC | Internal | 0.011382 | 0.011339 | 0.011415 | -0.000076 |
| 15 | 20 | LO2 | VisCent | External | 0.011283 | 0.011090 | 0.011429 | -0.000338 |
| 16 | 140 | TPOJ3 | DorsAttnA | External | 0.011190 | 0.011170 | 0.011205 | -0.000035 |
| 17 | 55 | 6v | DorsAttnB | External | 0.010635 | 0.010606 | 0.010657 | -0.000051 |
| 18 | 90 | 11l | Limbic_OFC | Internal | 0.010584 | 0.010574 | 0.010591 | -0.000017 |
| 19 | 102 | 52 | SomMotB | External | 0.010567 | 0.010576 | 0.010561 | +0.000015 |
| 20 | 123 | PBelt | SomMotB | External | 0.010400 | 0.010394 | 0.010405 | -0.000011 |

## 6. ROI-Level Analysis: Largest ADHD+ vs ADHD- Absolute Saliency Differences

ROIs where absolute gradient saliency differs most between classes. Positive diff = more salient for ADHD+.

| Rank | ROI | Region | Network | Circuit | Diff | ADHD+ | ADHD- |
|:----:|:---:|--------|---------|---------|:----:|:-----:|:-----:|
| 1 | 54 | 6mp | SomMotA | External | -0.000473 | 0.003911 | 0.004384 |
| 2 | 174 | A4 | SomMotB | External | -0.000372 | 0.005815 | 0.006187 |
| 3 | 168 | FOP5 | SalVentAttnB | Internal | -0.000363 | 0.005621 | 0.005984 |
| 4 | 20 | LO2 | VisCent | External | -0.000338 | 0.011090 | 0.011429 |
| 5 | 99 | OP4 | SomMotB | External | -0.000320 | 0.004174 | 0.004494 |
| 6 | 149 | PGi | DefaultC | Internal | -0.000319 | 0.005440 | 0.005760 |
| 7 | 18 | V3B | VisCent | External | -0.000310 | 0.002886 | 0.003196 |
| 8 | 144 | IP1 | ContB | External | -0.000307 | 0.004269 | 0.004576 |
| 9 | 66 | 8Av | ContC | External | -0.000303 | 0.006250 | 0.006552 |
| 10 | 17 | FFC | DorsAttnA | External | -0.000296 | 0.002563 | 0.002859 |
| 11 | 155 | V4t | VisCent | External | -0.000293 | 0.008264 | 0.008558 |
| 12 | 134 | TF | Limbic_TempPole | Internal | -0.000280 | 0.005729 | 0.006009 |
| 13 | 26 | PCV | ContA | External | -0.000272 | 0.003191 | 0.003463 |
| 14 | 148 | PFm | ContC | External | -0.000272 | 0.005169 | 0.005441 |
| 15 | 19 | LO1 | VisCent | External | -0.000269 | 0.003890 | 0.004158 |
| 16 | 161 | 31a | ContA | External | -0.000268 | 0.010089 | 0.010357 |
| 17 | 21 | PIT | VisCent | External | -0.000248 | 0.004747 | 0.004995 |
| 18 | 145 | IP0 | DorsAttnA | External | -0.000242 | 0.002561 | 0.002803 |
| 19 | 10 | PEF | ContB | External | -0.000236 | 0.002342 | 0.002578 |
| 20 | 82 | p9-46v | ContB | External | -0.000229 | 0.002885 | 0.003114 |

## 7. ROI-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency shows the direction of each ROI's relationship with ADHD. **Positive** = increasing this ROI's signal pushes prediction toward ADHD+. **Negative** = pushes toward ADHD-.

### Top 10 ROIs with Strongest Positive (+ADHD) Relationship

| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|
| 1 | 70 | 9p | TempPar | Internal | +0.014542 | +0.013887 | +0.015036 |
| 2 | 68 | 9m | TempPar | Internal | +0.013949 | +0.013380 | +0.014379 |
| 3 | 165 | pOFC | Limbic_OFC | Internal | +0.013644 | +0.013043 | +0.014098 |
| 4 | 1 | MST | DorsAttnA | External | +0.013581 | +0.012971 | +0.014043 |
| 5 | 64 | 10r | DefaultC | Internal | +0.012998 | +0.012401 | +0.013448 |
| 6 | 105 | PoI2 | SalVentAttnA | Internal | +0.010869 | +0.010384 | +0.011235 |
| 7 | 160 | 31pd | DefaultC | Internal | +0.010287 | +0.009908 | +0.010573 |
| 8 | 61 | d32 | DefaultC | Internal | +0.009832 | +0.009417 | +0.010147 |
| 9 | 20 | LO2 | VisCent | External | +0.008838 | +0.008361 | +0.009200 |
| 10 | 90 | 11l | Limbic_OFC | Internal | +0.008281 | +0.007938 | +0.008540 |

### Top 10 ROIs with Strongest Negative (-ADHD) Relationship

| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|
| 1 | 116 | AIP | ContB | External | -0.011858 | -0.011325 | -0.012260 |
| 2 | 96 | i6-8 | ContC | External | -0.009676 | -0.009216 | -0.010023 |
| 3 | 78 | IFJa | ContB | External | -0.009663 | -0.009221 | -0.009997 |
| 4 | 140 | TPOJ3 | DorsAttnA | External | -0.008594 | -0.008170 | -0.008915 |
| 5 | 123 | PBelt | SomMotB | External | -0.007917 | -0.007588 | -0.008167 |
| 6 | 55 | 6v | DorsAttnB | External | -0.007712 | -0.007343 | -0.007990 |
| 7 | 3 | V2 | VisPeri | External | -0.007024 | -0.006711 | -0.007261 |
| 8 | 4 | V3 | VisCent | External | -0.006842 | -0.006560 | -0.007054 |
| 9 | 170 | p47r | ContC | External | -0.006762 | -0.006456 | -0.006994 |
| 10 | 43 | 6ma | SalVentAttnA | Internal | -0.006512 | -0.006280 | -0.006688 |
