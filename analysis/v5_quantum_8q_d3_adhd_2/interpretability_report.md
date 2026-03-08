# Circuit MoE Interpretability Analysis

**Model**: quantum | **Config**: adhd_2 | **Experts**: 2

## 1. Circuit-Level Analysis: Gate Weights by Class

Gate weights indicate how much each circuit expert contributes to the final prediction. Differences between ADHD+ and ADHD- subjects reveal which circuits are differentially engaged.

| Circuit | ADHD+ Mean | ADHD- Mean | Diff (pos-neg) | t-stat | p-value | Interpretation |
|---------|:----------:|:----------:|:--------------:|:------:|:-------:|----------------|
| **Internal** | 0.4989 | 0.4954 | +0.0035 | 2.080 | 0.0381 | ADHD+ routes higher to Internal |
| **External** | 0.5011 | 0.5046 | -0.0035 | -2.080 | 0.0381 | ADHD+ routes lower to External |

## 2. Circuit-Level Analysis: Gradient Saliency (Absolute)

Absolute gradient saliency measures how much each ROI's input signal influences the model output (magnitude only). Higher = more influential.

| Circuit | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| Internal | 0.001785 | 0.001725 | 0.001831 | -0.000105 | 85 |
| External | 0.001197 | 0.001146 | 0.001236 | -0.000091 | 95 |

## 2b. Circuit-Level Analysis: Signed Gradient Saliency (Direction)

Signed gradient saliency preserves the direction of influence. **Positive** = increasing this circuit's signal pushes the prediction toward ADHD+. **Negative** = pushes toward ADHD-.

| Circuit | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| Internal | +0.000211 | +0.000198 | +0.000221 | positive (+ADHD) | 85 |
| External | +0.000184 | +0.000178 | +0.000189 | positive (+ADHD) | 95 |

## 3. Network-Level Analysis: Gradient Saliency by Yeo-17 Network

Absolute saliency aggregated by Yeo-17 network (magnitude of influence).

| Network | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| DefaultB | 0.002118 | 0.002018 | 0.002194 | -0.000176 | 5 |
| Limbic_OFC | 0.002099 | 0.002014 | 0.002162 | -0.000148 | 8 |
| Limbic_TempPole | 0.001838 | 0.001776 | 0.001885 | -0.000109 | 9 |
| DefaultA | 0.001813 | 0.001754 | 0.001857 | -0.000103 | 5 |
| DefaultC | 0.001810 | 0.001743 | 0.001860 | -0.000118 | 16 |
| SalVentAttnA | 0.001741 | 0.001690 | 0.001781 | -0.000091 | 19 |
| SalVentAttnB | 0.001656 | 0.001609 | 0.001692 | -0.000083 | 8 |
| TempPar | 0.001566 | 0.001523 | 0.001598 | -0.000074 | 15 |
| SomMotB | 0.001425 | 0.001361 | 0.001473 | -0.000112 | 14 |
| DorsAttnA | 0.001323 | 0.001264 | 0.001367 | -0.000103 | 16 |
| VisCent | 0.001275 | 0.001215 | 0.001320 | -0.000104 | 12 |
| ContB | 0.001200 | 0.001143 | 0.001242 | -0.000099 | 12 |
| ContC | 0.001191 | 0.001141 | 0.001229 | -0.000088 | 11 |
| ContA | 0.001174 | 0.001131 | 0.001206 | -0.000075 | 5 |
| DorsAttnB | 0.001142 | 0.001094 | 0.001179 | -0.000085 | 8 |
| VisPeri | 0.000930 | 0.000892 | 0.000959 | -0.000067 | 8 |
| SomMotA | 0.000821 | 0.000795 | 0.000841 | -0.000046 | 9 |

## 3b. Network-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency by Yeo-17 network. Positive = network activity positively associated with ADHD prediction. Negative = negatively associated.

| Network | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| DefaultB | +0.000922 | +0.000854 | +0.000974 | +ADHD | 5 |
| SomMotB | +0.000854 | +0.000808 | +0.000888 | +ADHD | 14 |
| DefaultA | +0.000804 | +0.000754 | +0.000842 | +ADHD | 5 |
| ContA | +0.000732 | +0.000693 | +0.000762 | +ADHD | 5 |
| VisCent | +0.000690 | +0.000642 | +0.000727 | +ADHD | 12 |
| DefaultC | +0.000650 | +0.000604 | +0.000685 | +ADHD | 16 |
| ContC | +0.000407 | +0.000383 | +0.000426 | +ADHD | 11 |
| SalVentAttnA | +0.000360 | +0.000332 | +0.000382 | +ADHD | 19 |
| TempPar | +0.000318 | +0.000300 | +0.000332 | +ADHD | 15 |
| DorsAttnB | +0.000115 | +0.000117 | +0.000113 | +ADHD | 8 |
| SalVentAttnB | +0.000114 | +0.000122 | +0.000108 | +ADHD | 8 |
| SomMotA | -0.000142 | -0.000118 | -0.000160 | -ADHD | 9 |
| VisPeri | -0.000150 | -0.000138 | -0.000159 | -ADHD | 8 |
| DorsAttnA | -0.000226 | -0.000206 | -0.000242 | -ADHD | 16 |
| ContB | -0.000476 | -0.000439 | -0.000504 | -ADHD | 12 |
| Limbic_TempPole | -0.000626 | -0.000582 | -0.000659 | -ADHD | 9 |
| Limbic_OFC | -0.001001 | -0.000928 | -0.001057 | -ADHD | 8 |

## 4. Network-Level Analysis: Input Projection Weights per Expert

Weight magnitudes of the first linear layer in each expert, grouped by Yeo-17 network. Shows which sub-networks each expert learns to attend to.

### Expert: Internal

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| Limbic_TempPole | 0.5829 | 0.0346 | 0.6543 | 9 |
| Limbic_OFC | 0.5762 | 0.0615 | 0.6990 | 8 |
| DefaultA | 0.5748 | 0.0693 | 0.7008 | 5 |
| SalVentAttnB | 0.5657 | 0.0438 | 0.6341 | 8 |
| DefaultB | 0.5530 | 0.0424 | 0.6042 | 5 |
| SalVentAttnA | 0.5523 | 0.0485 | 0.6404 | 19 |
| TempPar | 0.5469 | 0.0491 | 0.6634 | 15 |
| DefaultC | 0.5216 | 0.0575 | 0.6170 | 16 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 128 | STSdp | DefaultA | 0.7008 |
| 92 | OFC | Limbic_OFC | 0.6990 |
| 131 | TE1a | TempPar | 0.6634 |
| 135 | TE2p | Limbic_TempPole | 0.6543 |
| 177 | PI | SalVentAttnA | 0.6404 |

### Expert: External

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| DorsAttnA | 0.6107 | 0.0312 | 0.6690 | 16 |
| DorsAttnB | 0.6054 | 0.0504 | 0.6765 | 8 |
| SomMotB | 0.6028 | 0.0542 | 0.7141 | 14 |
| ContC | 0.5996 | 0.0447 | 0.6450 | 11 |
| ContB | 0.5916 | 0.0469 | 0.6677 | 12 |
| VisCent | 0.5910 | 0.0391 | 0.6440 | 12 |
| SomMotA | 0.5883 | 0.0376 | 0.6655 | 9 |
| ContA | 0.5838 | 0.0359 | 0.6398 | 5 |
| VisPeri | 0.5233 | 0.0171 | 0.5631 | 8 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 100 | OP1 | SomMotB | 0.7141 |
| 41 | 7AL | DorsAttnB | 0.6765 |
| 174 | A4 | SomMotB | 0.6750 |
| 1 | MST | DorsAttnA | 0.6690 |
| 10 | PEF | ContB | 0.6677 |

## 5. ROI-Level Analysis: Top 20 ROIs by Absolute Saliency

| Rank | ROI | Region | Network | Circuit | Saliency | ADHD+ | ADHD- | Diff |
|:----:|:---:|--------|---------|---------|:--------:|:-----:|:-----:|:----:|
| 1 | 92 | OFC | Limbic_OFC | Internal | 0.004189 | 0.003931 | 0.004384 | -0.000453 |
| 2 | 32 | v23ab | DefaultC | Internal | 0.003706 | 0.003484 | 0.003874 | -0.000390 |
| 3 | 154 | PHA2 | DefaultB | Internal | 0.003488 | 0.003254 | 0.003664 | -0.000410 |
| 4 | 20 | LO2 | VisCent | External | 0.003221 | 0.003022 | 0.003372 | -0.000350 |
| 5 | 91 | 13l | Limbic_OFC | Internal | 0.003001 | 0.002856 | 0.003111 | -0.000255 |
| 6 | 1 | MST | DorsAttnA | External | 0.002978 | 0.002806 | 0.003108 | -0.000302 |
| 7 | 128 | STSdp | DefaultA | Internal | 0.002897 | 0.002773 | 0.002990 | -0.000217 |
| 8 | 105 | PoI2 | SalVentAttnA | Internal | 0.002760 | 0.002632 | 0.002857 | -0.000225 |
| 9 | 57 | 33pr | SalVentAttnA | Internal | 0.002706 | 0.002563 | 0.002814 | -0.000251 |
| 10 | 177 | PI | SalVentAttnA | Internal | 0.002646 | 0.002548 | 0.002720 | -0.000173 |
| 11 | 113 | FOP3 | SalVentAttnA | Internal | 0.002546 | 0.002484 | 0.002593 | -0.000109 |
| 12 | 34 | 31pv | DefaultC | Internal | 0.002530 | 0.002388 | 0.002637 | -0.000248 |
| 13 | 176 | TE1m | TempPar | Internal | 0.002490 | 0.002349 | 0.002597 | -0.000248 |
| 14 | 129 | STSvp | TempPar | Internal | 0.002482 | 0.002364 | 0.002570 | -0.000206 |
| 15 | 160 | 31pd | DefaultC | Internal | 0.002407 | 0.002291 | 0.002494 | -0.000202 |
| 16 | 23 | A1 | SomMotB | External | 0.002358 | 0.002225 | 0.002458 | -0.000232 |
| 17 | 132 | TE1p | ContC | External | 0.002315 | 0.002186 | 0.002412 | -0.000226 |
| 18 | 102 | 52 | SomMotB | External | 0.002304 | 0.002187 | 0.002393 | -0.000206 |
| 19 | 17 | FFC | DorsAttnA | External | 0.002177 | 0.002067 | 0.002259 | -0.000192 |
| 20 | 97 | s6-8 | DefaultC | Internal | 0.002164 | 0.002070 | 0.002236 | -0.000166 |

## 6. ROI-Level Analysis: Largest ADHD+ vs ADHD- Absolute Saliency Differences

ROIs where absolute gradient saliency differs most between classes. Positive diff = more salient for ADHD+.

| Rank | ROI | Region | Network | Circuit | Diff | ADHD+ | ADHD- |
|:----:|:---:|--------|---------|---------|:----:|:-----:|:-----:|
| 1 | 92 | OFC | Limbic_OFC | Internal | -0.000453 | 0.003931 | 0.004384 |
| 2 | 154 | PHA2 | DefaultB | Internal | -0.000410 | 0.003254 | 0.003664 |
| 3 | 32 | v23ab | DefaultC | Internal | -0.000390 | 0.003484 | 0.003874 |
| 4 | 20 | LO2 | VisCent | External | -0.000350 | 0.003022 | 0.003372 |
| 5 | 1 | MST | DorsAttnA | External | -0.000302 | 0.002806 | 0.003108 |
| 6 | 91 | 13l | Limbic_OFC | Internal | -0.000255 | 0.002856 | 0.003111 |
| 7 | 57 | 33pr | SalVentAttnA | Internal | -0.000251 | 0.002563 | 0.002814 |
| 8 | 34 | 31pv | DefaultC | Internal | -0.000248 | 0.002388 | 0.002637 |
| 9 | 176 | TE1m | TempPar | Internal | -0.000248 | 0.002349 | 0.002597 |
| 10 | 23 | A1 | SomMotB | External | -0.000232 | 0.002225 | 0.002458 |
| 11 | 151 | V6A | DorsAttnA | External | -0.000228 | 0.001979 | 0.002207 |
| 12 | 132 | TE1p | ContC | External | -0.000226 | 0.002186 | 0.002412 |
| 13 | 105 | PoI2 | SalVentAttnA | Internal | -0.000225 | 0.002632 | 0.002857 |
| 14 | 128 | STSdp | DefaultA | Internal | -0.000217 | 0.002773 | 0.002990 |
| 15 | 103 | RI | SomMotB | External | -0.000215 | 0.001977 | 0.002192 |
| 16 | 102 | 52 | SomMotB | External | -0.000206 | 0.002187 | 0.002393 |
| 17 | 129 | STSvp | TempPar | Internal | -0.000206 | 0.002364 | 0.002570 |
| 18 | 116 | AIP | ContB | External | -0.000204 | 0.001745 | 0.001949 |
| 19 | 160 | 31pd | DefaultC | Internal | -0.000202 | 0.002291 | 0.002494 |
| 20 | 17 | FFC | DorsAttnA | External | -0.000192 | 0.002067 | 0.002259 |

## 7. ROI-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency shows the direction of each ROI's relationship with ADHD. **Positive** = increasing this ROI's signal pushes prediction toward ADHD+. **Negative** = pushes toward ADHD-.

### Top 10 ROIs with Strongest Positive (+ADHD) Relationship

| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|
| 1 | 32 | v23ab | DefaultC | Internal | +0.003375 | +0.003112 | +0.003574 |
| 2 | 154 | PHA2 | DefaultB | Internal | +0.003299 | +0.003039 | +0.003496 |
| 3 | 20 | LO2 | VisCent | External | +0.003156 | +0.002941 | +0.003318 |
| 4 | 1 | MST | DorsAttnA | External | +0.002825 | +0.002628 | +0.002974 |
| 5 | 57 | 33pr | SalVentAttnA | Internal | +0.002397 | +0.002215 | +0.002535 |
| 6 | 105 | PoI2 | SalVentAttnA | Internal | +0.002327 | +0.002135 | +0.002471 |
| 7 | 23 | A1 | SomMotB | External | +0.002221 | +0.002072 | +0.002334 |
| 8 | 34 | 31pv | DefaultC | Internal | +0.002196 | +0.002031 | +0.002320 |
| 9 | 102 | 52 | SomMotB | External | +0.002193 | +0.002056 | +0.002296 |
| 10 | 176 | TE1m | TempPar | Internal | +0.002168 | +0.002002 | +0.002293 |

### Top 10 ROIs with Strongest Negative (-ADHD) Relationship

| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|
| 1 | 92 | OFC | Limbic_OFC | Internal | -0.003795 | -0.003479 | -0.004033 |
| 2 | 91 | 13l | Limbic_OFC | Internal | -0.002765 | -0.002584 | -0.002902 |
| 3 | 151 | V6A | DorsAttnA | External | -0.001948 | -0.001796 | -0.002063 |
| 4 | 177 | PI | SalVentAttnA | Internal | -0.001913 | -0.001797 | -0.002001 |
| 5 | 116 | AIP | ContB | External | -0.001712 | -0.001580 | -0.001812 |
| 6 | 117 | EC | Limbic_TempPole | Internal | -0.001689 | -0.001550 | -0.001794 |
| 7 | 121 | PeEc | Limbic_TempPole | Internal | -0.001551 | -0.001451 | -0.001627 |
| 8 | 171 | TGv | Limbic_TempPole | Internal | -0.001504 | -0.001398 | -0.001585 |
| 9 | 109 | Pir | Limbic_TempPole | Internal | -0.001502 | -0.001403 | -0.001578 |
| 10 | 55 | 6v | DorsAttnB | External | -0.001490 | -0.001367 | -0.001583 |
