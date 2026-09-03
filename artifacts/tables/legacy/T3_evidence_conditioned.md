| question set | system | acc | complete | acc | incomplete | n complete | n incomplete |
|---|---|---|---|---|---|
| PubLayNet multi-hop (within-page) | baseline | 0.565 | 0.000 | 108 | 12 |
| PubLayNet multi-hop (within-page) | +KG | 0.546 | 0.000 | 108 | 12 |
| SPIQA multi-hop (within-paper) | baseline | 0.742 | — | 120 | 0 |
| SPIQA multi-hop (within-paper) | +KG | 0.750 | — | 120 | 0 |
| SPIQA multi-hop (cross-paper) | baseline | 0.847 | 0.584 | 111 | 89 |
| SPIQA multi-hop (cross-paper) | +KG | 0.865 | 0.640 | 111 | 89 |
| SPIQA multi-hop (cross-paper) | +KGret | 0.887 | 0.776 | 151 | 49 |
| HotpotQA bridge | baseline | 0.977 | 0.346 | 44 | 156 |
| HotpotQA bridge | +KG | 0.977 | 0.372 | 44 | 156 |
| HotpotQA bridge | +KGret | 0.913 | 0.435 | 92 | 108 |
| HotpotQA comparison (control) | baseline | 0.953 | 0.714 | 172 | 28 |
| HotpotQA comparison (control) | +KG | 0.948 | 0.679 | 172 | 28 |
| HotpotQA comparison (control) | +KGret | 0.939 | 0.500 | 196 | 4 |

_Pooled over generators. 'acc | incomplete' for baseline is the parametric-leakage floor: correct answers produced without complete gold evidence in context._
