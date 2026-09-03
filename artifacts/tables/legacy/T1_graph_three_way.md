| question set | system | AllGoldFound | gpt4o-mini | gemini-flash-lite | llama4-scout | llama4-maverick |
|---|---|---|---|---|---|---|
| PubLayNet multi-hop (within-page) | baseline | 0.9 | 0.433 (0.500) | 0.600 (0.467) | 0.500 (0.467) | 0.500 (0.467) |
| PubLayNet multi-hop (within-page) | +KG | 0.9 | 0.367 (0.467) | 0.533 (0.433) | 0.533 (0.467) | 0.533 (0.400) |
| SPIQA multi-hop (within-paper) | baseline | 1.0 | 0.767 (0.833) | 0.767 (0.833) | 0.667 (0.767) | 0.767 (0.767) |
| SPIQA multi-hop (within-paper) | +KG | 1.0 | 0.700 (0.767) | 0.767 (0.833) | 0.800 (0.767) | 0.733 (0.800) |
| SPIQA multi-hop (cross-paper) | baseline | 0.5 | 0.620 (0.660) | 0.820 (0.800) | 0.780 (0.720) | 0.700 (0.580) |
| SPIQA multi-hop (cross-paper) | +KG | 0.5 | 0.700 (0.640) | 0.780 (0.780) | 0.820 (0.800) | 0.760 (0.700) |
| SPIQA multi-hop (cross-paper) | +KGret | 0.72 | 0.900 (0.860) | 0.880 (0.900) | 0.820 (0.760) | 0.840 (0.720) |
| HotpotQA bridge | baseline | 0.22 | 0.520 (0.480) | 0.340 (0.340) | 0.520 (0.380) | 0.560 (0.540) |
| HotpotQA bridge | +KG | 0.22 | 0.480 (0.400) | 0.400 (0.360) | 0.540 (0.340) | 0.600 (0.400) |
| HotpotQA bridge | +KGret | 0.46 | 0.720 (0.640) | 0.560 (0.580) | 0.660 (0.400) | 0.680 (0.680) |
| HotpotQA comparison (control) | baseline | 0.86 | 0.880 (0.760) | 0.920 (0.840) | 0.960 (0.900) | 0.920 (0.840) |
| HotpotQA comparison (control) | +KG | 0.86 | 0.900 (0.700) | 0.900 (0.760) | 0.920 (0.720) | 0.920 (0.700) |
| HotpotQA comparison (control) | +KGret | 0.98 | 0.880 (0.840) | 0.940 (0.820) | 0.980 (0.880) | 0.920 (0.900) |

_Cells: accuracy (faithfulness). AllGoldFound is retrieval completeness over ALL gold sources; identical across models within a system because the retrieval stack is fixed. +KGret expands the candidate set, so its completeness differs by design._
