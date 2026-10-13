# Results

360 scored transcripts: 24 calls x 5 noise levels x 3 seeds.

## Excess verdict flips per criterion (noisy vs clean, minus the judge's clean-vs-clean floor)
Percentage points, 95% bootstrap interval over calls. Sorted by the 30% column.

| criterion | clean accuracy | self-flip (floor) | excess flips @ 5% WER | excess flips @ 10% WER | excess flips @ 20% WER | excess flips @ 30% WER |
|---|---|---|---|---|---|---|
| resolution_accuracy | 94% | 8% | +21 pts (+9 to +33) | +40 pts (+24 to +56) | +45 pts (+29 to +62) | +51 pts (+33 to +69) |
| closing | 86% | 8% | +7 pts (-3 to +18) | +6 pts (-3 to +19) | +6 pts (-2 to +16) | +14 pts (+4 to +25) |
| opening | 100% | 0% | +8 pts (+3 to +14) | +11 pts (+6 to +18) | +8 pts (+3 to +15) | +14 pts (+6 to +24) |
| verification | 96% | 0% | +8 pts (+1 to +17) | +22 pts (+11 to +33) | +17 pts (+6 to +29) | +12 pts (+4 to +24) |
| understanding | 97% | 6% | +1 pts (-4 to +6) | +14 pts (+4 to +24) | +9 pts (+2 to +15) | +12 pts (+4 to +19) |
| process_compliance | 82% | 14% | -3 pts (-9 to +3) | +2 pts (-6 to +13) | +0 pts (-6 to +8) | +5 pts (-4 to +13) |
| communication | 94% | 11% | +1 pts (-6 to +10) | +3 pts (-3 to +11) | -1 pts (-7 to +6) | +4 pts (-4 to +13) |

## Accuracy against the gold labels

| criterion | 0% WER | 5% WER | 10% WER | 20% WER | 30% WER |
|---|---|---|---|---|---|
| opening | 100% | 92% | 89% | 92% | 86% |
| verification | 96% | 88% | 74% | 79% | 83% |
| understanding | 97% | 94% | 81% | 85% | 82% |
| resolution_accuracy | 94% | 68% | 50% | 42% | 35% |
| process_compliance | 82% | 76% | 75% | 83% | 74% |
| communication | 94% | 92% | 88% | 94% | 88% |
| closing | 86% | 85% | 78% | 83% | 74% |

Cohen's kappa vs gold: 0% WER 0.83, 5% WER 0.67, 10% WER 0.52, 20% WER 0.57, 30% WER 0.49

## Which way the errors go (all criteria pooled)
False fail = the agent did it right and the judge marked it wrong. False pass = the reverse.

| WER | false fail | false pass | correct resolutions marked wrong |
|---|---|---|---|
| 0% | 10% | 0% | 7% |
| 5% | 20% | 0% | 43% |
| 10% | 31% | 1% | 67% |
| 20% | 26% | 2% | 78% |
| 30% | 33% | 2% | 87% |

Quotes found verbatim in the transcript the judge saw: 0% WER 99%, 5% WER 100%, 10% WER 99%, 20% WER 99%, 30% WER 100%

Measured WER of the noised transcripts: target 0% -> 0.0%, target 5% -> 4.9%, target 10% -> 10.0%, target 20% -> 20.0%, target 30% -> 30.1%

## Where the flips come from
When noise touched the turn that decides a criterion, the verdict left the clean majority 20% of the time (n=1563); when it did not, 5% (n=261).
