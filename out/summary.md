| Model | Params | LOSO acc % (mean ± sd) | LOSO macro-F1 % | Worst subject acc % |
|---|---|---|---|---|
| Random Forest (features) | - | 95.9 ± 5.4 | 95.9 ± 5.1 | 81.8 |
| LSTM only | 21516 | 73.0 ± 12.6 | 69.8 ± 12.8 | 46.1 |
| CNN only | 26444 | 84.6 ± 9.7 | 82.4 ± 10.9 | 65.9 |
| CNN-LSTM | 59724 | 86.8 ± 4.4 | 84.0 ± 5.9 | 79.2 |
| CNN-LSTM + augmentation | 59724 | 90.4 ± 7.4 | 89.4 ± 7.8 | 79.6 |
| CNN-LSTM + aug + magnitude ch. | 61324 | 92.4 ± 5.9 | 91.2 ± 6.9 | 81.8 |

Random window split (leaky):
- cnn_lstm_random: acc 99.78%, F1 99.79%
- cnn_random: acc 99.85%, F1 99.85%
