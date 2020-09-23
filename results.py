import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 5 runs on 11-09-2020 {12:38, 13:43}, 500k steps, WITHOUT dropout layer, imb 0.001730:
# EPS_MIN = 0.1 ; EPS_STEPS = 200_000; GAMMA = 0.5; LR = 0.00025; WARMUP_STEPS = 60_000; TARGET_MODEL_UPDATE = 0.0005

TP = [78, 81, 75, 78, 78]
TN = [56834, 56825, 56821, 56806, 56842]
FP = [29, 38, 42, 57, 21]  # 37; 12.2
FN = [21, 18, 24, 21, 21]  # 21; 1.9

# 5 runs on 14-09-2020 {10:56, 12:21}
# Difference: WITH dropout layer;

TP = [74, 71, 70, 82, 75]
TN = [56826, 56817, 56844, 56839, 56840]
FP = [37, 46, 19, 24, 23]  # 29; 10.1
FN = [25, 28, 29, 17, 24]  # 24; 4.2

# 5 runs on 17-09-2020 {14:02, 15:22}
# Difference: DOUBLE_DQN=True

TP = [80, 74, 76, 80, 78]
TN = [56817, 56837, 56825, 56842, 56830]
FP = [46, 26, 38, 21, 33]  # 32; 8.8
FN = [19, 25, 23, 19, 21]  # 21; 2.3

# 6 runs on 16-09-2020 {10:48, 12:56}
# Difference: Normalization by Z-score

TP = [83, 83, 83, 84, 86, 78]
TN = [56842, 56782, 56766, 56756, 56699, 56810]
FP = [21, 81, 97, 107, 164, 53]  # 87; 44.7
FN = [16, 16, 16, 15, 13, 21]  # 16; 2.4

# 5 runs on 17-09-2020 {12:17, 13:34}
# Difference: TARGET_MODEL_UPDATE=5_000

TP = [77, 74, 78, 78, 77]
TN = [56815, 56830, 56842, 56812, 56827]
FP = [48, 33, 21, 51, 36]  # 37; 10.8
FN = [22, 25, 21, 21, 22]  # 22; 1.5

# 5 runs on 16-09-2020 {15:58, 17:18}
# Difference: TARGET_MODEL_UPDATE=10_000

TP = [82, 76, 72, 82, 79]
TN = [56752, 56762, 56842, 56821, 56826]
FP = [111, 101, 21, 42, 37]  # 62; 36.4
FN = [17, 23, 27, 17, 20]  # 20; 3.8

# 5 runs on 17-09-2020 {10:36, 11:53}
# Difference: TARGET_MODEL_UPDATE=20_000

TP = [77, 75, 68, 83, 76]
TN = [56824, 56850, 56784, 56792, 56821]
FP = [39, 13, 79, 71, 42]  # 48; 23.8
FN = [22, 24, 31, 16, 23]  # 23; 4.8

# 5 runs on 14-09-2020 {14:48, 16:07}
# Difference: GAMMA = 0.9;

TP = [80, 62, 75, 78, 77]
TN = [56839, 56844, 56833, 56843, 56841]
FP = [24, 19, 30, 20, 22]  # 23; 3.9
FN = [19, 37, 24, 21, 22]  # 24; 6.4

# 5 runs on 14-09-2020 {12:40, 13:52}
# Difference: GAMMA = 0.95;

TP = [71, 81, 67, 77, 78]
TN = [56824, 56822, 56840, 56830, 56839]
FP = [39, 41, 23, 33, 24]  # 32; 7.4
FN = [28, 18, 32, 22, 21]  # 24; 5.1

# 5 runs on 14-09-2020 {16:29, 10:57}
# Difference: GAMMA = 0.9; terminal=True if current action is for minority and correct;

TP = [79, 76, 81, 76, 75]
TN = [56835, 56848, 56844, 56837, 56828]
FP = [28, 15, 19, 26, 35]  # 24; 7.0
FN = [20, 23, 18, 23, 24]  # 21; 2.2

# 5 runs on 15-09-2020 {13:57, 15:17}
# Difference: GAMMA = 0.95; terminal=True if current action is for minority and correct;

TP = [69, 62, 80, 64, 75]
TN = [56841, 56848, 56795, 56836, 56835]
FP = [22, 15, 68, 27, 28]  # 32; 18.6
FN = [30, 37, 19, 35, 24]  # 29; 6.7

# 5 runs on 11-09-2020 {09:39, 10:39}
# Difference: EPS_MIN = 0.05;

TP = [78, 77, 72, 81, 70]
TN = [56844, 56809, 56835, 56844, 56843]
FP = [19, 54, 28, 19, 20]  # 28; 13.4
FN = [21, 22, 27, 18, 29]  # 23; 4.0

# 5 runs on 11-09-2020 {10:56, 12:13}
# Difference: EPS_MIN = 0.05; LR = 0.001;

TP = [63, 69, 77, 67, 70]
TN = [56836, 56844, 56851, 56860, 56848]
FP = [27, 19, 12, 3, 15]  # 15; 7.9
FN = [36, 30, 22, 32, 29]  # 29; 4.6

# 5 runs on 18-09-2020 {11:41, 13:08}
# Difference: terminal=True; EPS_MIN = 0.05; LR = 0.001; DOUBLE_DQN = True; NORMALIZATION = True; TARGET_MODEL_UPDATE = 5_000

TP = [72, 81, 74, 73, 81]
TN = [56843, 56841, 56848, 56497, 56841]
FP = [20, 22, 15, 366, 22]  # 89; 138.5
FN = [27, 18, 25, 26, 18]  # 22; 4.0

# 5 runs on 18-09-2020 {15:57, 18:19}
# Difference: terminal=True; EPS_MIN = 0.05; LR = 0.001; DOUBLE_DQN = True; NORMALIZATION = True

TP = [77, 76, 81, 72, 80]
TN = [56848, 56844, 56850, 56851, 56851]
FP = [15, 19, 13, 12, 12]  # 14; 2.6
FN = [22, 23, 18, 27, 19]  # 21; 3.2

# 5 runs on 21-09-2020 {12:18, 15:08}
# Difference: EPS_MIN = 0.05; LR = 0.001; DOUBLE_DQN = True; NORMALIZATION = True; GAMMA = 0.95

TP = [81, 77, 86, 78, 85]
TN = [56854, 56848, 56450, 56852, 56802]
FP = [9, 15, 413, 11, 61]  # 101; 156.8
FN = [18, 22, 13, 21, 14]  # 17; 3.6

# 5 runs on 22-09-2020 {10:51, 12:10}
# Difference: EPS_MIN = 0.05; LR = 0.001; DOUBLE_DQN = True; Fixed test dataset

TP = [70, 71, 77, 76, 77]
TN = [56841, 56843, 56836, 56835, 56827]
FP = [23, 21, 28, 29, 37]  # 27; 5.5
FN = [28, 27, 21, 22, 21]  # 23; 3.1

avg = [np.mean(x, dtype=int) for x in [TP, TN, FP, FN]]
std = [np.std(x) for x in [TP, TN, FP, FN]]
print([f"{mu}, {sigma:.4f}" for mu, sigma in zip(avg, std)])

ticklabels = ("Minority", "Majority")
sns.heatmap(((avg[0], avg[3]), (avg[2], avg[1])), annot=True, fmt="_d", cmap="viridis", xticklabels=ticklabels, yticklabels=ticklabels)

plt.title("Confusion matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()
