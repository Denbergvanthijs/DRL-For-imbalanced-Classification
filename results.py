import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 5 runs on 11-09-2020 {12:38, 13:43}, 500k steps, WITHOUT dropout layer, imb 0.001730:
# EPS_MIN = 0.1 ; EPS_STEPS = 200_000; GAMMA = 0.5; LR = 0.00025; WARMUP_STEPS = 60_000; TARGET_MODEL_UPDATE = 0.0005

tp = [78, 81, 75, 78, 78]
tn = [56834, 56825, 56821, 56806, 56842]
fp = [29, 38, 42, 57, 21]  # 37; 12.2
fn = [21, 18, 24, 21, 21]  # 21; 1.9

# 5 runs on 14-09-2020 {10:56, 12:21}
# Difference: WITH dropout layer;

tp = [74, 71, 70, 82, 75]
tn = [56826, 56817, 56844, 56839, 56840]
fp = [37, 46, 19, 24, 23]  # 29; 10.1
fn = [25, 28, 29, 17, 24]  # 24; 4.2

# 5 runs on 14-09-2020 {12:40, 13:52}
# Difference: GAMMA = 0.95;

tp = [71, 81, 67, 77, 78]
tn = [56824, 56822, 56840, 56830, 56839]
fp = [39, 41, 23, 33, 24]  # 32; 7.4
fn = [28, 18, 32, 22, 21]  # 24; 5.1

# 5 runs on 14-09-2020 {14:48, 16:07}
# Difference: GAMMA = 0.9;

tp = [80, 62, 75, 78, 77]
tn = [56839, 56844, 56833, 56843, 56841]
fp = [24, 19, 30, 20, 22]  # 23; 3.9
fn = [19, 37, 24, 21, 22]  # 24; 6.4

# 5 runs on 14-09-2020 {16:29, 10:57}
# Difference: GAMMA = 0.9; terminal=True if current action is for minority and correct;

tp = [79, 76, 81, 76, 75]
tn = [56835, 56848, 56844, 56837, 56828]
fp = [28, 15, 19, 26, 35]  # 24; 7.0
fn = [20, 23, 18, 23, 24]  # 21; 2.2

# 5 runs on 15-09-2020 {13:57, 15:17}
# Difference: GAMMA = 0.95; terminal=True if current action is for minority and correct;

tp = [69, 62, 80, 64, 75]
tn = [56841, 56848, 56795, 56836, 56835]
fp = [22, 15, 68, 27, 28]  # 32; 18.6
fn = [30, 37, 19, 35, 24]  # 29; 6.7

# 5 runs on 11-09-2020 {09:39, 10:39}
# Difference: EPS_MIN = 0.05;

tp = [78, 77, 72, 81, 70]
tn = [56844, 56809, 56835, 56844, 56843]
fp = [19, 54, 28, 19, 20]  # 28; 13.4
fn = [21, 22, 27, 18, 29]  # 23; 4.0

# 5 runs on 11-09-2020 {10:56, 12:13}
# Difference: EPS_MIN = 0.05; LR = 0.001;

tp = [63, 69, 77, 67, 70]
tn = [56836, 56844, 56851, 56860, 56848]
fp = [27, 19, 12, 3, 15]  # 15; 7.9
fn = [36, 30, 22, 32, 29]  # 29; 4.6

avg = [np.mean(x, dtype=int) for x in [tp, tn, fp, fn]]
std = [np.std(x) for x in [tp, tn, fp, fn]]
print([f"{a}, {s:.4f}" for a, s in zip(avg, std)])

ticklabels = ("Minority", "Majority")
sns.heatmap(((avg[0], avg[3]), (avg[2], avg[1])), annot=True, fmt="_d", cmap="viridis", xticklabels=ticklabels, yticklabels=ticklabels)

plt.title("Confusion matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()
