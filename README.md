# JourneyEscape ML Tech II

# **DQN Atari Agent — JourneyEscape Project**

# 🎮 **Gameplay Demo**

[![Watch the demo](https://img.youtube.com/vi/6q1p2EbelBE/0.jpg)](https://youtu.be/6q1p2EbelBE)

[https://youtu.be/6q1p2EbelBE](https://youtu.be/6q1p2EbelBE)

---

### Group Members:

- Patrick 
- Alice
- Kellia
- Phinnah

## **1. Project Overview**

This project implements a **Deep Q-Network (DQN)** agent trained to play an Atari game using **Stable-Baselines3** and **Gymnasium**.

Environment used:

### **`ALE/JourneyEscape-v5`**

The project includes:

- DQN agent training
- Hyperparameter tuning (10 experiments per member → 40 total)
- CNN vs MLP policy comparison
- Final gameplay demo
- Agent evaluation using Greedy Q-policy

This README acts as both the project report and setup guide.

---

# **2. Repository Setup**

Clone the project:

```bash
git https://github.com/thepatrickniyo/journey-escape.git
cd journey-escape.git
```

---

## **3. Create Virtual Environment**

```bash
python3 -m vvenv .venv
source .venv/bin/activate      # Mac/Linux
.\.venv\Scripts\activate       # Windows
```

---

## **4. Install Dependencies**

```bash
pip install -r requirements.txt
```

Required packages:

```
stable-baselines3
gymnasium[atari]
gymnasium[accept-rom-license]
opencv-python
numpy
```

---

# **5. Training the Agent**

```bash
python train.py
```

### What `train.py` does

- Loads **ALE/JourneyEscape-v5**
- Builds a DQN agent (MLP + CNN options)
- Trains for a specified number of timesteps
- Logs training metrics
- Saves the trained model at:
  `models/dqn_model.zip`

---

# **6. Playing the Trained Agent**

```bash
python play.py
```
### What `play.py` does

- Loads the trained model
- Uses **Greedy Q-policy**
- Renders the Atari game in real time

---

# 🔍 **7. Hyperparameter Tuning Report**

Each member ran 10 experiments varying key DQN hyperparameters. Results are drawn from `hyperparameter_results.csv`.

---

## Patrick — Hyperparameter Experiments

| Exp | lr | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior |
|-----|----|-------|-------|-----------|---------|-----------|-------------------|
| 1 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | Low lr and small batch with reduced gamma discounting. Slow learning, high variance — agent struggled to converge, yielding one of the worst rewards (-22780). |
| 2 | 5e-4 | 0.97 | 64 | 0.9 | 0.10 | 0.20 | Moderate lr and higher gamma with medium batch. Faster exploration decay helped but high epsilon end limited exploitation (-10470). |
| 3 | 1e-3 | 0.99 | 128 | 1.0 | 0.00 | 0.05 | High lr, strong discounting, large batch. Full exploitation by end; large batch stabilized updates. Competitive reward (-9670). |
| 4 | 5e-4 | 0.90 | 32 | 0.8 | 0.05 | 0.10 | Reduced gamma weakened long-term credit assignment. Small batch with partial initial exploration caused instability and poor rewards (-22070). |
| 5 | 1e-4 | 0.99 | 64 | 1.0 | 0.20 | 0.20 | Very slow lr with high epsilon floor; agent remained too exploratory throughout, limiting exploitation of the learned policy (-13750). |
| 6 | 1e-3 | 0.92 | 128 | 0.9 | 0.05 | 0.05 | High lr with large batch compensated for lower gamma. Fast convergence to greedy policy. Best single-run reward (-9250). |
| 7 | 7.5e-4 | 0.95 | 32 | 1.0 | 0.00 | 0.10 | Mid lr, moderate gamma, small batch. Zero epsilon end forced full greedy policy; unstable due to small batch despite decent lr (-10520). |
| 8 | 5e-4 | 0.97 | 64 | 0.8 | 0.05 | 0.20 | Best overall. Balanced lr, strong gamma, medium batch, rapid exploration decay. Good stability and exploitation (-7850). |
| 9 | 1e-3 | 0.99 | 128 | 1.0 | 0.10 | 0.05 | High lr and large batch but slow exploration decay kept the agent exploratory too long, hurting final policy quality (-14760). |
| 10 | 2.5e-4 | 0.93 | 64 | 0.9 | 0.20 | 0.10 | Low lr, weak gamma, high epsilon floor. Agent over-explored without fully exploiting learned policy; poor convergence (-18070). |

---

## Alice — Hyperparameter Experiments

| Exp | lr | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior |
|-----|----|-------|-------|-----------|---------|-----------|-------------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | Baseline config with small batch. Steady but slow learning; noisy gradients from small batch limited performance (-12970). |
| 2 | 5e-4 | 0.97 | 64 | 0.9 | 0.10 | 0.20 | Moderate lr, high gamma, medium batch with faster epsilon decay. Well-balanced setup; agent converged efficiently (-6.9). |
| 3 | 1e-3 | 0.99 | 128 | 1.0 | 0.00 | 0.05 | Best reward in Alice's set. High lr, max gamma, large batch, full greedy end. Very stable learning with strong long-term credit assignment (-3.5). |
| 4 | 1e-4 | 0.92 | 64 | 0.8 | 0.05 | 0.10 | Low lr weakened the learning signal; reduced gamma hurt long-term planning. Moderate but inconsistent results (-10.3). |
| 5 | 5e-4 | 0.95 | 128 | 1.0 | 0.20 | 0.20 | Good balance of lr and large batch; high epsilon end maintained some exploration throughout. Stable, consistent improvement (-4.4). |
| 6 | 1e-3 | 0.97 | 32 | 0.9 | 0.10 | 0.05 | High lr with small batch caused unstable gradient updates; slow exploration decay compounded instability (-14.8). |
| 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.00 | 0.20 | Very low lr with fast epsilon decay; agent transitioned to greedy policy before learning sufficiently. Worst result  (-18.4). |
| 8 | 5e-4 | 0.92 | 32 | 0.8 | 0.05 | 0.10 | Reduced gamma and small batch with partial initial exploration led to unstable updates and poor reward (-15.2). |
| 9 | 1e-3 | 0.95 | 64 | 1.0 | 0.20 | 0.05 | High lr with moderate gamma and slow epsilon decay; kept reasonable exploration while allowing exploitation (-6.3). |
| 10 | 7.5e-4 | 0.97 | 128 | 0.9 | 0.10 | 0.10 | Second best. Strong combination of mid-high lr, high gamma, and large batch for stable and consistent gradient updates (-2.9). |

---

## Phinnah — Hyperparameter Experiments

| Exp | lr | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior |
|-----|----|-------|-------|-----------|---------|-----------|-------------------|
| 1 | 5e-4 | 0.80 | 32 | 1.0 | 0.05 | 0.10 | Very low gamma severely limited long-term value estimation. Short-sighted policy — worst result (-27250). |
| 2 | 5e-4 | 0.85 | 64 | 0.9 | 0.10 | 0.20 | Increased gamma and batch over exp01; noticeable improvement. Moderate exploration decay aided exploitation (-9010). |
| 3 | 5e-4 | 0.90 | 128 | 1.0 | 0.00 | 0.05 | Larger batch stabilized training; full greedy end with slow decay. Reasonable performance as gamma grows (-6440). |
| 4 | 5e-4 | 0.92 | 32 | 0.8 | 0.05 | 0.10 | Small batch and partial initial exploration combined with lower gamma caused noisy updates and poor convergence (-27040). |
| 5 | 5e-4 | 0.95 | 64 | 1.0 | 0.20 | 0.20 | Moderate gamma and batch; fast decay but high epsilon floor maintained too much exploration, limiting exploitation (-10080). |
| 6 | 5e-4 | 0.97 | 128 | 0.9 | 0.10 | 0.05 | High gamma with large batch; slow epsilon decay kept exploration long. Stable training but not fully exploiting by end (-11740). |
| 7 | 5e-4 | 0.99 | 32 | 1.0 | 0.00 | 0.20 | Max gamma with small batch; fast epsilon transition to fully greedy destabilized learning despite strong discounting (-14820). |
| 8 | 5e-4 | 0.93 | 64 | 0.8 | 0.05 | 0.10 | Best result. Good gamma, medium batch, low epsilon end. Well-balanced exploration-exploitation tradeoff (-5120). |
| 9 | 5e-4 | 0.96 | 128 | 1.0 | 0.10 | 0.05 | Strong gamma and large batch; slow epsilon decay maintained healthy exploration. Second best (-7510). |
| 10 | 5e-4 | 0.94 | 64 | 0.9 | 0.20 | 0.10 | High epsilon floor prevented full exploitation; moderate gamma and batch led to inconsistent convergence (-15440). |

---

## Kellia — Hyperparameter Experiments

| Exp | lr | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior |
|-----|----|-------|-------|-----------|---------|-----------|-------------------|
| 1 | 5e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | Moderate lr with small batch and decent gamma. Slow gradient updates due to small batch; moderate but inconsistent performance (-11580). |
| 2 | 5e-4 | 0.97 | 64 | 0.9 | 0.10 | 0.20 | Best result. Higher gamma with medium batch and fast epsilon decay. Good balance of stability and exploitation (-7750). |
| 3 | 5e-4 | 0.99 | 128 | 1.0 | 0.00 | 0.05 | Max gamma and large batch with slow epsilon decay and full greedy end. Stable but slow convergence; large batch helped consistency (-11710). |
| 4 | 5e-4 | 0.92 | 32 | 0.8 | 0.05 | 0.10 | Reduced gamma and small batch with partial initial exploration caused noisy updates and weaker long-term planning (-17820). |
| 5 | 5e-4 | 0.95 | 64 | 1.0 | 0.20 | 0.20 | High epsilon floor maintained too much exploration throughout; fast decay couldn't compensate, limiting policy exploitation (-14440). |
| 6 | 5e-4 | 0.97 | 128 | 0.9 | 0.10 | 0.05 | Second best. Large batch with high gamma and slow decay allowed stable, consistent learning with good exploitation (-6800). |
| 7 | 5e-4 | 0.99 | 32 | 1.0 | 0.00 | 0.20 | Max gamma with small batch and fast epsilon decay; agent went greedy too early before learning adequately — worst result (-20660). |
| 8 | 5e-4 | 0.93 | 64 | 0.8 | 0.05 | 0.10 | Moderate gamma and batch with low epsilon end. Partial initial exploration aided early diversity; reasonable convergence (-9180). |
| 9 | 5e-4 | 0.96 | 128 | 1.0 | 0.10 | 0.05 | Strong gamma and large batch with slow decay; steady improvement but exploration maintained slightly too long (-12160). |
| 10 | 5e-4 | 0.94 | 32 | 0.9 | 0.20 | 0.10 | High epsilon floor and small batch prevented full exploitation; moderate gamma limited long-term value estimation (-16930). |

---

# **8. Policy Architecture Comparison**

## MLPPolicy

- Simple
- Works for 1D inputs
- ❌ Poor for image-based Atari games

## CNNPolicy

- Extracts spatial features
- Stable
- ✔ Best for Atari

### **Final Choice:** CNNPolicy

---

# 📈 **9. Key Insights from Tuning**

**Improvements:**

- γ = 0.99
- lr = 1e-4
- batch = 64–128
- slow epsilon decay

**Hurts performance:**

- High lr
- Small batch
- Fast epsilon decay

---

# **10. Project Structure**

```
.
├── train.py
├── play.py
├── models/
├── experiments/
├── logs/
├── hyperparameter_results.csv
├── requirements.txt
├── README.md
└── .venv/
```
