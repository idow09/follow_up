import numpy as np

from benchmark_framework.sample_data import Prediction

CHANCE_FOR_RANDOM_PREDICTION = 0.03

NUM_PREDS_LIST = [0, 1, 2, 3]

NUM_PREDS_PROBS = [0.05, 0.8, 0.1, 0.05]


def _generate_random_probability(mu=0.8, sigma=0.5):
    sc = np.random.normal(mu, sigma)
    sc = 1.0 if sc > 1 else sc
    sc = 0.0 if sc < 0 else sc
    return sc


def _generate_random_pred():
    x = int(np.random.uniform(1, 1279))
    y = int(np.random.uniform(1, 719))
    r = np.random.normal(15, 8)
    sc = _generate_random_probability(0.1, 0.1)
    return x, y, r, sc


def _generate_fake_pred(label):
    if np.random.uniform(0, 1) < CHANCE_FOR_RANDOM_PREDICTION:
        return _generate_random_pred()
    x, y, r = label.x, label.y, label.r
    dx, dy, dr = np.random.normal(0, r / 2), np.random.normal(0, r / 2), np.random.normal(0, r / 10)
    x, y, r = int(x + dx), int(y + dy), r + dr
    # sc = generate_random_probability()
    sc = np.exp(-(dx ** 2 + dy ** 2 + dr ** 2) / 150)
    return x, y, r, sc


def generate_fake_preds(preds_path, labels, persist=False):
    preds = []
    time = np.random.exponential(0.3)
    if persist:
        open(preds_path, 'a').close()

    for label in labels:
        pred_num_for_label = np.random.choice(NUM_PREDS_LIST, p=NUM_PREDS_PROBS)
        for i in range(pred_num_for_label):
            x, y, r, sc = _generate_fake_pred(label)
            if persist:
                with open(preds_path, 'a') as pred_f:
                    pred_f.write(('%g ' * 5 + '\n') % (x, y, r, sc, time))
            pred = Prediction(x, y, r, sc)
            pred.match_label(labels)
            preds.append(pred)
    return preds, time
