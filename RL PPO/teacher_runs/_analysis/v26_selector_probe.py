import numpy as np

from teacher_rl.twin_static_training import _oracle_rows


rows, _ = _oracle_rows(
    "teacher_runs/v26_twin_mpc_ppo/stage1_oracle/oracle_report.json")
train = [row for row in rows if row["split"] == "train"]
validation = [row for row in rows if row["split"] == "validation"]
mean = np.vstack([row["context"] for row in train]).mean(0)
scale = np.maximum(np.vstack([row["context"] for row in train]).std(0), .05)


def report(name, actions):
    selected = np.stack([row["outcomes"][action]
                         for row, action in zip(validation, actions)])
    released = selected[:, 0] > .5
    hit = selected[:, 1] > .5
    delta = selected[:, 2] * 40. - np.asarray(
        [row["baseline_utility"] for row in validation])
    print(name, dict(release=float(released.mean()),
        hit_given_release=float(hit[released].mean()) if released.any() else 0.,
        positive=float(np.mean(hit & (delta > 0.))), delta=float(delta.mean()),
        counts=np.bincount(actions, minlength=9).tolist()))


for action in range(9):
    report(f"constant-{action}", [action] * len(validation))

train_x = (np.vstack([row["context"] for row in train]) - mean) / scale
validation_x = (np.vstack([row["context"] for row in validation]) - mean) / scale
train_outcomes = np.stack([row["outcomes"] for row in train])
train_scores = np.stack([row["scores"] for row in train])
distances = ((validation_x[:, None, :] - train_x[None, :, :]) ** 2).mean(-1)
for k in (1, 3, 5, 9, 15, 25, 50, 100):
    nearest = np.argsort(distances, axis=1)[:, :k]
    actions = []
    for index, neighbours in enumerate(nearest):
        weights = 1. / np.maximum(distances[index, neighbours], 1e-6)
        estimates = np.average(train_scores[neighbours], axis=0, weights=weights)
        actions.append(int(estimates.argmax()))
    report(f"knn-score-{k}", actions)
    actions = []
    for index, neighbours in enumerate(nearest):
        weights = 1. / np.maximum(distances[index, neighbours], 1e-6)
        estimates = np.average(train_outcomes[neighbours], axis=0, weights=weights)
        # Release and hit dominate; error/time/delta break ties smoothly.
        score = 80.*estimates[:, 0] + 40.*estimates[:, 1] + estimates[:, 2]*40. \
            - 80.*estimates[:, 3] - 2.7*estimates[:, 4]
        actions.append(int(score.argmax()))
    report(f"knn-outcome-{k}", actions)

try:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    for cls in (ExtraTreesRegressor, RandomForestRegressor):
        for leaf in (2, 4, 8, 12):
            model = cls(n_estimators=500, min_samples_leaf=leaf,
                max_features=.75, random_state=20261202, n_jobs=-1)
            model.fit(train_x, train_scores)
            actions = model.predict(validation_x).argmax(1)
            report(f"{cls.__name__}-leaf{leaf}", actions)
except ImportError as error:
    print("sklearn unavailable", error)
