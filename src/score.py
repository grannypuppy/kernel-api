import numpy as np

def geometric_mean_speed_ratio_correct_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    prod = np.prod(speed_up)
    n_correct = np.sum(is_correct) # Count number of correct samples

    return prod ** (1 / n_correct) if n_correct > 0 else 0

def geometric_mean_speed_ratio_correct_and_faster_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples that have speedup > 1
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    speed_up = np.array([x for x in speed_up if x > 1])
    prod = np.prod(speed_up)
    n_correct_and_faster = len(speed_up)

    return prod ** (1 / n_correct_and_faster) if n_correct_and_faster > 0 else 0

def fastp(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float) -> float:
    """
    Rate of samples within a threshold p
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0

def geometric_mean_speed_ratio_correct_only_batch(actual_info: np.ndarray, baseline_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples
    """
    speed_up_avg, speed_up_max = [], []
    for entry, base_speed in zip(actual_info, baseline_speed):
        correct_speed, incorrect_speed = [], []
        for x in entry:
            if x["correctness"]:
                correct_speed.append(base_speed/x["runtime"])
            else:
                incorrect_speed.append(0)
        # if len(correct_speed) == 0:
        #     continue
        speedup = np.sum(correct_speed+incorrect_speed)/16
        speed_up_avg.append(speedup)
        if len(correct_speed)>0:
            speed_up_max.append(max(correct_speed))
        else:
            speed_up_max.append(1)

    return np.mean(speed_up_max), np.mean(speed_up_avg)

def fastp_batch(actual_info: np.ndarray, baseline_speed: np.ndarray, n: int, p: float) -> float:
    """
    Rate of samples within a threshold p
    """
    fast_p_best, fast_p_avg = [], []
    for entry, base_speed in zip(actual_info, baseline_speed):
        correct_speed, incorrect_speed = [], []
        for x in entry:
            if x["correctness"]:
                correct_speed.append(base_speed/x["runtime"])
            else:
                incorrect_speed.append(0)
        # if len(correct_speed) == 0:
        #     continue
        speed_up_avg = correct_speed + incorrect_speed
        if len(correct_speed):
            speed_up_max = max(correct_speed)
        else:
            speed_up_max = 0
        fast_p_best.append(int(speed_up_max) > p)
        fast_p_avg.append(np.sum([_ > p for _ in speed_up_avg])/16)
    return np.sum(fast_p_best)/n*100, np.sum(fast_p_avg)/n*100