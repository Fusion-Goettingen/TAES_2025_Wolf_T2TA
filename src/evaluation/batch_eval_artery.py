import sys

sys.path.append(".")
import os
from src.evaluation.run_eval_artery import run_eval
from src.abspath import artery_results_path


first_folder = 1
last_folder = 10
num_sweeps = 50
num_sweeps = 1

start_time = 15.0
end_time = 45.0
end_time = 15.1


for run in range(first_folder, last_folder + 1):
    print ('###### run: ', run)
    for rules in ["full", "etsi"]: # full communication or etsi rules
        print('######## rules: ', rules)

        for penetration_rate in ["25", "50", "100"]: # market penetration rate
            print('########## MPR: ', penetration_rate)

            path = os.path.join(artery_results_path, str(run), rules, penetration_rate)
            if not os.path.exists(path):
                print(path, "does not exist!")
                continue

            run_eval(
                root_path=artery_results_path,
                run=run,
                rules=rules,
                penetration_rate=penetration_rate,
                num_sweeps=num_sweeps,
                start_time=start_time,
                end_time=end_time,
                save_tracks_time=1.0,
                invalidate_time=1.0,
            )
