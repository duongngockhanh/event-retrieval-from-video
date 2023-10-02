import os

submission_folder = "submission"
os.makedirs(submission_folder)

n = 30

for i in range(1, n+1):
    os.system(f"touch {submission_folder}/query-p3-{i}.csv")