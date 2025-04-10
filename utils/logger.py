import time
from contextlib import contextmanager

@contextmanager
def log_tokens_per_second(task_name="⏱️ Task"):
    start = time.time()
    print(f"{task_name} started...")
    yield
    end = time.time()
    duration = end - start
    print(f"{task_name} completed in {duration:.2f} seconds.")
