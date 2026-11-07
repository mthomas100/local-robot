"""procs.py — process hygiene for the cell's parallel jobs (2026-11-02).

bin/memguard kills only its direct child. A multiprocessing pool's spawn workers then live on as orphans (12 of them held
~7 GB after an F3 evaluation was stopped). Every pool here passes `initializer=die_with_parent` (or calls it first in
its own initializer): each worker exits within a second of its parent dying.
"""
import os, threading, time


def die_with_parent():
    ppid = os.getppid()

    def watch():
        while True:
            if os.getppid() != ppid: os._exit(1)
            time.sleep(1.0)
    threading.Thread(target=watch, daemon=True).start()
