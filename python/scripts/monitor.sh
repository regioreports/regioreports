#!/bin/bash
until python final_with_db.py; do
    echo "'myscript.py' crashed with exit code $?. Restarting..." >&2
    sleep 1
done