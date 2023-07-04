import threading
import signal
import time
import os

COMMAND_JUPYTER = "jupyter lab -y --collaborative --YDocExtension.document_save_delay=0.5 --port=8889 . > /dev/null 2>&1"
COMMAND_TUNNEL = "ngrok http 8889"

def run_command(command):   
    os.system(command)

def main():
    # suppress output of jupyter lab
    jupyter_lab = threading.Thread(target=run_command, args=(COMMAND_JUPYTER,))
    jupyter_lab.start()

    # wait for jupyter lab to start
    time.sleep(5)

    ngrok_tunnel = threading.Thread(target=run_command, args=(COMMAND_TUNNEL,))
    ngrok_tunnel.start()