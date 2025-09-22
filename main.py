from flask_server import create_flask_app
from dqn_model import DQNAgent
import threading
import subprocess
import time
from results import makeResults

BASE_PORT = 3000
jar_path = "DDQN_ACT.jar"
MODEL_PATH = "C:/Users/ss4587s/Desktop/Models/"
num_epochs = 100

EXPERIMENT_CONDITIONS = {
    # [model path, port number, batchsize, load balancer type, epochs, simualtion name]
    # "RL-1" : [MODEL_PATH+"50-100/checkpoint_step_100.pth", BASE_PORT+1, 50, 4, num_epochs, "RL-M50-CK100-B50"],
    # "RL-2": [MODEL_PATH+"50-100/checkpoint_step_5000.pth", BASE_PORT+2, 50, 4, num_epochs, "RL-M50-CK5000-B50"],
    # "RL-3": [MODEL_PATH+"50-100/checkpoint_step_3100.pth", BASE_PORT+3, 50, 4, num_epochs, "RL-M50-CK3100-B50"],
    # "RL-4": [MODEL_PATH+"50-100/checkpoint_step_1000.pth", BASE_PORT+4, 50, 4, num_epochs, "RL-M50-CK1000-B50"],
    # "RL-8": [MODEL_PATH+"150-100/checkpoint_step_2500.pth", BASE_PORT+8, 150, 4, num_epochs, "RL-M150-CK2500-B150"],
    "RL-9": [MODEL_PATH+"150-100/checkpoint_step_3200.pth", BASE_PORT+9, 150, 4, num_epochs, "RL-M150-CK3200-B150"],
    "RL-10": [MODEL_PATH+"150-100/checkpoint_step_3900.pth", BASE_PORT+10, 150, 4, num_epochs, "RL-M150-CK3900-B150"],
    "RL-11": [MODEL_PATH+"150-100/checkpoint_step_5000.pth", BASE_PORT+11, 150, 4, num_epochs, "RL-M150-CK5000-B150"],
}

# Track running Flask threads to shut down later if needed
flask_threads = []

for name, (model_path, port, batch_size, lb_type, epochs, simName) in EXPERIMENT_CONDITIONS.items():
    if name.startswith("RL"):
        agent = DQNAgent(model_path)
        app = create_flask_app(agent, port)

        thread = threading.Thread(target=app.run, kwargs={"port": port})
        thread.daemon = True
        thread.start()
        flask_threads.append(thread)

        print(f"🧠 Flask server for {name} running on port {port}")
        time.sleep(2)  # Let Flask server start

    # Call your Java process with the correct args
    print(f"🚀 Running experiment {name}...")
    proc = subprocess.Popen([
        "java", "-jar", jar_path,
        str(port), str(batch_size), str(lb_type), str(epochs), simName
    ])
    proc.wait()
    time.sleep(2)

print("✅ All experiments complete.")
# makeResults()
