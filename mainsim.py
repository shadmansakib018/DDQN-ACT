from flask_server import create_flask_app
from dqn_model import DQNAgent
import threading
import subprocess
import time
from results import makeResults
from ppo_model import PPOAgent

BASE_PORT = 3000
jar_path = "PPO_ACT_10VM.jar"
MODEL_PATH = "C:/Users/ss4587s/Desktop/DDQN-150/checkpoints/checkpoint_step_3200.pth"
num_epochs = 100
MAX_CONCURRENT = 2  # Limit to 2 concurrent Java processes

EXPERIMENT_CONDITIONS = {
    # "RL-1": [MODEL_PATH+"/checkpoint_step_2200.pth", BASE_PORT+1, 50, 4, num_epochs, "RL-M150-CK2200-B50"],
    # "RL-2": [MODEL_PATH+"/checkpoint_step_2200.pth", BASE_PORT+2, 100, 4, num_epochs, "RL-M150-CK2200-B100"],
    # "RL-3": [MODEL_PATH+"/checkpoint_step_2200.pth", BASE_PORT+3, 150, 4, num_epochs, "RL-M150-CK2200-B150"],
    # "RL-4": [MODEL_PATH+"/checkpoint_step_2200.pth", BASE_PORT+4, 200, 4, num_epochs, "RL-M150-CK2200-B200"],
    # "RL-5": [MODEL_PATH+"/checkpoint_step_2200.pth", BASE_PORT+5, 250, 4, num_epochs, "RL-M150-CK2200-B250"],
    # "RL-6": ["--", BASE_PORT+6, 300, 5, num_epochs, "DY-V2-300"],
    # "RL-7": ["--", BASE_PORT+6, 350, 5, num_epochs, "DY-V2-350"],
    # "RL-8": ["--", BASE_PORT+6, 400, 5, num_epochs, "DY-V2-400"],
    # "RL-9": ["--", BASE_PORT+6, 450, 5, num_epochs, "DY-V2-450"],
    # "RL-19": ["--", BASE_PORT+6, 500, 5, num_epochs, "DY-V2-500"],
    # "RL-6": ["--", BASE_PORT+6, 100, 3 ]
    "RL-1": [MODEL_PATH, BASE_PORT+1, 50, 4, num_epochs, "RL-PPO-M50-CK3200-B50"],
    "RL-2": [MODEL_PATH, BASE_PORT+2, 100, 4, num_epochs, "RL-PPO-M50-CK3200-B100"],
    "RL-3": [MODEL_PATH, BASE_PORT+3, 150, 4, num_epochs, "RL-PPO-M50-CK3200-B150"],
    "RL-4": [MODEL_PATH, BASE_PORT+4, 200, 4, num_epochs, "RL-PPO-M50-CK3200-B200"],
    "RL-5": [MODEL_PATH, BASE_PORT+5, 250, 4, num_epochs, "RL-PPO-M50-CK3200-B250"],
    "RL-6": [MODEL_PATH, BASE_PORT+6, 300, 4, num_epochs, "RL-PPO-M50-CK3200-B300"],
    "RL-7": [MODEL_PATH, BASE_PORT+7, 350, 4, num_epochs, "RL-PPO-M50-CK3200-B350"],
    "RL-8": [MODEL_PATH, BASE_PORT+8, 400, 4, num_epochs, "RL-PPO-M50-CK3200-B400"],
    "RL-9": [MODEL_PATH, BASE_PORT+9, 450, 4, num_epochs, "RL-PPO-M50-CK3200-B450"],
    "RL-10": [MODEL_PATH, BASE_PORT+10, 500, 4, num_epochs, "RL-PPO-M50-CK3200-B500"],

}

flask_threads = []
experiment_threads = []
semaphore = threading.Semaphore(MAX_CONCURRENT)

# Start Flask servers
for name, (model_path, port, _, _, _, _) in EXPERIMENT_CONDITIONS.items():
# for name, (model_path, port, _, _ ) in EXPERIMENT_CONDITIONS.items(): 
    # agent = DQNAgent(model_path)
    agent = PPOAgent(model_path)
    app = create_flask_app(agent, port)
    thread = threading.Thread(target=app.run, kwargs={"port": port})
    thread.daemon = True
    thread.start()
    flask_threads.append(thread)
    print(f"🧠 Flask server for {name} running on port {port}")
    time.sleep(1.5)  # Allow server to start

# Worker function to run each experiment
def run_experiment(name, port, batch_size, lb_type, epochs, simName):
# def run_experiment(name, port, batch_size, lb_type):
    with semaphore:
        print(f"🚀 Starting experiment {name}...")
        proc = subprocess.Popen([
            "java", "-jar", jar_path,
            str(port), str(batch_size), str(lb_type), str(epochs), simName
            # str(port), str(batch_size), str(lb_type)
        ])
        proc.wait()
        print(f"✅ Finished experiment {name}")

# Create and start threads (up to MAX_CONCURRENT will run at a time)
for name, (model_path, port, batch_size, lb_type, epochs, simName) in EXPERIMENT_CONDITIONS.items():
# for name, (model_path, port, batch_size, lb_type) in EXPERIMENT_CONDITIONS.items():
    thread = threading.Thread(target=run_experiment, args=(name, port, batch_size, lb_type, epochs, simName))
    # thread = threading.Thread(target=run_experiment, args=(name, port, batch_size, lb_type))
    thread.start()
    experiment_threads.append(thread)

# Wait for all experiment threads to complete
for t in experiment_threads:
    t.join()

print("🎉 All experiments complete.")
# makeResults()
