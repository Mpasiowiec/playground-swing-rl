import subprocess

# Ścieżki do Pythona oraz do skryptu trenowania agenta
python_exe = r"C:/Users/micha/Desktop/repo/playground-swing-rl/swing-rl/Scripts/python.exe"
script_path = r"c:/Users/micha/Desktop/repo/playground-swing-rl/scripts/train_agent.py"

# Lista używanych algorytmów "A2C",
algorithms = ["A2C", "DDPG", "PPO", "SAC", "TD3"]

# 3-krotne uruchomienie dla każdego algorytmu
for algo in algorithms:
    for iter in range(3):
        print(f"Uruchamiam {algo}, powtórzenie {iter+1}")
        subprocess.run([python_exe, script_path, "--sb3_algo", algo, '--test', '--test_model', f'{iter+1}/best_model'])
