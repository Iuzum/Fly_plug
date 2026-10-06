"""
Скачать Hemibrain коннектом через neuprint API (fetch_custom).
Сохраняет в /tmp/hemibrain.pkl для coarse-graining.
"""
from neuprint import Client
import time
import pickle
import os

print("Connecting to neuprint...", flush=True)
c = Client('https://neuprint.janelia.org', dataset='hemibrain:v1.2.1')
print("Connected.", flush=True)

# === 1. Нейроны: bodyId, тип, размер, регион ===
print("Fetching neurons...", flush=True)
t0 = time.time()
q_neurons = """
MATCH (n:Neuron)
RETURN n.bodyId AS bodyId,
       n.type AS type,
       n.instance AS instance,
       n.pre AS pre,
       n.post AS post,
       n.size AS size
"""
neurons = c.fetch_custom(q_neurons)
print(f"Got {len(neurons)} neurons in {time.time()-t0:.1f}s", flush=True)
print(f"Columns: {list(neurons.columns)}", flush=True)

# === 2. Синапсы: pre → post ===
print("Fetching synapses (this will take 10-40 min)...", flush=True)
t0 = time.time()
q_synapses = """
MATCH (a:Neuron)-[s:ConnectsTo]->(b:Neuron)
RETURN a.bodyId AS pre,
       b.bodyId AS post,
       s.weight AS weight,
       s.count AS count
"""
synapses = c.fetch_custom(q_synapses)
print(f"Got {len(synapses)} synapses in {time.time()-t0:.1f}s", flush=True)

# === 3. Сохранить ===
print("Saving to /tmp/hemibrain.pkl ...", flush=True)
with open('/tmp/hemibrain.pkl', 'wb') as f:
    pickle.dump({'neurons': neurons, 'synapses': synapses}, f)
print("DONE. File size:", flush=True)
print(f"{os.path.getsize('/tmp/hemibrain.pkl') / 1e6:.1f} MB", flush=True)
