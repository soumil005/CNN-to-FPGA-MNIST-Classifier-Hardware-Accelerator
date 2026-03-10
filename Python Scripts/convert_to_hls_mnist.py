# convert_to_hls.py
import numpy as np
import tf_keras as keras    # ← tf_keras here too
import hls4ml
import os
import json

os.environ['PATH'] = r'E:\Xilinx\Vitis_HLS\2023.1\bin' + os.pathsep + os.environ['PATH']

BASE_DIR = r"C:\Users\soumi\OneDrive\Desktop\BTech Project\fpga_claude_cnn"
OUTPUT_DIR = os.path.join(BASE_DIR, "hls_project")

# ── Load model and test data ───────────────────────────────────────────────────
model  = keras.models.load_model('small_cnn.h5')
X_test = np.load('x_test_sample.npy')

# ── Verify the model config has batch_input_shape (sanity check) ───────────────
config_json = model.to_json()
config_dict = json.loads(config_json)
first_layer = config_dict['config']['layers'][0]['config']
print("First layer config keys:", list(first_layer.keys()))
# You should see 'batch_input_shape' in the list

# ── Generate hls4ml config ─────────────────────────────────────────────────────
config = hls4ml.utils.config_from_keras_model(
    model,
    granularity='name'
)

# ── Tune precision ─────────────────────────────────────────────────────────────
config['Model']['Precision']   = 'ap_fixed<12,4>'
config['Model']['ReuseFactor'] = 16
config['Model']['Strategy'] = 'Resource'

config['LayerName']['conv1']['Precision']['weight']  = 'ap_fixed<8,3>'
config['LayerName']['conv1']['Precision']['bias']    = 'ap_fixed<8,3>'
config['LayerName']['conv1']['ReuseFactor'] = 9

config['LayerName']['conv2']['Precision']['weight']  = 'ap_fixed<8,3>'
config['LayerName']['conv2']['Precision']['bias']    = 'ap_fixed<8,3>'
config['LayerName']['conv2']['ReuseFactor'] = 9

config['LayerName']['dense1']['Precision']['weight'] = 'ap_fixed<8,3>'
config['LayerName']['dense1']['Precision']['bias']   = 'ap_fixed<8,3>'
config['LayerName']['dense1']['ReuseFactor'] = 32

config['LayerName']['output']['ReuseFactor'] = 16

print("\n── hls4ml config ──────────────────────────────")
print(json.dumps(config, indent=2))

# ── Convert to HLS ─────────────────────────────────────────────────────────────
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=r'D:\fpga_cnn_mnist\hls_project',
    backend='Vitis',
    part='xc7a200tfbg676-2',    # ← change to your FPGA part
    clock_period=10,
    io_type='io_stream'
)

hls_model.write()

print("\n── Conversion complete! ───────────────────────")
print("HLS project written to: hls_project/")

print("\n── Verifying output ───────────────────────────")
if os.path.exists(OUTPUT_DIR):
    print(f"✓ Folder exists at: {OUTPUT_DIR}")
    print("\nContents:")
    for item in os.listdir(OUTPUT_DIR):
        print(f"  {item}")
else:
    print(f"✗ Folder NOT found at: {OUTPUT_DIR}")
    print("  Check for permission errors or OneDrive sync issues.")