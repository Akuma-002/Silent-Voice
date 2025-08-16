from machine import ADC, Pin
import network
import socket
import time

# -------------------------
# ADC setup
# -------------------------
adc = ADC(Pin(34))            # ADC pin (ensure a proper ADC-capable pin)
adc.atten(ADC.ATTN_11DB)     # Full 0-3.3V range

# -------------------------
# WiFi credentials
# -------------------------
SSID = 'Airtel_AIRTEL'
PASSWORD = 'Airtel@2007'

# -------------------------
# Server details
# -------------------------
SERVER_IP = '192.168.1.8'   # Raspberry Pi IP
SERVER_PORT = 12345

# -------------------------
# Connect to WiFi
# -------------------------
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

print("Connecting to WiFi...", end='')
while not wlan.isconnected():
    time.sleep(0.5)
    print('.', end='')

print('\nConnected. IP:', wlan.ifconfig()[0])

# -------------------------
# Connect to Raspberry Pi TCP server
# -------------------------
s = socket.socket()
while True:
    try:
        s.connect((SERVER_IP, SERVER_PORT))
        print(f"Connected to server at {SERVER_IP}:{SERVER_PORT}")
        break
    except OSError:
        print("Connection failed, retrying in 2s...")
        time.sleep(2)

# -------------------------
# Send ADC values in a loop
# -------------------------
try:
    while True:
        value = adc.read()  # 0–4095
        msg = f"{value}\n"
        print("ADC:", value)          # Optional: print on ESP32 console
        s.send(msg.encode('utf-8'))   # Send to Pi
        time.sleep(0.1)               # 100ms delay
except KeyboardInterrupt:
    print("Stopping...")
finally:
    s.close()

