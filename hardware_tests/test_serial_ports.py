import serial, serial.tools.list_ports, time

# Find the Aris port
print("Available ports:")
for p in serial.tools.list_ports.comports():
    print(f"  {p.device}  —  {p.description}  [{p.hwid}]")

# Try the one that shows VID_354F
for p in serial.tools.list_ports.comports():
    if "354F" in p.hwid:
        print(f"\nFound Aris on {p.device}")
        try:
            s = serial.Serial(p.device, baudrate=115200, timeout=2)
            print("Opened OK")
            # Try sending a question mark / identity command
            for cmd in [b'?', b'*IDN?', b'\x00', b'i', b'I', b'S', b's']:
                s.write(cmd + b'\r\n')
                time.sleep(0.1)
                r = s.read_all()
                if r:
                    print(f"  CMD {cmd}: reply = {r}")
            s.close()
        except Exception as e:
            print(f"  Error: {e}")