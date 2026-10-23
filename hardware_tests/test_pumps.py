import serial, time

ser = serial.Serial('COM10', 9600, timeout=2)
time.sleep(3)
ser.reset_input_buffer()

def test(command, label, on_time=0):
    ser.write((command + '\n').encode())
    if on_time > 0:
        time.sleep(on_time)
    time.sleep(0.8)
    resp = ser.read_all().decode(errors='replace').strip()
    result = resp if resp else "(no response)"
    print(label + ": " + result)

print("=== FULL CONNECTION TEST ===")

test('write 0x01', 'GPA0 Drain ON',  3)
test('write 0x00', 'GPA0 OFF')
time.sleep(1)

test('write 0x02', 'GPA1 DI ON',     3)
test('write 0x00', 'GPA1 OFF')
time.sleep(1)

test('write 0x08', 'GPA3 Sample ON', 3)
test('write 0x00', 'GPA3 OFF')
time.sleep(1)

test('status',     'Status')

ser.close()
print("=== DONE ===")
