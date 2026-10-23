import serial, time
ser = serial.Serial('COM10', 9600, timeout=1)
time.sleep(2)
print('--- Flip switch LEFT and RIGHT now ---')
for _ in range(20):
    l = ser.readline().decode(errors='replace').strip()
    if l: print(l)
ser.close()
