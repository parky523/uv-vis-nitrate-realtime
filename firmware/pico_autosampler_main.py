from machine import Pin, I2C
import time

# I2C -- hardware I2C0, GP4=SDA GP5=SCL
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)

MCP_ADDR = 0x20
IODIRA   = 0x00
IODIRB   = 0x01
OLATA    = 0x14

DI_BIT     = 0x04  # GPA2
SAMPLE_BIT = 0x02  # GPA1
WASTE_BIT  = 0x01  # GPA0

SWITCH_PIN = 2
LED_PIN    = 3

led    = Pin(LED_PIN, Pin.OUT)
switch = Pin(SWITCH_PIN, Pin.IN, Pin.PULL_UP)

def mcp_write(reg, val):
    try:
        i2c.writeto_mem(MCP_ADDR, reg, bytes([val]))
        return True
    except:
        return False

def all_off():
    mcp_write(OLATA, 0x00)

def led_on():  led.value(1)
def led_off(): led.value(0)

# Init MCP23017
time.sleep_ms(100)
for _ in range(5):
    try:
        i2c.writeto_mem(MCP_ADDR, IODIRA, bytes([0x00]))
        i2c.writeto_mem(MCP_ADDR, IODIRB, bytes([0x00]))
        break
    except:
        time.sleep_ms(50)
all_off()

aborted    = False
wasAborted = False

def wait_ms(ms):
    global aborted
    started_low = (switch.value() == 0)
    start = time.ticks_ms()
    high_since = 0
    seen_high = False
    while time.ticks_diff(time.ticks_ms(), start) < ms:
        now_high = (switch.value() == 1)
        if started_low:
            if now_high:
                if not seen_high:
                    high_since = time.ticks_ms()
                    seen_high = True
                elif time.ticks_diff(time.ticks_ms(), high_since) >= 100:
                    all_off(); led_off(); aborted = True
                    print("STATUS: ABORTED")
                    return False
            else:
                seen_high = False
        time.sleep_ms(10)
    return True

def run_pump(bit, name, ms):
    if aborted: return
    all_off()
    ok = mcp_write(OLATA, bit)
    print("ON  -->", name, "[OK]" if ok else "[FAIL]")
    if not wait_ms(ms): return
    all_off()
    print("OFF -->", name)

def run_cleaning():
    global aborted, wasAborted
    wasAborted = False
    aborted = False
    led_on()
    print("=== CLEANING: 50 cycles DI 7.5s -> Drain 10s ===")
    for i in range(1, 51):
        ok = mcp_write(OLATA, DI_BIT)
        print("[{}ms] Cycle {}/50 DI ON{}".format(time.ticks_ms(), i, "" if ok else " [FAIL]"))
        time.sleep_ms(7500)
        ok = mcp_write(OLATA, WASTE_BIT)
        print("[{}ms] DRAIN ON{}".format(time.ticks_ms(), "" if ok else " [FAIL]"))
        time.sleep_ms(10000)
    all_off(); led_off()
    print("=== CLEANING COMPLETE ===")
    print("STATUS: STANDBY")

def run_sequence():
    global aborted, wasAborted
    aborted = False
    led_on()

    if wasAborted:
        wasAborted = False
        print("STATUS: CLEANUP_DRAIN")
        run_pump(WASTE_BIT, "Drain(GPA0)", 8000)
        if aborted: wasAborted = True; led_off(); return

    print("STATUS: SEQUENCE_STARTED")

    print("=== STEP 1: DI In ===")
    run_pump(DI_BIT, "DI(GPA2)", 8000)
    if aborted: wasAborted = True; led_off(); return
    time.sleep_ms(100)

    print("=== STEP 2: Drain ===")
    run_pump(WASTE_BIT, "Drain(GPA0)", 10000)
    if aborted: wasAborted = True; led_off(); return
    time.sleep_ms(500)

    print("=== STEP 3: DI Reference ===")
    run_pump(DI_BIT, "DI(GPA2)", 8000)
    if aborted: wasAborted = True; led_off(); return
    print("STATUS: REFERENCE_TRIGGER")
    if not wait_ms(5000): wasAborted = True; led_off(); return

    print("=== STEP 4: DI Drain ===")
    run_pump(WASTE_BIT, "Drain(GPA0)", 10000)
    if aborted: wasAborted = True; led_off(); return
    time.sleep_ms(100)

    print("=== STEP 5: Sample flush ===")
    run_pump(SAMPLE_BIT, "Sample(GPA1)", 10000)
    if aborted: wasAborted = True; led_off(); return
    time.sleep_ms(100)

    print("=== STEP 6: Drain ===")
    run_pump(WASTE_BIT, "Drain(GPA0)", 10000)
    if aborted: wasAborted = True; led_off(); return
    time.sleep_ms(100)

    print("=== STEP 7: skipped ===")

    print("=== STEP 8: Drain ===")
    run_pump(WASTE_BIT, "Drain(GPA0)", 10000)
    if aborted: wasAborted = True; led_off(); return
    time.sleep_ms(100)

    print("=== STEP 9: Sample measure ===")
    run_pump(SAMPLE_BIT, "Sample(GPA1)", 7000)
    if aborted: wasAborted = True; led_off(); return
    print("STATUS: MEASURE_NOW")
    if not wait_ms(10000): wasAborted = True; led_off(); return

    print("=== STEP 10: Drain ===")
    run_pump(WASTE_BIT, "Drain(GPA0)", 10000)
    if aborted: wasAborted = True; led_off(); return
    time.sleep_ms(100)

    print("=== STEP 11: DI rinse ===")
    run_pump(DI_BIT, "DI(GPA2)", 7000)
    if aborted: wasAborted = True; led_off(); return
    time.sleep_ms(100)

    print("=== STEP 12: Final drain ===")
    run_pump(WASTE_BIT, "Drain(GPA0)", 15000)
    if aborted: wasAborted = True; led_off(); return

    all_off(); led_off()
    print("STATUS: SEQUENCE_COMPLETE")
    print("STATUS: STANDBY")

# =====================================================
# MAIN LOOP
# =====================================================
print("=== AUTOSAMPLER v1.0 (Pico W) ===")
print("DI=GPA2  SAMPLE=GPA1  DRAIN=GPA0")
print("Switch GP2: LEFT=LOW=RUN  RIGHT=HIGH=STANDBY")
print("Commands: start | clean | di | sample | drain | off")
print("STATUS: STANDBY")

prev_high    = (switch.value() == 1)
low_since    = 0
arm_trigger  = False
import sys, select

while True:
    sw_high = (switch.value() == 1)
    if not sw_high:
        if prev_high:
            low_since = time.ticks_ms()
            arm_trigger = True
        if arm_trigger and time.ticks_diff(time.ticks_ms(), low_since) >= 500:
            arm_trigger = False
            print("Starting sequence...")
            run_sequence()
            prev_high = (switch.value() == 1)
            continue
    else:
        arm_trigger = False
    prev_high = sw_high

    # Check for serial command (non-blocking)
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        cmd = sys.stdin.readline().strip().lower()
        if cmd == "start" or cmd == "run":
            print("Starting sequence...")
            run_sequence()
        elif cmd == "clean":
            run_cleaning()
        elif cmd == "di":
            ok = mcp_write(OLATA, DI_BIT); led_on()
            print("DI ON 10min" if ok else "DI ON [FAIL]")
            time.sleep_ms(600000)
            all_off(); led_off(); print("DI OFF")
        elif cmd == "sample":
            ok = mcp_write(OLATA, SAMPLE_BIT); led_on()
            print("Sample ON 10min" if ok else "Sample ON [FAIL]")
            time.sleep_ms(600000)
            all_off(); led_off(); print("Sample OFF")
        elif cmd == "drain":
            run_pump(WASTE_BIT, "Drain(GPA0)", 30000)
        elif cmd == "off":
            all_off(); led_off()
            aborted = True; wasAborted = True
            print("ALL OFF")
        elif cmd.startswith("write ") or cmd.startswith("w "):
            v = cmd.split()[1]
            val = int(v, 16) if v.startswith("0x") else int(v)
            ok = mcp_write(OLATA, val)
            print("mcp_write 0x{:02x} {}".format(val, "[OK]" if ok else "[FAIL]"))

    time.sleep_ms(20)
