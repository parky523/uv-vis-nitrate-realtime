// SOFTWARE I2C  (A4=SDA, A5=SCL)
#define SDA_PIN  A4
#define SCL_PIN  A5
#define I2C_US   5
#define IODIRA  0x00
#define IODIRB  0x01
#define OLATA   0x14
#define DI_BIT     0x04   // GPA2
#define SAMPLE_BIT 0x02   // GPA1
#define WASTE_BIT  0x01   // GPA0
#define SWITCH_PIN 3
#define LED_PIN    4

void sdaH() { pinMode(SDA_PIN, INPUT_PULLUP); }
void sdaL() { pinMode(SDA_PIN, OUTPUT); digitalWrite(SDA_PIN, LOW); }

void i2c_recover() {
  pinMode(SCL_PIN, OUTPUT); pinMode(SDA_PIN, OUTPUT);
  digitalWrite(SDA_PIN, HIGH);
  for (int i = 0; i < 9; i++) {
    digitalWrite(SCL_PIN, HIGH); delayMicroseconds(10);
    digitalWrite(SCL_PIN, LOW);  delayMicroseconds(10);
  }
  digitalWrite(SDA_PIN, LOW);  delayMicroseconds(10);
  digitalWrite(SCL_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(SDA_PIN, HIGH); delayMicroseconds(10);
}

void i2c_init() { pinMode(SCL_PIN, OUTPUT); sdaH(); digitalWrite(SCL_PIN, HIGH); }

void i2c_start() {
  sdaH(); delayMicroseconds(I2C_US); digitalWrite(SCL_PIN, HIGH); delayMicroseconds(I2C_US);
  sdaL(); delayMicroseconds(I2C_US); digitalWrite(SCL_PIN, LOW);  delayMicroseconds(I2C_US);
}
void i2c_stop() {
  sdaL(); delayMicroseconds(I2C_US); digitalWrite(SCL_PIN, HIGH); delayMicroseconds(I2C_US);
  sdaH(); delayMicroseconds(I2C_US);
}
bool i2c_byte(uint8_t data) {
  for (int i = 7; i >= 0; i--) {
    if ((data >> i) & 1) sdaH(); else sdaL();
    delayMicroseconds(I2C_US);
    digitalWrite(SCL_PIN, HIGH); delayMicroseconds(I2C_US);
    digitalWrite(SCL_PIN, LOW);  delayMicroseconds(I2C_US);
  }
  sdaH(); delayMicroseconds(I2C_US);
  digitalWrite(SCL_PIN, HIGH); delayMicroseconds(I2C_US);
  bool ack = (digitalRead(SDA_PIN) == LOW);
  digitalWrite(SCL_PIN, LOW);  delayMicroseconds(I2C_US);
  return ack;
}
bool mcp_write(uint8_t reg, uint8_t val) {
  for (int r = 0; r < 5; r++) {
    i2c_start();
    bool ok = i2c_byte(0x40); ok &= i2c_byte(reg); ok &= i2c_byte(val);
    i2c_stop();
    if (ok) return true;
    i2c_recover(); delay(5);
  }
  return false;
}
void allOff() { mcp_write(OLATA, 0x00); }
void ledOn()  { digitalWrite(LED_PIN, HIGH); }
void ledOff() { digitalWrite(LED_PIN, LOW); }

bool aborted = false, wasAborted = false, sequencePending = false;

bool waitMs(unsigned long ms) {
  bool startedLow = (digitalRead(SWITCH_PIN) == LOW);
  unsigned long start = millis(), highSince = 0;
  bool seenHigh = false;
  while (millis() - start < ms) {
    bool nowHigh = (digitalRead(SWITCH_PIN) == HIGH);
    if (startedLow) {
      if (nowHigh) {
        if (!seenHigh) { highSince = millis(); seenHigh = true; }
        else if (millis() - highSince >= 100) {
          allOff(); ledOff(); aborted = true;
          Serial.println("STATUS: ABORTED"); Serial.flush(); return false;
        }
      } else seenHigh = false;
    }
    delay(10);
  }
  return true;
}

void runPump(uint8_t bit, const char* name, unsigned long ms) {
  if (aborted) return;
  allOff();
  bool ok = mcp_write(OLATA, bit);
  Serial.print("ON --> "); Serial.print(name);
  Serial.println(ok ? " [OK]" : " [FAIL]"); Serial.flush();
  if (!waitMs(ms)) return;
  allOff();
  Serial.print("OFF --> "); Serial.println(name); Serial.flush();
}

void runCleaning() {
  sequencePending = false; ledOn();
  Serial.println("=== CLEANING: 50 cycles DI 7.5s -> Drain 10s ==="); Serial.flush();
  for (int i = 1; i <= 50; i++) {
    bool ok = mcp_write(OLATA, DI_BIT);
    Serial.print("["); Serial.print(millis()); Serial.print("ms] Cycle "); Serial.print(i);
    Serial.println(ok ? "/50 DI ON" : "/50 DI ON [FAIL]"); Serial.flush();
    delay(7500);
    ok = mcp_write(OLATA, WASTE_BIT);
    Serial.print("["); Serial.print(millis());
    Serial.println(ok ? "ms] DRAIN ON" : "ms] DRAIN ON [FAIL]"); Serial.flush();
    delay(10000);
  }
  allOff(); ledOff(); sequencePending = false; wasAborted = false;
  Serial.println("=== CLEANING COMPLETE ==="); Serial.println("STATUS: STANDBY"); Serial.flush();
}

void runSequence() {
  aborted = false; ledOn();
  if (wasAborted) {
    wasAborted = false;
    Serial.println("STATUS: CLEANUP_DRAIN"); Serial.flush();
    runPump(WASTE_BIT, "Drain(GPA0)", 8000);
    if (aborted) { wasAborted = true; ledOff(); return; }
  }
  Serial.println("STATUS: SEQUENCE_STARTED"); Serial.flush();
  Serial.println("=== STEP 1: DI In ==="); Serial.flush();
  runPump(DI_BIT, "DI(GPA2)", 8000);
  if (aborted) { wasAborted = true; ledOff(); return; } delay(100);
  Serial.println("=== STEP 2: Drain ==="); Serial.flush();
  runPump(WASTE_BIT, "Drain(GPA0)", 10000);
  if (aborted) { wasAborted = true; ledOff(); return; } delay(500);
  Serial.println("=== STEP 3: DI Reference ==="); Serial.flush();
  runPump(DI_BIT, "DI(GPA2)", 8000);
  if (aborted) { wasAborted = true; ledOff(); return; }
  Serial.println("STATUS: REFERENCE_TRIGGER"); Serial.flush();
  if (!waitMs(5000)) { wasAborted = true; ledOff(); return; }
  Serial.println("=== STEP 4: DI Drain ==="); Serial.flush();
  runPump(WASTE_BIT, "Drain(GPA0)", 10000);
  if (aborted) { wasAborted = true; ledOff(); return; } delay(100);
  Serial.println("=== STEP 5: Sample flush ==="); Serial.flush();
  runPump(SAMPLE_BIT, "Sample(GPA1)", 10000);
  if (aborted) { wasAborted = true; ledOff(); return; } delay(100);
  Serial.println("=== STEP 6: Drain ==="); Serial.flush();
  runPump(WASTE_BIT, "Drain(GPA0)", 10000);
  if (aborted) { wasAborted = true; ledOff(); return; } delay(100);
  Serial.println("=== STEP 7: skipped ==="); Serial.flush();
  Serial.println("=== STEP 8: Drain ==="); Serial.flush();
  runPump(WASTE_BIT, "Drain(GPA0)", 10000);
  if (aborted) { wasAborted = true; ledOff(); return; } delay(100);
  Serial.println("=== STEP 9: Sample measure ==="); Serial.flush();
  runPump(SAMPLE_BIT, "Sample(GPA1)", 7000);
  if (aborted) { wasAborted = true; ledOff(); return; }
  Serial.println("STATUS: MEASURE_NOW"); Serial.flush();
  if (!waitMs(10000)) { wasAborted = true; ledOff(); return; }
  Serial.println("=== STEP 10: Drain ==="); Serial.flush();
  runPump(WASTE_BIT, "Drain(GPA0)", 10000);
  if (aborted) { wasAborted = true; ledOff(); return; } delay(100);
  Serial.println("=== STEP 11: DI rinse ==="); Serial.flush();
  runPump(DI_BIT, "DI(GPA2)", 7000);
  if (aborted) { wasAborted = true; ledOff(); return; } delay(100);
  Serial.println("=== STEP 12: Final drain ==="); Serial.flush();
  runPump(WASTE_BIT, "Drain(GPA0)", 15000);
  if (aborted) { wasAborted = true; ledOff(); return; }
  allOff(); ledOff();
  Serial.println("STATUS: SEQUENCE_COMPLETE"); Serial.println("STATUS: STANDBY"); Serial.flush();
}

void handleSerial(String cmd) {
  cmd.trim(); cmd.toLowerCase();
  if      (cmd == "start" || cmd == "run") { sequencePending = true; }
  else if (cmd == "clean")  { runCleaning(); }
  else if (cmd == "di")     { bool ok = mcp_write(OLATA, DI_BIT); ledOn(); Serial.println(ok ? "DI ON 10min" : "DI ON [FAIL]"); Serial.flush(); delay(600000UL); allOff(); ledOff(); Serial.println("DI OFF"); Serial.flush(); }
  else if (cmd == "sample") { bool ok = mcp_write(OLATA, SAMPLE_BIT); ledOn(); Serial.println(ok ? "Sample ON 10min" : "Sample ON [FAIL]"); Serial.flush(); delay(600000UL); allOff(); ledOff(); Serial.println("Sample OFF"); Serial.flush(); }
  else if (cmd == "drain")  { runPump(WASTE_BIT, "Drain(GPA0)", 30000); }
  else if (cmd == "off")    { allOff(); ledOff(); aborted = true; wasAborted = true; sequencePending = false; Serial.println("ALL OFF"); Serial.flush(); }
  else if (cmd.startsWith("write ") || cmd.startsWith("w ")) {
    String v = cmd.substring(cmd.indexOf(' ') + 1); v.trim();
    uint8_t val = (v.startsWith("0x") || v.startsWith("0X")) ? (uint8_t)strtol(v.c_str(), NULL, 16) : (uint8_t)v.toInt();
    bool ok = mcp_write(OLATA, val);
    Serial.print("mcp_write 0x"); Serial.print(val, HEX); Serial.println(ok ? " [OK]" : " [FAIL]"); Serial.flush();
  }
  else if (cmd == "status") { Serial.println("STATUS: STANDBY"); Serial.flush(); }
}

bool prevHigh = true;
static unsigned long lowSince = 0;
static bool armTrigger = false;

void setup() {
  Serial.begin(9600);
  unsigned long _t = millis();
  while (!Serial && millis() - _t < 3000) { delay(10); }
  pinMode(LED_PIN, OUTPUT); ledOff();
  pinMode(SWITCH_PIN, INPUT_PULLUP);
  TWCR = 0;
  i2c_recover(); delay(100);
  i2c_init(); delay(50);
  for (int r = 0; r < 3; r++) {
    if (mcp_write(IODIRA, 0x00) && mcp_write(IODIRB, 0x00)) break;
    delay(50);
  }
  allOff();
  Serial.println("=== AUTOSAMPLER v11 ===");
  Serial.println("DI=GPA2 SAMPLE=GPA1 DRAIN=GPA0");
  Serial.println("Commands: start|clean|di|sample|drain|off|write <val>");
  Serial.print("Switch="); Serial.println(digitalRead(SWITCH_PIN)==LOW?"LOW(RUN)":"HIGH(STANDBY)");
  Serial.println("STATUS: STANDBY"); Serial.flush();
  prevHigh = (digitalRead(SWITCH_PIN) == HIGH);
}

void loop() {
  bool swHigh = (digitalRead(SWITCH_PIN) == HIGH);
  if (!swHigh) {
    if (prevHigh) { lowSince = millis(); armTrigger = true; }
    if (armTrigger && millis() - lowSince >= 500) { sequencePending = true; armTrigger = false; }
  } else { armTrigger = false; }
  prevHigh = swHigh;
  if (Serial.available()) { String cmd = Serial.readStringUntil('\n'); handleSerial(cmd); }
  if (sequencePending) {
    sequencePending = false;
    while (Serial.available()) Serial.read();
    Serial.println("Starting sequence..."); Serial.flush();
    runSequence();
    prevHigh = (digitalRead(SWITCH_PIN) == HIGH);
  }
}
