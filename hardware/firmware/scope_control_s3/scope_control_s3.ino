/*
  scope_control_s3.ino
  ============================================================================
  RUNTIME control firmware for the autonomous-endoscope rig (ESP32-S3).

  This is the deployment sibling of the bring-up sketches in this folder. It is
  NOT interactive and NOT a calibration tool -- it speaks the binary
  scope_link protocol to the WSL ROS node `scope_link` and drives the four
  tendon servos to follow the policy's steering setpoints, with a safety
  governor on top.

      scope_link (WSL)  <--UART 921600, COBS+CRC16-->  this firmware
        /scope/action  ->  SETPOINT  ->  [normalised -1..1 -> ticks (asymmetric
                                          per-direction MAX_PULL), SYNC WRITE the pairs]
        /scope/command ->  ENABLE / DISABLE / CLEAR_SAFETY / ESTOP
                       <-  TELEMETRY (50 Hz): seq echo, tip_contact, tension,
                                              servo pos + current, safety state

  Protocol is defined once in
    deploy/ros2_ws/src/scope_control/scope_control/scope_link_proto.py
  -- keep this file in lockstep and bump PROTO_VERSION together.

  --------------------------------------------------------------- BOARD
  Board:            "ESP32S3 Dev Module"
  USB CDC On Boot:  Disabled     -> Serial == UART0 == the CP2102 "UART" jack.
  FLASH via:        the NATIVE-USB jack (leave the CP2102 jack forwarded to WSL
                    via usbipd for the protocol link -- see the ROS 2 deployment
                    memory).
  Serial (UART0):   921600 8N1  -- the scope_link protocol. Do NOT Serial.print
                    plain text here; all human-readable output goes out as
                    T_LOG frames (fwlog()).

  ----------------------------------------------------------------- PINS
    Servo bus (UART1, 1 Mbaud 8N1)   -- same as scope_bringup_s3.ino
      GPIO17  TX -> FE-URT-2 RX
      GPIO18  RX <- FE-URT-2 TX
      GND        -- common with FE-URT-2 and the servo supply

    HX711 cable-tension amps (bit-banged, shared clock)
      GPIO14  SCK  -> all four HX711 PD_SCK
      DOUT, IN SERVO ORDER (see hardware/bringup/measured_constants.md +
      the "HX711 sensor <-> servo mapping" memo -- the sensors are NOT wired
      in servo-ID order):
        servo 1 -> GPIO10
        servo 2 -> GPIO11
        servo 3 -> GPIO12
        servo 4 -> GPIO13
      HX711 RATE pin: tie HIGH for 80 SPS. If left low (default) you get
      10 SPS and the tension safety path is only ~10 Hz -- the fast path then
      leans on servo current instead. Set HX_RATE_PIN below if you wire it.

    E-stop authority (safety core owns this alone -- a bus torque-off is NOT a
    safety stop, see the deployment memory)
      ESTOP_MOSFET_PIN  -> gate of the high-side switch in the SERVO SUPPLY.
      *** TODO: confirm the pin and the polarity against the actual board. ***

    Tip-contact flex sensor  -> voltage divider -> ADC pin.
      *** TODO: not wired / not calibrated yet. TIP_CONTACT_FORCE_ZERO keeps
          the reported flag at 0 until then. ***

  ------------------------------------------------------ WHAT IS STILL A GUESS
  Marked "TODO/VERIFY" below and safe to run only on the bench until checked:
    - AXIS_SERVO / AXIS_INVERT  (which servo pair is cmd_x vs cmd_y, and the
                                 steering sign -- watch the image and flip)
    - ESTOP_MOSFET_PIN + ESTOP_ENABLE_LEVEL
    - CURRENT_LSB_MA        (STS3032 REG 69 scale)
  MAXPULL_DEFAULT is the measured `bend` result (2026-09-09); scope_link's
  CONFIG frame overrides it from scope_params.yaml, so it need not be exact.
    - TIP_CONTACT_* pins/threshold
*/

#include <string.h>
#include <stdio.h>
#include <math.h>

// ======================================================================
//  CONFIG
// ======================================================================

static const uint8_t  PROTO_VERSION = 2;   // must match scope_link_proto.py

// ---- pins ----
static const int8_t  BUS_TX_PIN = 17;
static const int8_t  BUS_RX_PIN = 18;
static const uint8_t HX_SCK     = 14;
static const uint8_t HX_DT_FOR_SERVO[4] = { 10, 11, 12, 13 };  // index = servo (id-1)
static const int8_t  HX_RATE_PIN = -1;      // set to the GPIO if you wire RATE high (80 SPS)

static const int8_t  ESTOP_MOSFET_PIN   = 21;    // TODO verify
static const uint8_t ESTOP_ENABLE_LEVEL = HIGH;  // level that ENABLES the servo supply

static const int8_t  TIP_CONTACT_PIN = 4;        // ADC1_CH3; TODO verify
static const bool     TIP_CONTACT_FORCE_ZERO = true;   // sensor not wired yet
static const int      TIP_CONTACT_ADC_THRESHOLD = 1800; // TODO calibrate

// ---- tendon geometry ----
static const int16_t ZERO_TICK     = 2048;                // EEPROM zero from `setzero`
static const int16_t SEAM          = 100;                 // stay this far from the 0/4095 wrap
static const int16_t MAX_ABS_TICKS_FROM_ZERO = 950;       // hard outer clamp (~83 deg); a bad CONFIG can't drive past this

// Per-axis, per-direction bend limits in ENCODER TICKS -- the rig's + and -
// directions are ASYMMETRIC (measured 2026-09-09 with `bend`;
// hardware/bringup/measured_constants.md). The normalised command -1..+1 is
// mapped to ticks with these. Compiled defaults; CONFIG overrides at runtime.
// Index [axis][0] = + direction, [axis][1] = - direction.
static const uint16_t MAXPULL_DEFAULT[2][2] = { { 724, 681 }, { 840, 603 } };
static const float    MAXPULL_HEADROOM_DEFAULT = 0.9f;    // bend was measured tip-free; friction reduces it

// Axis -> antagonistic servo pair. AXIS_SERVO[axis] = { pull+, pull- }.
// axis 0 = ScopeAction.cmd_x_n, axis 1 = cmd_y_n.
// The two servos of a pair share a PULL_SIGN (measured 2026-09-08).
static const uint8_t AXIS_SERVO[2][2] = { { 1, 3 }, { 2, 4 } };   // TODO verify which is X vs Z
static const bool    AXIS_INVERT[2]   = { false, false };         // flip if an axis steers backwards
static const int8_t  PULL_SIGN[5]     = { 0, -1, +1, -1, +1 };    // [servo id]; 1&3 pull on -, 2&4 on +

// ---- servo motion ----
static const uint16_t SERVO_SPEED = 3000;   // per-setpoint interpolation speed (0 = max)
static const uint8_t  SERVO_ACC   = 0;      // acceleration (0 = disabled)
// Firmware-side slew clamp: the most any servo goal may move per motor cycle.
// A second line of defence against a policy glitch or a garbled-but-valid
// setpoint -- bounds the tip speed no matter what is commanded.
static const int16_t  MAX_TICKS_PER_CYCLE = 8;   // @200 Hz -> ~1600 ticks/s ~= 140 deg/s

// ---- safety (compiled defaults; CONFIG frame overrides at runtime) ----
static const float    CURRENT_LSB_MA = 6.5f;     // STS3032 REG 69 scale; TODO verify

struct SafetyCfg {
  float    current_limit_ma  = 800.0f;
  float    tension_limit_g   = 250.0f;
  uint16_t spike_debounce_ms = 20;
  uint16_t comms_timeout_ms  = 120;
  uint16_t motor_hz          = 200;
  uint16_t telem_hz          = 50;
  uint16_t maxpull[2][2]     = { { MAXPULL_DEFAULT[0][0], MAXPULL_DEFAULT[0][1] },
                                 { MAXPULL_DEFAULT[1][0], MAXPULL_DEFAULT[1][1] } };
  float    maxpull_headroom  = MAXPULL_HEADROOM_DEFAULT;
};
static SafetyCfg cfg;

// applied tick limit for axis (0=x,1=z), direction (positive?), headroom folded in
static int16_t appliedMaxPull(uint8_t axis, bool positive) {
  float raw = (float)cfg.maxpull[axis][positive ? 0 : 1];
  int16_t v = (int16_t)lroundf(raw * cfg.maxpull_headroom);
  if (v < 1) v = 1;
  if (v > MAX_ABS_TICKS_FROM_ZERO) v = MAX_ABS_TICKS_FROM_ZERO;
  return v;
}

// ---- HX711 tension calibration (servo order) ----
// counts/gram is one shared factor (hardware/bringup/measured_constants.md).
static const float HX_COUNTS_PER_GRAM = 1105.0488f;
// no-load offsets, RE-ORDERED into servo order from the GPIO-labelled table
// (ch1/GPIO13 = servo 4, ... ch4/GPIO10 = servo 1):
static const long  HX_NOLOAD[4] = {
  -55713,   // servo 1  (GPIO10, was "ch4")
  -73492,   // servo 2  (GPIO11, was "ch3")
  -62355,   // servo 3  (GPIO12, was "ch2")
  -56125,   // servo 4  (GPIO13, was "ch1")
};

// ======================================================================
//  PROTOCOL: message types, framing (COBS + CRC-16/CCITT-FALSE)
// ======================================================================

enum : uint8_t {
  T_SETPOINT  = 0x01,
  T_CONFIG    = 0x02,
  T_COMMAND   = 0x03,
  T_PING      = 0x04,
  T_TELEMETRY = 0x81,
  T_LOG       = 0x82,
  T_PONG      = 0x84,
  T_HELLO     = 0x85,
};

// ScopeCommand values
enum : uint8_t { CMD_DISABLE = 0, CMD_ENABLE = 1, CMD_CLEAR_SAFETY = 2, CMD_ESTOP = 3 };

// telemetry flag bits
static const uint8_t FLAG_TIP_CONTACT  = 0x01;
static const uint8_t FLAG_SAFETY_SHIFT = 1;
static const uint8_t FLAG_SAFETY_MASK  = 0x03;
static const uint8_t FLAG_ENABLED      = 0x08;

// safety_state values (must match scope_msgs/ScopeTelemetry)
enum : uint8_t { SAFETY_OK = 0, SAFETY_CLAMP = 1, SAFETY_HOLD = 2, SAFETY_ESTOP = 3 };

// log levels
enum : uint8_t { LOG_DBG = 0, LOG_INFO = 1, LOG_WARN = 2, LOG_ERR = 3 };

static uint16_t crc16(const uint8_t* d, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)d[i] << 8;
    for (int b = 0; b < 8; b++)
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

// COBS encode: out must hold n + n/254 + 2 bytes. Returns encoded length.
static size_t cobsEncode(const uint8_t* in, size_t n, uint8_t* out) {
  size_t codeIdx = 0, w = 1;
  uint8_t code = 1;
  for (size_t i = 0; i < n; i++) {
    if (in[i] != 0) {
      out[w++] = in[i];
      if (++code == 0xFF) { out[codeIdx] = code; codeIdx = w++; code = 1; }
    } else {
      out[codeIdx] = code; codeIdx = w++; code = 1;
    }
  }
  out[codeIdx] = code;
  return w;
}

// COBS decode. Returns decoded length, or -1 on a malformed segment.
static int cobsDecode(const uint8_t* in, size_t n, uint8_t* out, size_t outCap) {
  size_t r = 0, w = 0;
  while (r < n) {
    uint8_t code = in[r++];
    if (code == 0) return -1;
    for (uint8_t i = 1; i < code; i++) {
      if (r >= n || w >= outCap) return -1;
      out[w++] = in[r++];
    }
    if (code != 0xFF && r < n) {
      if (w >= outCap) return -1;
      out[w++] = 0;
    }
  }
  return (int)w;
}

static void sendFrame(uint8_t type, const uint8_t* payload, size_t plen) {
  uint8_t body[132];
  if (plen > sizeof(body) - 3) return;
  body[0] = type;
  memcpy(body + 1, payload, plen);
  uint16_t c = crc16(body, 1 + plen);
  body[1 + plen] = (uint8_t)(c & 0xFF);
  body[2 + plen] = (uint8_t)(c >> 8);
  uint8_t framed[160];
  size_t fl = cobsEncode(body, plen + 3, framed);
  framed[fl++] = 0x00;
  Serial.write(framed, fl);
}

static void fwlog(uint8_t level, const char* msg) {
  uint8_t buf[120];
  buf[0] = level;
  size_t n = strlen(msg);
  if (n > sizeof(buf) - 1) n = sizeof(buf) - 1;
  memcpy(buf + 1, msg, n);
  sendFrame(T_LOG, buf, n + 1);
}

// little-endian scalar readers/writers (ESP32 is LE; payloads use "<" layout)
static float    rdF32(const uint8_t* p) { float v; memcpy(&v, p, 4); return v; }
static uint32_t rdU32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint16_t rdU16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
static void wrF32(uint8_t* p, float v)    { memcpy(p, &v, 4); }
static void wrU32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }
static void wrU16(uint8_t* p, uint16_t v) { memcpy(p, &v, 2); }
static void wrI16(uint8_t* p, int16_t v)  { memcpy(p, &v, 2); }

// ======================================================================
//  SERVO BUS (Feetech SCS / Dynamixel 1.0 framing) -- from scope_bringup_s3
// ======================================================================

static const uint8_t BROADCAST_ID   = 0xFE;
static const uint8_t INST_PING       = 0x01;
static const uint8_t INST_READ       = 0x02;
static const uint8_t INST_WRITE      = 0x03;
static const uint8_t INST_SYNC_WRITE = 0x83;

static const uint8_t REG_TORQUE_EN   = 40;
static const uint8_t REG_ACC         = 41;   // 7-byte block: acc,posL,posH,timeL,timeH,spdL,spdH
static const uint8_t REG_PRESENT_POS = 56;
static const uint8_t REG_PRESENT_LD  = 60;
static const uint8_t REG_CURRENT     = 69;

static uint8_t sbTx[64], sbRx[64];
static uint8_t sbTxLen = 0;

static void busOpen() {
  Serial1.end();
  Serial1.setRxBufferSize(256);
  Serial1.begin(1000000, SERIAL_8N1, BUS_RX_PIN, BUS_TX_PIN);
  delay(5);
  while (Serial1.read() != -1) {}
}

static void sbSend(uint8_t id, uint8_t inst, const uint8_t* params, uint8_t nParams) {
  uint8_t sum = id + (uint8_t)(nParams + 2) + inst;
  sbTx[0] = 0xFF; sbTx[1] = 0xFF; sbTx[2] = id;
  sbTx[3] = nParams + 2; sbTx[4] = inst;
  for (uint8_t i = 0; i < nParams; i++) { sbTx[5 + i] = params[i]; sum += params[i]; }
  sbTx[5 + nParams] = ~sum;
  sbTxLen = 6 + nParams;
  while (Serial1.read() != -1) {}
  Serial1.write(sbTx, sbTxLen);
  Serial1.flush();
}

// Read until `want` bytes are in, or `timeoutMs` elapses. Returns as soon as it
// has enough -- it does NOT wait for a quiet gap (that cost the bring-up tool
// ~4 ms per transaction, which the 200 Hz motor loop cannot afford).
static uint8_t sbCollectN(uint8_t want, uint16_t timeoutMs) {
  uint8_t n = 0;
  unsigned long t0 = millis();
  while (n < want && n < sizeof(sbRx) && (millis() - t0) < timeoutMs) {
    int c = Serial1.read();
    if (c >= 0) sbRx[n++] = (uint8_t)c;
  }
  return n;
}

// find a checksum-valid status packet from `id` carrying nParams bytes, skipping our echo
static bool sbParse(uint8_t id, uint8_t rxLen, uint8_t* params, uint8_t nParams) {
  uint8_t start = 0;
  if (rxLen >= sbTxLen && memcmp(sbRx, sbTx, sbTxLen) == 0) start = sbTxLen;
  uint8_t total = nParams + 6;
  for (int i = start; i + total <= rxLen; i++) {
    if (sbRx[i] != 0xFF || sbRx[i + 1] != 0xFF) continue;
    if (sbRx[i + 2] != id || sbRx[i + 3] != nParams + 2) continue;
    uint8_t sum = 0;
    for (uint8_t k = 2; k <= nParams + 4; k++) sum += sbRx[i + k];
    if ((uint8_t)(~sum) != sbRx[i + nParams + 5]) continue;
    for (uint8_t k = 0; k < nParams; k++) params[k] = sbRx[i + 5 + k];
    return true;
  }
  return false;
}

// expected reply length = adapter echo (sbTxLen) + status packet (6 + nParams)
static const uint16_t SB_TIMEOUT_MS = 3;

static bool sbPing(uint8_t id) {
  sbSend(id, INST_PING, nullptr, 0);
  return sbParse(id, sbCollectN(sbTxLen + 6, SB_TIMEOUT_MS), nullptr, 0);
}

static bool sbRead16(uint8_t id, uint8_t addr, long* out) {
  uint8_t p[2] = { addr, 2 }, v[2];
  sbSend(id, INST_READ, p, 2);
  if (!sbParse(id, sbCollectN(sbTxLen + 8, SB_TIMEOUT_MS), v, 2)) return false;
  *out = (long)((uint16_t)v[0] | ((uint16_t)v[1] << 8));
  return true;
}

static bool sbWrite8(uint8_t id, uint8_t addr, uint8_t val) {
  uint8_t p[2] = { addr, val };
  sbSend(id, INST_WRITE, p, 2);
  if (id == BROADCAST_ID) return true;
  return sbParse(id, sbCollectN(sbTxLen + 6, SB_TIMEOUT_MS), nullptr, 0);
}

static bool sbWritePos1(uint8_t id, int16_t pos, uint16_t spd, uint8_t acc) {
  if (pos < 0) pos = 0;
  if (pos > 4095) pos = 4095;
  uint8_t p[8] = { REG_ACC, acc,
                   (uint8_t)(pos & 0xFF), (uint8_t)((pos >> 8) & 0xFF),
                   0, 0,
                   (uint8_t)(spd & 0xFF), (uint8_t)((spd >> 8) & 0xFF) };
  sbSend(id, INST_WRITE, p, 8);
  return sbParse(id, sbCollectN(sbTxLen + 6, SB_TIMEOUT_MS), nullptr, 0);
}

// One packet, all four servos start together. goal indexed by servo id 1..4.
static void sbSyncWritePos4(const int16_t goal[5], uint16_t spd, uint8_t acc) {
  const uint8_t L = 7, N = 4;
  uint8_t buf[64];
  uint8_t n = 0;
  buf[n++] = 0xFF; buf[n++] = 0xFF; buf[n++] = BROADCAST_ID;
  buf[n++] = (L + 1) * N + 4;            // LEN
  buf[n++] = INST_SYNC_WRITE;
  buf[n++] = REG_ACC;                     // start address
  buf[n++] = L;
  uint8_t sum = BROADCAST_ID + buf[3] + INST_SYNC_WRITE + REG_ACC + L;
  for (uint8_t id = 1; id <= 4; id++) {
    int16_t p = goal[id];
    if (p < 0) p = 0;
    if (p > 4095) p = 4095;
    const uint8_t rec[8] = { id, acc,
                             (uint8_t)(p & 0xFF), (uint8_t)((p >> 8) & 0xFF),
                             0, 0,
                             (uint8_t)(spd & 0xFF), (uint8_t)((spd >> 8) & 0xFF) };
    for (uint8_t k = 0; k < 8; k++) { buf[n++] = rec[k]; sum += rec[k]; }
  }
  buf[n++] = ~sum;
  while (Serial1.read() != -1) {}
  Serial1.write(buf, n);
  Serial1.flush();
}

// ======================================================================
//  HX711 (non-blocking: only reads when all four DOUT are ready)
// ======================================================================

static long hxRaw[4] = { 0, 0, 0, 0 };       // servo order
static float hxGrams[4] = { 0, 0, 0, 0 };

static bool hxAllReady() {
  for (uint8_t s = 0; s < 4; s++)
    if (digitalRead(HX_DT_FOR_SERVO[s]) != LOW) return false;
  return true;
}

// ~125 us with interrupts off. Call only when hxAllReady().
static void hxReadNow() {
  uint32_t v[4] = { 0, 0, 0, 0 };
  noInterrupts();
  for (uint8_t i = 0; i < 24; i++) {
    digitalWrite(HX_SCK, HIGH);
    delayMicroseconds(1);
    for (uint8_t s = 0; s < 4; s++)
      v[s] = (v[s] << 1) | (digitalRead(HX_DT_FOR_SERVO[s]) ? 1u : 0u);
    digitalWrite(HX_SCK, LOW);
    delayMicroseconds(1);
  }
  digitalWrite(HX_SCK, HIGH); delayMicroseconds(1);   // 25th pulse: next = ch A gain 128
  digitalWrite(HX_SCK, LOW);  delayMicroseconds(1);
  interrupts();
  for (uint8_t s = 0; s < 4; s++) {
    if (v[s] & 0x800000) v[s] |= 0xFF000000;
    hxRaw[s]   = (long)(int32_t)v[s];
    hxGrams[s] = (hxRaw[s] - HX_NOLOAD[s]) / HX_COUNTS_PER_GRAM;
  }
}

// ======================================================================
//  STATE
// ======================================================================

enum SysState : uint8_t { ST_BOOT, ST_HOLD, ST_RUN, ST_ESTOP };
static SysState g_state = ST_BOOT;
static bool     g_enableIntent = false;   // operator wants RUN
static uint8_t  g_holdReason   = 4;       // 0 none, 1 comms, 2 current, 3 tension, 4 boot/disable
static bool     g_allServos    = false;

// latest setpoint from the host: NORMALISED command per pair, -1..+1
static float    g_setXn = 0.0f, g_setYn = 0.0f;
static uint32_t g_setSeq  = 0;
static uint32_t g_lastSetpointMs = 0;

// per-servo working state (index by id: [1..4])
static int16_t g_goalTicks[5]   = { 0, ZERO_TICK, ZERO_TICK, ZERO_TICK, ZERO_TICK };  // slew-limited target
static int16_t g_holdTicks[5]   = { 0, ZERO_TICK, ZERO_TICK, ZERO_TICK, ZERO_TICK };
static int16_t g_presentTicks[5]= { 0, ZERO_TICK, ZERO_TICK, ZERO_TICK, ZERO_TICK };
static int16_t g_currentMa[5]   = { 0, 0, 0, 0, 0 };
static int16_t g_loadRaw[5]     = { 0, 0, 0, 0, 0 };

// safety debounce: millis() a channel first went over-limit (0 = not over)
static uint32_t g_curOverSince[5] = { 0, 0, 0, 0, 0 };
static uint32_t g_tenOverSince[5] = { 0, 0, 0, 0, 0 };

static bool g_tipContact = false;

static uint8_t safetyStateForTelem() {
  if (g_state == ST_ESTOP) return SAFETY_ESTOP;
  if (g_state == ST_HOLD)  return SAFETY_HOLD;
  return SAFETY_OK;
}

static void servoSupply(bool enable) {
  if (ESTOP_MOSFET_PIN >= 0)
    digitalWrite(ESTOP_MOSFET_PIN, enable ? ESTOP_ENABLE_LEVEL : !ESTOP_ENABLE_LEVEL);
}

// ---- state transitions ----

static void captureHold() {
  for (uint8_t id = 1; id <= 4; id++) g_holdTicks[id] = g_presentTicks[id];
}

static void enterHold(uint8_t reason) {
  if (g_state == ST_RUN) captureHold();
  g_state = ST_HOLD;
  g_holdReason = reason;
  for (uint8_t id = 1; id <= 4; id++) g_goalTicks[id] = g_holdTicks[id];
  const char* r = reason == 1 ? "comms timeout" : reason == 2 ? "servo current" :
                  reason == 3 ? "cable tension" : "disable";
  char m[48]; snprintf(m, sizeof(m), "SAFE_HOLD (%s)", r);
  fwlog(LOG_WARN, m);
}

static void enterRun() {
  for (uint8_t id = 1; id <= 4; id++) g_goalTicks[id] = g_presentTicks[id];  // start from where we are
  g_state = ST_RUN;
  fwlog(LOG_INFO, "RUN");
}

static void enterEstop() {
  servoSupply(false);
  g_state = ST_ESTOP;
  g_enableIntent = false;
  fwlog(LOG_ERR, "E-STOP: servo supply cut");
}

static void clearSafety() {
  if (g_state == ST_ESTOP) {
    servoSupply(true);
    delay(300);                       // let the servos boot
    busOpen();
    for (uint8_t id = 1; id <= 4; id++) {
      long p;
      if (sbRead16(id, REG_PRESENT_POS, &p)) {
        g_presentTicks[id] = (int16_t)p;
        g_holdTicks[id] = (int16_t)p;
        sbWritePos1(id, (int16_t)p, 200, 10);
        sbWrite8(id, REG_TORQUE_EN, 1);
      }
    }
  }
  g_holdReason = 0;
  if (g_state != ST_ESTOP) enterHold(0); else { g_state = ST_HOLD; g_holdReason = 4; }
  fwlog(LOG_INFO, "safety cleared -> SAFE_HOLD (send ENABLE to resume)");
}

// ======================================================================
//  COMMAND / CONFIG / SETPOINT handlers
// ======================================================================

static void onSetpoint(const uint8_t* p) {
  g_setXn = rdF32(p);
  g_setYn = rdF32(p + 4);
  g_setSeq  = rdU32(p + 8);
  g_lastSetpointMs = millis();
}

static void onConfig(const uint8_t* p) {
  cfg.current_limit_ma  = rdF32(p);
  cfg.tension_limit_g   = rdF32(p + 4);
  cfg.spike_debounce_ms = rdU16(p + 8);
  cfg.comms_timeout_ms  = rdU16(p + 10);
  cfg.motor_hz          = rdU16(p + 12);
  cfg.telem_hz          = rdU16(p + 14);
  cfg.maxpull[0][0]     = rdU16(p + 16);   // x +
  cfg.maxpull[0][1]     = rdU16(p + 18);   // x -
  cfg.maxpull[1][0]     = rdU16(p + 20);   // z +
  cfg.maxpull[1][1]     = rdU16(p + 22);   // z -
  cfg.maxpull_headroom  = rdF32(p + 24);
  if (cfg.motor_hz < 20)  cfg.motor_hz = 20;
  if (cfg.motor_hz > 500) cfg.motor_hz = 500;
  if (cfg.telem_hz < 1)   cfg.telem_hz = 1;
  if (cfg.telem_hz > 100) cfg.telem_hz = 100;
  if (cfg.maxpull_headroom < 0.1f) cfg.maxpull_headroom = 0.1f;
  if (cfg.maxpull_headroom > 1.0f) cfg.maxpull_headroom = 1.0f;
  for (uint8_t a = 0; a < 2; a++)
    for (uint8_t d = 0; d < 2; d++)
      if (cfg.maxpull[a][d] < 1 || cfg.maxpull[a][d] > 4000) cfg.maxpull[a][d] = MAXPULL_DEFAULT[a][d];
  char m[110];
  snprintf(m, sizeof(m),
           "CONFIG: I<%.0fmA T<%.0fg deb%u comms%u motor%uHz telem%uHz MAXPULL x+%d/-%d z+%d/-%d",
           cfg.current_limit_ma, cfg.tension_limit_g, cfg.spike_debounce_ms,
           cfg.comms_timeout_ms, cfg.motor_hz, cfg.telem_hz,
           appliedMaxPull(0, true), appliedMaxPull(0, false),
           appliedMaxPull(1, true), appliedMaxPull(1, false));
  fwlog(LOG_INFO, m);
}

static void onCommand(uint8_t c) {
  switch (c) {
    case CMD_ENABLE:
      g_enableIntent = true;
      if (g_state == ST_ESTOP) { fwlog(LOG_WARN, "ENABLE ignored: in E-STOP, send CLEAR_SAFETY"); break; }
      if (!g_allServos) { fwlog(LOG_ERR, "ENABLE refused: not all four servos on the bus"); break; }
      if (g_state == ST_HOLD) enterRun();
      break;
    case CMD_DISABLE:
      g_enableIntent = false;
      if (g_state != ST_ESTOP) enterHold(4);
      break;
    case CMD_CLEAR_SAFETY:
      clearSafety();
      break;
    case CMD_ESTOP:
      enterEstop();
      break;
    default:
      fwlog(LOG_WARN, "unknown command");
  }
}

// ======================================================================
//  FRAME RX
// ======================================================================

static uint8_t rxSeg[128];
static size_t  rxSegLen = 0;

static void handleSegment(const uint8_t* seg, size_t n) {
  uint8_t body[96];
  int bl = cobsDecode(seg, n, body, sizeof(body));
  if (bl < 3) return;
  uint16_t got  = (uint16_t)body[bl - 2] | ((uint16_t)body[bl - 1] << 8);
  if (got != crc16(body, bl - 2)) return;          // corrupt -> drop
  uint8_t type = body[0];
  const uint8_t* p = body + 1;
  size_t plen = bl - 3;
  switch (type) {
    case T_SETPOINT: if (plen >= 12) onSetpoint(p); break;
    case T_CONFIG:   if (plen >= 28) onConfig(p);   break;
    case T_COMMAND:  if (plen >= 1)  onCommand(p[0]); break;
    case T_PING:     if (plen >= 4)  sendFrame(T_PONG, p, 4); break;
    default: break;
  }
}

static void pumpSerial() {
  while (Serial.available()) {
    uint8_t b = (uint8_t)Serial.read();
    if (b == 0x00) {
      if (rxSegLen > 0) handleSegment(rxSeg, rxSegLen);
      rxSegLen = 0;
    } else if (rxSegLen < sizeof(rxSeg)) {
      rxSeg[rxSegLen++] = b;
    } else {
      rxSegLen = 0;   // overflow -> resync on next delimiter
    }
  }
}

// ======================================================================
//  MOTOR + SAFETY + TELEMETRY
// ======================================================================

static int16_t clampGoal(int16_t g) {
  int16_t lo = ZERO_TICK - MAX_ABS_TICKS_FROM_ZERO;
  int16_t hi = ZERO_TICK + MAX_ABS_TICKS_FROM_ZERO;
  if (g < lo) g = lo;
  if (g > hi) g = hi;
  if (g < SEAM) g = SEAM;
  if (g > 4095 - SEAM) g = 4095 - SEAM;
  return g;
}

static void desiredGoalsFromSetpoint(int16_t out[5]) {
  for (uint8_t a = 0; a < 2; a++) {
    float n = (a == 0) ? g_setXn : g_setYn;
    if (AXIS_INVERT[a]) n = -n;
    if (n >  1.0f) n =  1.0f;
    if (n < -1.0f) n = -1.0f;
    uint8_t p = AXIS_SERVO[a][0];
    uint8_t q = AXIS_SERVO[a][1];
    // asymmetric: the tick limit depends on which way we're bending
    int16_t lim = appliedMaxPull(a, n >= 0.0f);
    int16_t d = (int16_t)lroundf((float)PULL_SIGN[p] * n * (float)lim);
    out[p] = clampGoal(ZERO_TICK + d);
    out[q] = clampGoal(ZERO_TICK - d);
  }
}

static void motorTick() {
  int16_t want[5];
  if (g_state == ST_RUN) {
    desiredGoalsFromSetpoint(want);
  } else {
    for (uint8_t id = 1; id <= 4; id++) want[id] = g_holdTicks[id];
  }
  // slew-limit toward `want`
  for (uint8_t id = 1; id <= 4; id++) {
    int16_t delta = want[id] - g_goalTicks[id];
    if (delta >  MAX_TICKS_PER_CYCLE) delta =  MAX_TICKS_PER_CYCLE;
    if (delta < -MAX_TICKS_PER_CYCLE) delta = -MAX_TICKS_PER_CYCLE;
    g_goalTicks[id] += delta;
  }
  if (g_state != ST_ESTOP)
    sbSyncWritePos4(g_goalTicks, SERVO_SPEED, SERVO_ACC);
}

// poll one servo's present pos / current / load per call (round-robin)
static uint8_t  g_diagId = 1;
static uint16_t g_diagFails = 0;                 // consecutive failed transactions
static const uint16_t DIAG_FAIL_LIMIT = 12;      // ~ (12/3) servo cycles of silence -> SAFE_HOLD

static void diagPoll() {
  uint8_t id = g_diagId;
  long v;
  bool okPos = sbRead16(id, REG_PRESENT_POS, &v);
  if (okPos) {
    g_presentTicks[id] = (int16_t)v;
    if (sbRead16(id, REG_CURRENT, &v))    g_currentMa[id] = (int16_t)lroundf(v * CURRENT_LSB_MA);
    if (sbRead16(id, REG_PRESENT_LD, &v)) g_loadRaw[id] = (int16_t)((v & 0x8000) ? -(long)(v & 0x7FFF) : v);
  }

  if (okPos) {
    g_diagFails = 0;
  } else if (g_state == ST_RUN || g_state == ST_HOLD) {
    if (++g_diagFails >= DIAG_FAIL_LIMIT && g_state != ST_ESTOP) {
      fwlog(LOG_ERR, "servo bus lost -- SAFE_HOLD");
      enterHold(2);
      g_diagFails = 0;
    }
  }

  // current safety (fast path)
  bool over = fabsf((float)g_currentMa[id]) > cfg.current_limit_ma;
  if (over) {
    if (g_curOverSince[id] == 0) g_curOverSince[id] = millis();
    if (millis() - g_curOverSince[id] >= cfg.spike_debounce_ms &&
        (g_state == ST_RUN || g_state == ST_HOLD)) {
      if (g_state != ST_ESTOP && g_holdReason != 2) enterHold(2);
    }
  } else {
    g_curOverSince[id] = 0;
  }

  g_diagId = (g_diagId % 4) + 1;
}

static void tensionPoll() {
  if (!hxAllReady()) return;
  hxReadNow();
  for (uint8_t s = 0; s < 4; s++) {
    bool over = fabsf(hxGrams[s]) > cfg.tension_limit_g;
    if (over) {
      if (g_tenOverSince[s + 1] == 0) g_tenOverSince[s + 1] = millis();
      if (millis() - g_tenOverSince[s + 1] >= cfg.spike_debounce_ms &&
          g_state != ST_ESTOP && g_holdReason != 3) {
        enterHold(3);
      }
    } else {
      g_tenOverSince[s + 1] = 0;
    }
  }
}

static void tipPoll() {
  if (TIP_CONTACT_FORCE_ZERO) { g_tipContact = false; return; }
  if (TIP_CONTACT_PIN >= 0)
    g_tipContact = analogRead(TIP_CONTACT_PIN) > TIP_CONTACT_ADC_THRESHOLD;
}

static void sendTelemetry() {
  uint8_t p[41];
  wrU32(p + 0, g_setSeq);
  uint8_t flags = 0;
  if (g_tipContact) flags |= FLAG_TIP_CONTACT;
  flags |= (safetyStateForTelem() & FLAG_SAFETY_MASK) << FLAG_SAFETY_SHIFT;
  if (g_state == ST_RUN) flags |= FLAG_ENABLED;
  p[4] = flags;
  for (uint8_t s = 0; s < 4; s++) wrF32(p + 5 + s * 4, hxGrams[s]);
  for (uint8_t s = 0; s < 4; s++) wrI16(p + 21 + s * 2, g_presentTicks[s + 1]);
  for (uint8_t s = 0; s < 4; s++) wrI16(p + 29 + s * 2, g_currentMa[s + 1]);
  wrU32(p + 37, millis());
  sendFrame(T_TELEMETRY, p, sizeof(p));
}

static void sendHello() {
  uint8_t p[18];                                  // <BB4H4h>
  p[0] = PROTO_VERSION;
  p[1] = 4;
  wrU16(p + 2, (uint16_t)appliedMaxPull(0, true));   // x +
  wrU16(p + 4, (uint16_t)appliedMaxPull(0, false));  // x -
  wrU16(p + 6, (uint16_t)appliedMaxPull(1, true));   // z +
  wrU16(p + 8, (uint16_t)appliedMaxPull(1, false));  // z -
  for (uint8_t s = 0; s < 4; s++) wrI16(p + 10 + s * 2, ZERO_TICK);
  sendFrame(T_HELLO, p, sizeof(p));
}

static void commsWatchdog(uint32_t now) {
  if (g_state == ST_RUN && (now - g_lastSetpointMs) > cfg.comms_timeout_ms) {
    enterHold(1);
    return;
  }
  // auto-recover from a comms-only HOLD once setpoints resume and the operator
  // still wants RUN. A current/tension HOLD needs an explicit CLEAR_SAFETY.
  if (g_state == ST_HOLD && g_holdReason == 1 && g_enableIntent &&
      (now - g_lastSetpointMs) < cfg.comms_timeout_ms && g_allServos) {
    enterRun();
  }
}

// ======================================================================
//  setup / loop
// ======================================================================

void setup() {
  Serial.setRxBufferSize(512);
  Serial.begin(921600);
  delay(150);

  if (ESTOP_MOSFET_PIN >= 0) { pinMode(ESTOP_MOSFET_PIN, OUTPUT); servoSupply(true); }
  if (HX_RATE_PIN >= 0) { pinMode(HX_RATE_PIN, OUTPUT); digitalWrite(HX_RATE_PIN, HIGH); }

  pinMode(HX_SCK, OUTPUT);
  digitalWrite(HX_SCK, LOW);
  for (uint8_t s = 0; s < 4; s++) pinMode(HX_DT_FOR_SERVO[s], INPUT);
  if (TIP_CONTACT_PIN >= 0 && !TIP_CONTACT_FORCE_ZERO) pinMode(TIP_CONTACT_PIN, INPUT);

  delay(400);                 // servo power-up
  busOpen();
  for (uint8_t i = 0; i < 3; i++) { sbWrite8(BROADCAST_ID, REG_TORQUE_EN, 0); delay(10); }

  uint8_t found = 0;
  for (uint8_t id = 1; id <= 4; id++) {
    if (!sbPing(id)) continue;
    found++;
    long p;
    if (sbRead16(id, REG_PRESENT_POS, &p)) {
      g_presentTicks[id] = (int16_t)p;
      g_goalTicks[id] = (int16_t)p;
      g_holdTicks[id] = (int16_t)p;
      sbWritePos1(id, (int16_t)p, 200, 10);   // goal = present -> no jump
      sbWrite8(id, REG_TORQUE_EN, 1);          // hold here
    }
  }
  g_allServos = (found == 4);

  g_state = ST_HOLD;
  g_holdReason = 4;
  g_lastSetpointMs = millis();

  sendHello();
  fwlog(LOG_INFO, g_allServos ? "scope_control_s3 ready, SAFE_HOLD"
                              : "scope_control_s3 ready, SAFE_HOLD -- NOT all 4 servos found");
  {
    char m[100];
    snprintf(m, sizeof(m), "MAXPULL ticks (headroom %.2f): x+%d x-%d z+%d z-%d",
             cfg.maxpull_headroom,
             appliedMaxPull(0, true), appliedMaxPull(0, false),
             appliedMaxPull(1, true), appliedMaxPull(1, false));
    fwlog(LOG_INFO, m);
  }
  if (ESTOP_MOSFET_PIN < 0)
    fwlog(LOG_WARN, "no ESTOP_MOSFET_PIN configured -- e-stop cannot cut the supply");
}

void loop() {
  uint32_t now = millis();
  static uint32_t tMotor = 0, tDiag = 0, tTelem = 0;

  pumpSerial();

  const uint32_t motorPeriod = 1000UL / cfg.motor_hz;
  const uint32_t telemPeriod = 1000UL / cfg.telem_hz;

  if (now - tMotor >= motorPeriod) { tMotor = now; motorTick(); }
  if (now - tDiag  >= motorPeriod) { tDiag  = now; diagPoll(); }   // one servo/cycle -> each @ motor_hz/4
  tensionPoll();                                                    // effective rate = HX711 SPS
  tipPoll();
  commsWatchdog(now);
  if (now - tTelem >= telemPeriod) { tTelem = now; sendTelemetry(); }
}
