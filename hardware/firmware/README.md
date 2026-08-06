# hardware/firmware/

Microcontroller code driving the tendon motors and feed rollers. **Empty as of 2026-08-06.**

## What this layer is responsible for

The policy emits three numbers in [-1, 1] at 25 Hz. Everything between those numbers and the
motors is firmware's problem:

| Policy output | Meaning | What firmware must do |
|---|---|---|
| `a[0]` | X-axis bend **setpoint** | drive the X cable pair to an absolute pull in mm — one cable pulls, the opposing one releases by the same amount |
| `a[1]` | Z-axis bend **setpoint** | same, on the other pair |
| `a[2]` | insertion **rate**, signed | accumulate into a monotonic depth target and drive the feed rollers |

`a[0]` and `a[1]` are **absolute positions, not rates.** `a[2]` *is* a rate. Getting this
backwards produces a scope that appears to work and steers nothing.

## Three things that will silently break the policy

1. **Cables pull only.** In sim the tendon force rail is `[-10, 0] N` — a cable can never push.
   Any implementation where a "negative pull" drives the cable outward is not the trained
   system.
2. **The PD command shaper is not firmware's to invent.** It sits *inside* the observation loop:
   its output is fed back as two of the six policy inputs. It must be reproduced exactly, with
   the same gains, at the same rate, on whichever side of the wire runs the policy — and its
   gains **do not rescale arithmetically with rate**. See
   [`../../docs/architecture.md`](../../docs/architecture.md).
3. **`MAX_PULL_X` and `MAX_PULL_Z` differ** (6.795 vs 7.361 mm in sim, because the axes span 12
   vs 13 joints), and both are **sim-derived numbers pending measurement on the real scope**.
   They normalise live policy inputs, so a wrong value is not a scaling nuisance — it feeds the
   network the wrong observation.

## What firmware must report back

Only one thing the policy actually consumes: a **binary tip-contact flag**. Everything else in
the state vector is software state reproduced on the policy side, not measured.

Encoder feedback is still needed for the motors' own closed loops, and for the pre-flight
`MAX_PULL` measurement — but tendon length is deliberately **not** a policy input. See
`architecture.md` § "What is deliberately not observed" for why that was removed rather than
kept.
