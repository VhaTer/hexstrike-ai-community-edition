# HTB "Debug" — Challenge Debrief (2026-06-02)

## Goal
Decode UART serial signal from satellite dish boot sequence captured in a Saleae Logic 2 .sal file. Find flag in format `HTB{...}`.

## File
`hw_debug.sal` — Saleae .sal (renamed ZIP containing):
- `meta.json`: capture config (25MHz, 2 digital channels, 37.5s circular buffer, no trigger)
- `digital-0.bin`: 88,610 bytes — TX channel
- `digital-1.bin`: 178,171 bytes — RX channel

## Format Discovered
- **File header** (75B): ASCII text `"100,0,1,0\0..."` (RX) or `"100,1,1,0\0..."` (TX)
- **Metadata blocks**: 20 bytes each (uint64 sample_number=25000000, uint64 byte_offset=1, int32 bit_state). But ALL byte_offsets=1 (suspicious). bit_state is 3 for idle blocks, larger values (476-780) for data blocks.
- **RLE between metadata**: varint-encoded sample counts. Idle pattern: `72 ff 7f 01` = [114, 16383, 1] + zeros (fill).
- **Block structure**: 79B per interval (20B metadata + variable RLE). Idle blocks have 59B RLE. Data blocks have 65-1272B RLE.
- **TX**: 1121 blocks, ALL idle (59B RLE each). ZERO real data.
- **RX**: 1121 blocks. ~890 idle, ~219 data blocks interspersed. ~12s active signal + ~25s idle.

## Decoding Attempted

### UART (standard)
- All standard baud rates 300-3,000,000
- Both inverted and non-inverted
- sigrok-cli with proper syntax: `-P uart:baudrate=9600 -A uart=rx-data`
- Pure Python decoders with bit-level sampling
- **Result**: garbled output at all rates. Best printable-ASCII score ~47% (e.g., `M.U5UUU......UUE5.CU...`)

### Non-standard baud rates
- Derived from common run lengths in signal
- Bit periods that divide observed run lengths
- **Result**: same — garbled

### RLE format variations (5 options)
- A: all values toggle + count (standard RLE)
- B: val=0 = fill (no time, no toggle)
- C: val=1 = toggle only
- D: alternating even/odd
- E: zeros ignored
- **Result**: none produce readable UART output

### Metadata reinterpretation
- 20B vs 24B metadata (confirmed 20B via Saleae forum)
- But byte_offset=1 for ALL blocks — contradicts "byte offset into RLE data"
- bit_state values: 3 (idle), 476/464/596/780 (data blocks)
- **Result**: bit_state doesn't encode flag directly

### Varints as direct ASCII
- 74.7% of varints from non-idle blocks fall in printable ASCII range (32-126)
- Filtered output shows REPEATING patterns: `AXAXC1AXC1AXAXAXC1EC0AY...`
- This appears in ALMOST ALL data blocks
- 'A'(65) and 'X'(88) dominate as framing/idle bytes
- Flag `HTB{` NOT found in this filtered stream

### Binary transforms
- XOR with all keys, bit-reversed, base64, delta-encoded
- Every-Nth byte, varint→mod128→ASCII
- **Result**: no flag found

## Key Observations

1. **"AXAXC1" pattern**: The ASCII-range varints form a VERY consistent pattern across ALL data blocks: `AXAXC1AXC1AXAXAXC1EC0AY...` with variations. This looks like PROTOCOL FRAMING, not random data.

2. **'A'(65) and 'X'(88) are dominant**: Every block has long runs of A(65)/X(88) alternating. These are likely idle/sync bytes in a serial protocol.

3. **Occasional deviations**: Some blocks deviate from the pattern with 'C'(67), '1'(49), '0'(48), 'E'(69), 'F'(70), 'c'(99), 'd'(100), 'J'(74), 'H'(72), '<'(60), '='(61), '\n'(10).

4. **Single character prefix**: Each block's data starts with a different ASCII character (h, C, P, G, [, ], M, E, ...). Could be a block counter or identifier.

5. **No 'HTB{' anywhere**: Not in raw bytes, metadata, varints, XOR, or any transformation.

## Open Questions

1. **Is this really UART?** With 2 channels, could be: UART (TX/RX), I2C (SCL/SDA), SPI (with shared MISO/MOSI, missing CLK), or a custom protocol.

2. **What if RLE values directly encode bytes?** 74.7% of varints being printable ASCII is statistically suspicious. Maybe the non-idle blocks contain PRE-DECODED UART data (Saleae storing decoded output, not raw samples).

3. **Where's the actual flag?** Not in the signal data. Maybe in the metadata bit_state values, in the timing between bytes, or in a hidden part of the file.

4. **TX variance from RX?** What does XOR(TX, RX) reveal? Or the DIFFERENCE in timing between the two channels?

5. **Format misinterpretation**: Is the Metadata format really 20 bytes? What if it's 8+8+4+4=24 with an extra int32? Or 8+8+8+4=28?
