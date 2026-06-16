"""
MIDI File Parser & Event Processing Engine
===========================================
Parses Standard MIDI Files (Format 0/1), decodes VLQ time deltas,
handles running status, merges multi-track events by absolute time,
converts tick-based timing to wall-clock seconds via tempo meta-events,
and pairs Note-On / Note-Off events to compute note durations.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import BinaryIO, List, Optional, Tuple


class EventType(IntEnum):
    NOTE_OFF = 0x80
    NOTE_ON = 0x90
    POLY_PRESSURE = 0xA0
    CONTROL_CHANGE = 0xB0
    PROGRAM_CHANGE = 0xC0
    CHANNEL_PRESSURE = 0xD0
    PITCH_BEND = 0xE0
    META = 0xFF
    SYSEX = 0xF0
    ESCAPE = 0xF7


class MetaType(IntEnum):
    SEQUENCE_NUMBER = 0x00
    TEXT = 0x01
    COPYRIGHT = 0x02
    TRACK_NAME = 0x03
    INSTRUMENT_NAME = 0x04
    LYRIC = 0x05
    MARKER = 0x06
    CUE_POINT = 0x07
    CHANNEL_PREFIX = 0x20
    MIDI_PORT = 0x21
    END_OF_TRACK = 0x2F
    TEMPO = 0x51
    SMPTE_OFFSET = 0x54
    TIME_SIGNATURE = 0x58
    KEY_SIGNATURE = 0x59
    SEQUENCER_SPECIFIC = 0x7F


@dataclass
class MidiEvent:
    tick: int
    track: int
    delta: int
    event_type: EventType
    channel: int
    data: bytes = b""
    meta_type: Optional[MetaType] = None
    text: str = ""

    @property
    def is_note_on(self) -> bool:
        return self.event_type == EventType.NOTE_ON and self.data[1] != 0 if len(self.data) >= 2 else False

    @property
    def is_note_off(self) -> bool:
        if self.event_type == EventType.NOTE_OFF:
            return True
        if self.event_type == EventType.NOTE_ON and len(self.data) >= 2 and self.data[1] == 0:
            return True
        return False


@dataclass
class NoteEvent:
    note: int
    channel: int
    velocity: int
    start_tick: int
    end_tick: int
    start_seconds: float
    end_seconds: float
    track: int

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class TempoEvent:
    tick: int
    seconds: float
    microseconds_per_beat: int


@dataclass
class TimeSignatureEvent:
    tick: int
    numerator: int
    denominator: int
    metronome: int
    thirty_seconds: int


@dataclass
class MidiFile:
    format_type: int
    ticks_per_beat: int
    tracks: List[List[MidiEvent]] = field(default_factory=list)

    def all_events_sorted(self) -> List[MidiEvent]:
        merged: List[MidiEvent] = []
        for track_idx, track in enumerate(self.tracks):
            for ev in track:
                merged.append(ev)
        merged.sort(key=lambda e: (e.tick, e.track))
        return merged

    def tempo_map(self) -> List[TempoEvent]:
        events = self.all_events_sorted()
        tmap: List[TempoEvent] = []
        for ev in events:
            if ev.event_type == EventType.META and ev.meta_type == MetaType.TEMPO:
                us_per_beat = struct.unpack(">I", b"\x00" + ev.data)[0]
                tmap.append(TempoEvent(tick=ev.tick, seconds=0.0, microseconds_per_beat=us_per_beat))
        if not tmap or tmap[0].tick != 0:
            tmap.insert(0, TempoEvent(tick=0, seconds=0.0, microseconds_per_beat=500000))
        current_seconds = 0.0
        current_tick = 0
        current_ustempo = tmap[0].microseconds_per_beat
        tmap[0].seconds = 0.0
        for i in range(1, len(tmap)):
            delta_ticks = tmap[i].tick - current_tick
            delta_seconds = delta_ticks * (current_ustempo / 1_000_000.0) / self.ticks_per_beat
            current_seconds += delta_seconds
            current_tick = tmap[i].tick
            current_ustempo = tmap[i].microseconds_per_beat
            tmap[i].seconds = current_seconds
        return tmap

    def tick_to_seconds(self, tick: int, tmap: Optional[List[TempoEvent]] = None) -> float:
        if tmap is None:
            tmap = self.tempo_map()
        if not tmap:
            return 0.0
        idx = 0
        for i in range(len(tmap) - 1, -1, -1):
            if tmap[i].tick <= tick:
                idx = i
                break
        delta = tick - tmap[idx].tick
        return tmap[idx].seconds + delta * (tmap[idx].microseconds_per_beat / 1_000_000.0) / self.ticks_per_beat

    def assign_seconds(self) -> List[MidiEvent]:
        tmap = self.tempo_map()
        events = self.all_events_sorted()
        for ev in events:
            ev._seconds = self.tick_to_seconds(ev.tick, tmap)
        return events

    def extract_notes(self) -> List[NoteEvent]:
        tmap = self.tempo_map()
        events = self.all_events_sorted()
        pending: dict[Tuple[int, int, int], MidiEvent] = {}
        notes: List[NoteEvent] = []
        for ev in events:
            if ev.is_note_on:
                key = (ev.track, ev.channel, ev.data[0])
                if key in pending:
                    old = pending.pop(key)
                    start_s = self.tick_to_seconds(old.tick, tmap)
                    end_s = self.tick_to_seconds(ev.tick, tmap)
                    notes.append(NoteEvent(
                        note=old.data[0], channel=old.channel,
                        velocity=old.data[1], start_tick=old.tick,
                        end_tick=ev.tick, start_seconds=start_s,
                        end_seconds=end_s, track=old.track,
                    ))
                pending[key] = ev
            elif ev.is_note_off:
                key = (ev.track, ev.channel, ev.data[0])
                if key in pending:
                    old = pending.pop(key)
                    start_s = self.tick_to_seconds(old.tick, tmap)
                    end_s = self.tick_to_seconds(ev.tick, tmap)
                    notes.append(NoteEvent(
                        note=old.data[0], channel=old.channel,
                        velocity=old.data[1], start_tick=old.tick,
                        end_tick=ev.tick, start_seconds=start_s,
                        end_seconds=end_s, track=old.track,
                    ))
        return notes


class MidiReader:
    def __init__(self, stream: BinaryIO):
        self._s = stream
        self._pos = 0

    def _read(self, n: int) -> bytes:
        data = self._s.read(n)
        if len(data) < n:
            raise EOFError(f"Expected {n} bytes at offset {self._pos}, got {len(data)}")
        self._pos += len(data)
        return data

    def _read_uint8(self) -> int:
        return self._read(1)[0]

    def _read_uint16(self) -> int:
        return struct.unpack(">H", self._read(2))[0]

    def _read_uint32(self) -> int:
        return struct.unpack(">I", self._read(4))[0]

    def _read_vlq(self) -> int:
        """
        VLQ (Variable-Length Quantity) decoding.

        Each byte uses bit 7 (0x80) as a continuation flag:
          - If bit 7 is SET (1), more bytes follow.
          - If bit 7 is CLEAR (0), this is the last byte.
        The remaining 7 bits of each byte carry the payload.

        Example: the value 0x3FFF encodes as [0xFF, 0x7F]
          Byte 0: 0xFF = 1_1111111 → continue, payload = 0x7F
          Byte 1: 0x7F = 0_1111111 → last,     payload = 0x7F
          Value = 0x7F << 7 | 0x7F = 0x3FFF
        """
        value = 0
        while True:
            byte = self._read_uint8()
            value = (value << 7) | (byte & 0x7F)
            if not (byte & 0x80):
                break
        return value

    def _read_chunk_header(self) -> Tuple[str, int]:
        tag = self._read(4).decode("ascii")
        length = self._read_uint32()
        return tag, length

    def parse(self) -> MidiFile:
        tag, _ = self._read_chunk_header()
        if tag != "MThd":
            raise ValueError(f"Invalid MIDI header: expected 'MThd', got '{tag}'")
        fmt = self._read_uint16()
        num_tracks = self._read_uint16()
        division = self._read_uint16()
        if division & 0x8000:
            raise ValueError("SMPTE time division not supported")
        ticks_per_beat = division & 0x7FFF
        midi = MidiFile(format_type=fmt, ticks_per_beat=ticks_per_beat)
        for i in range(num_tracks):
            midi.tracks.append(self._parse_track(i))
        return midi

    def _parse_track(self, track_idx: int) -> List[MidiEvent]:
        tag, length = self._read_chunk_header()
        if tag != "MTrk":
            raise ValueError(f"Invalid track header: expected 'MTrk', got '{tag}'")
        end_pos = self._pos + length
        events: List[MidiEvent] = []
        abs_tick = 0
        running_status = 0

        while self._pos < end_pos:
            delta = self._read_vlq()
            abs_tick += delta

            peek = self._read_uint8()

            if peek == 0xFF:
                meta_type = self._read_uint8()
                meta_len = self._read_vlq()
                meta_data = self._read(meta_len) if meta_len else b""
                text_str = ""
                if meta_type in (MetaType.TEXT, MetaType.COPYRIGHT, MetaType.TRACK_NAME,
                                 MetaType.INSTRUMENT_NAME, MetaType.LYRIC, MetaType.MARKER,
                                 MetaType.CUE_POINT):
                    try:
                        text_str = meta_data.decode("utf-8")
                    except UnicodeDecodeError:
                        text_str = meta_data.decode("latin-1")
                events.append(MidiEvent(
                    tick=abs_tick, track=track_idx, delta=delta,
                    event_type=EventType.META, channel=0,
                    data=meta_data, meta_type=MetaType(meta_type), text=text_str,
                ))
                if meta_type == MetaType.END_OF_TRACK:
                    break

            elif peek == 0xF0:
                sysex_len = self._read_vlq()
                sysex_data = self._read(sysex_len)
                if sysex_data and sysex_data[-1] == 0xF7:
                    sysex_data = sysex_data[:-1]
                events.append(MidiEvent(
                    tick=abs_tick, track=track_idx, delta=delta,
                    event_type=EventType.SYSEX, channel=0,
                    data=sysex_data,
                ))
                running_status = 0

            elif peek == 0xF7:
                esc_len = self._read_vlq()
                esc_data = self._read(esc_len)
                events.append(MidiEvent(
                    tick=abs_tick, track=track_idx, delta=delta,
                    event_type=EventType.ESCAPE, channel=0,
                    data=esc_data,
                ))
                running_status = 0

            else:
                status = peek
                if status < 0x80:
                    status = running_status
                    self._pos -= 1
                    self._s.seek(-1, 1)
                else:
                    running_status = status

                if status < 0x80:
                    raise ValueError(f"Invalid status byte 0x{status:02X} at offset {self._pos}")

                hi = status & 0xF0
                ch = status & 0x0F

                if hi == EventType.PROGRAM_CHANGE or hi == EventType.CHANNEL_PRESSURE:
                    d = bytes([self._read_uint8()])
                else:
                    d = bytes([self._read_uint8(), self._read_uint8()])

                events.append(MidiEvent(
                    tick=abs_tick, track=track_idx, delta=delta,
                    event_type=EventType(hi), channel=ch, data=d,
                ))

        if self._pos < end_pos:
            self._s.seek(end_pos)
            self._pos = end_pos

        return events


def parse_midi(path: str) -> MidiFile:
    with open(path, "rb") as f:
        return MidiReader(f).parse()


def format_time_signature(ev: MidiEvent) -> str:
    if ev.meta_type != MetaType.TIME_SIGNATURE or len(ev.data) < 4:
        return ""
    num = ev.data[0]
    den = 2 ** ev.data[1]
    metro = ev.data[2]
    ts32 = ev.data[3]
    return f"{num}/{den}  metro={metro}  32nd={ts32}"


def format_tempo(ev: MidiEvent) -> str:
    if ev.meta_type != MetaType.TEMPO or len(ev.data) < 3:
        return ""
    us = struct.unpack(">I", b"\x00" + ev.data)[0]
    bpm = 60_000_000.0 / us
    return f"{bpm:.2f} BPM  ({us} μs/beat)"


def print_summary(midi: MidiFile) -> None:
    print(f"Format: {midi.format_type}  |  Tracks: {len(midi.tracks)}  |  Ticks/Beat: {midi.ticks_per_beat}")
    print("=" * 70)
    for idx, track in enumerate(midi.tracks):
        print(f"\n--- Track {idx} ({len(track)} events) ---")
        for ev in track[:20]:
            desc = _describe(ev)
            print(f"  tick={ev.tick:>6d}  δ={ev.delta:>4d}  {desc}")
        if len(track) > 20:
            print(f"  ... ({len(track) - 20} more events)")
    print("\n" + "=" * 70)
    merged = midi.all_events_sorted()
    print(f"Merged event stream: {len(merged)} events (sorted by absolute tick)")
    tmap = midi.tempo_map()
    print(f"Tempo map: {len(tmap)} tempo changes")
    for te in tmap:
        bpm = 60_000_000.0 / te.microseconds_per_beat
        print(f"  tick={te.tick:>6d}  {te.seconds:>8.3f}s  {bpm:.2f} BPM")
    notes = midi.extract_notes()
    print(f"\nNote events: {len(notes)}")
    for n in notes[:15]:
        print(f"  ch={n.channel:>2d}  note={n.note:>3d}  vel={n.velocity:>3d}  "
              f"ticks={n.start_tick}-{n.end_tick} ({n.duration_ticks})  "
              f"sec={n.start_seconds:.3f}-{n.end_seconds:.3f} ({n.duration_seconds:.3f}s)")
    if len(notes) > 15:
        print(f"  ... ({len(notes) - 15} more notes)")


def _describe(ev: MidiEvent) -> str:
    if ev.event_type == EventType.META:
        mt = MetaType(ev.meta_type) if ev.meta_type is not None else None
        if mt == MetaType.TEMPO:
            return f"Meta:Tempo {format_tempo(ev)}"
        if mt == MetaType.TIME_SIGNATURE:
            return f"Meta:TimeSig {format_time_signature(ev)}"
        if mt == MetaType.TRACK_NAME:
            return f"Meta:TrackName \"{ev.text}\""
        if mt == MetaType.END_OF_TRACK:
            return "Meta:EndOfTrack"
        return f"Meta:0x{ev.meta_type:02X} ({len(ev.data)}b)"
    if ev.event_type == EventType.SYSEX:
        return f"SysEx ({len(ev.data)}b)"
    if ev.event_type == EventType.ESCAPE:
        return f"Escape ({len(ev.data)}b)"
    ch = ev.channel
    if ev.event_type == EventType.NOTE_ON:
        vel = ev.data[1] if len(ev.data) > 1 else 0
        note = ev.data[0] if ev.data else 0
        if vel == 0:
            return f"Ch{ch:>2d} NoteOff  note={note:>3d}"
        return f"Ch{ch:>2d} NoteOn   note={note:>3d} vel={vel:>3d}"
    if ev.event_type == EventType.NOTE_OFF:
        note = ev.data[0] if ev.data else 0
        vel = ev.data[1] if len(ev.data) > 1 else 0
        return f"Ch{ch:>2d} NoteOff  note={note:>3d} vel={vel:>3d}"
    if ev.event_type == EventType.CONTROL_CHANGE:
        cc = ev.data[0] if ev.data else 0
        val = ev.data[1] if len(ev.data) > 1 else 0
        return f"Ch{ch:>2d} CC#{cc:>3d}={val:>3d}"
    if ev.event_type == EventType.PROGRAM_CHANGE:
        pg = ev.data[0] if ev.data else 0
        return f"Ch{ch:>2d} Program={pg:>3d}"
    if ev.event_type == EventType.PITCH_BEND:
        raw = (ev.data[1] << 7 | ev.data[0]) if len(ev.data) >= 2 else 0
        return f"Ch{ch:>2d} PitchBend={raw}"
    if ev.event_type == EventType.POLY_PRESSURE:
        note = ev.data[0] if ev.data else 0
        pres = ev.data[1] if len(ev.data) > 1 else 0
        return f"Ch{ch:>2d} PolyPressure note={note:>3d} pressure={pres:>3d}"
    if ev.event_type == EventType.CHANNEL_PRESSURE:
        pres = ev.data[0] if ev.data else 0
        return f"Ch{ch:>2d} ChannelPressure={pres:>3d}"
    return f"Ch{ch:>2d} Event 0x{ev.event_type:02X}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python midi_engine.py <file.mid>")
        sys.exit(1)
    midi = parse_midi(sys.argv[1])
    print_summary(midi)
