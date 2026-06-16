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
import json
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import BinaryIO, List, Optional, Tuple, Dict, Any


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
    meta_type_code: Optional[int] = None
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
class OverlapWarning:
    tick: int
    track: int
    channel: int
    note: int
    new_velocity: int
    old_start_tick: int
    old_velocity: int
    action: str


@dataclass
class NoteExtractionResult:
    notes: List[NoteEvent]
    warnings: List[OverlapWarning] = field(default_factory=list)
    unresolved_note_ons: int = 0

    def by_track_and_channel(self) -> Dict[int, Dict[int, List[NoteEvent]]]:
        grouped: Dict[int, Dict[int, List[NoteEvent]]] = defaultdict(lambda: defaultdict(list))
        for n in self.notes:
            grouped[n.track][n.channel].append(n)
        for t in grouped:
            for ch in grouped[t]:
                grouped[t][ch].sort(key=lambda n: n.start_tick)
        return dict(grouped)


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
        return self.extract_notes_extended().notes

    def extract_notes_extended(self) -> NoteExtractionResult:
        tmap = self.tempo_map()
        events = self.all_events_sorted()
        pending: dict[Tuple[int, int, int], MidiEvent] = {}
        notes: List[NoteEvent] = []
        warnings: List[OverlapWarning] = []

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
                    warnings.append(OverlapWarning(
                        tick=ev.tick, track=ev.track, channel=ev.channel,
                        note=ev.data[0], new_velocity=ev.data[1],
                        old_start_tick=old.tick, old_velocity=old.data[1],
                        action="prematurely closed previous note-on before new one",
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
                else:
                    warnings.append(OverlapWarning(
                        tick=ev.tick, track=ev.track, channel=ev.channel,
                        note=ev.data[0], new_velocity=ev.data[1] if len(ev.data) > 1 else 0,
                        old_start_tick=-1, old_velocity=-1,
                        action="note-off with no matching note-on (orphaned)",
                    ))

        unresolved = len(pending)
        for (track, ch, note), ev in pending.items():
            warnings.append(OverlapWarning(
                tick=ev.tick, track=track, channel=ch,
                note=note, new_velocity=ev.data[1] if len(ev.data) > 1 else 0,
                old_start_tick=-1, old_velocity=-1,
                action="unresolved note-on (never received note-off)",
            ))

        return NoteExtractionResult(notes=notes, warnings=warnings, unresolved_note_ons=unresolved)

    def time_signature_changes(self) -> List[TimeSignatureEvent]:
        events = self.all_events_sorted()
        sigs: List[TimeSignatureEvent] = []
        for ev in events:
            if ev.event_type == EventType.META and ev.meta_type == MetaType.TIME_SIGNATURE and len(ev.data) >= 4:
                num = ev.data[0]
                den = 2 ** ev.data[1]
                metro = ev.data[2]
                ts32 = ev.data[3]
                sigs.append(TimeSignatureEvent(
                    tick=ev.tick, numerator=num, denominator=den,
                    metronome=metro, thirty_seconds=ts32,
                ))
        return sigs

    def analyze(self) -> Dict[str, Any]:
        tmap = self.tempo_map()
        events = self.all_events_sorted()
        notes_result = self.extract_notes_extended()
        notes = notes_result.notes
        sig_changes = self.time_signature_changes()

        max_tick = 0
        for ev in events:
            if ev.tick > max_tick:
                max_tick = ev.tick
        total_seconds = self.tick_to_seconds(max_tick, tmap) if max_tick > 0 else 0.0

        track_names: Dict[int, str] = {}
        for ev in events:
            if ev.event_type == EventType.META and ev.meta_type == MetaType.TRACK_NAME and ev.track not in track_names:
                track_names[ev.track] = ev.text

        track_note_counts: Dict[int, int] = defaultdict(int)
        track_pitch_ranges: Dict[int, Dict[str, int]] = {}
        for n in notes:
            track_note_counts[n.track] += 1
            if n.track not in track_pitch_ranges:
                track_pitch_ranges[n.track] = {"min": n.note, "max": n.note}
            else:
                pr = track_pitch_ranges[n.track]
                pr["min"] = min(pr["min"], n.note)
                pr["max"] = max(pr["max"], n.note)

        all_notes = [n.note for n in notes]
        overall_range = {"min": min(all_notes), "max": max(all_notes)} if all_notes else {"min": 0, "max": 0}

        tempo_changes: List[Dict[str, Any]] = []
        for te in tmap:
            tempo_changes.append({
                "tick": te.tick,
                "seconds": round(te.seconds, 4),
                "microseconds_per_beat": te.microseconds_per_beat,
                "bpm": round(60_000_000.0 / te.microseconds_per_beat, 2),
            })

        time_sig_changes: List[Dict[str, Any]] = []
        for ts in sig_changes:
            time_sig_changes.append({
                "tick": ts.tick,
                "numerator": ts.numerator,
                "denominator": ts.denominator,
                "signature": f"{ts.numerator}/{ts.denominator}",
                "seconds": round(self.tick_to_seconds(ts.tick, tmap), 4),
            })

        tracks_info: Dict[int, Dict[str, Any]] = {}
        for idx, track in enumerate(self.tracks):
            tracks_info[idx] = {
                "name": track_names.get(idx, f"Track {idx}"),
                "total_events": len(track),
                "note_count": track_note_counts.get(idx, 0),
                "pitch_range": track_pitch_ranges.get(idx, {"min": 0, "max": 0}),
                "note_range_semitones": (
                    track_pitch_ranges[idx]["max"] - track_pitch_ranges[idx]["min"]
                    if idx in track_pitch_ranges else 0
                ),
            }

        return {
            "format_type": self.format_type,
            "tracks_count": len(self.tracks),
            "ticks_per_beat": self.ticks_per_beat,
            "total_ticks": max_tick,
            "total_seconds": round(total_seconds, 4),
            "total_minutes": round(total_seconds / 60.0, 4),
            "total_notes": len(notes),
            "overall_pitch_range": overall_range,
            "overall_range_semitones": overall_range["max"] - overall_range["min"],
            "tempo_changes": tempo_changes,
            "time_signature_changes": time_sig_changes,
            "tracks": tracks_info,
            "warnings_count": len(notes_result.warnings),
            "overlaps_count": sum(1 for w in notes_result.warnings if "closed previous" in w.action),
            "orphaned_note_offs": sum(1 for w in notes_result.warnings if "orphaned" in w.action),
            "unresolved_note_ons": notes_result.unresolved_note_ons,
        }


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
                try:
                    meta_enum = MetaType(meta_type)
                except ValueError:
                    meta_enum = None
                if meta_enum in (MetaType.TEXT, MetaType.COPYRIGHT, MetaType.TRACK_NAME,
                                 MetaType.INSTRUMENT_NAME, MetaType.LYRIC, MetaType.MARKER,
                                 MetaType.CUE_POINT):
                    try:
                        text_str = meta_data.decode("utf-8")
                    except UnicodeDecodeError:
                        text_str = meta_data.decode("latin-1")
                events.append(MidiEvent(
                    tick=abs_tick, track=track_idx, delta=delta,
                    event_type=EventType.META, channel=0,
                    data=meta_data, meta_type=meta_enum,
                    meta_type_code=meta_type, text=text_str,
                ))
                running_status = 0
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
        mt = ev.meta_type
        if mt is None:
            code = ev.meta_type_code if ev.meta_type_code is not None else 0
            return f"Meta:0x{code:02X} (type={code} data={len(ev.data)}b raw={ev.data.hex()})"
        if mt == MetaType.TEMPO:
            return f"Meta:Tempo {format_tempo(ev)}"
        if mt == MetaType.TIME_SIGNATURE:
            return f"Meta:TimeSig {format_time_signature(ev)}"
        if mt == MetaType.TRACK_NAME:
            return f"Meta:TrackName \"{ev.text}\""
        if mt == MetaType.END_OF_TRACK:
            return "Meta:EndOfTrack"
        if mt == MetaType.TEXT:
            return f"Meta:Text \"{ev.text}\""
        if mt == MetaType.COPYRIGHT:
            return f"Meta:Copyright \"{ev.text}\""
        if mt == MetaType.INSTRUMENT_NAME:
            return f"Meta:Instrument \"{ev.text}\""
        if mt == MetaType.LYRIC:
            return f"Meta:Lyric \"{ev.text}\""
        if mt == MetaType.MARKER:
            return f"Meta:Marker \"{ev.text}\""
        if mt == MetaType.CUE_POINT:
            return f"Meta:CuePoint \"{ev.text}\""
        if mt == MetaType.SEQUENCE_NUMBER:
            sn = struct.unpack(">H", ev.data[:2])[0] if len(ev.data) >= 2 else 0
            return f"Meta:SequenceNumber seq={sn}"
        if mt == MetaType.CHANNEL_PREFIX:
            ch = ev.data[0] if ev.data else 0
            return f"Meta:ChannelPrefix ch={ch}"
        if mt == MetaType.MIDI_PORT:
            port = ev.data[0] if ev.data else 0
            return f"Meta:MidiPort port={port}"
        if mt == MetaType.KEY_SIGNATURE:
            sf = ev.data[0] if len(ev.data) > 0 else 0
            mi = ev.data[1] if len(ev.data) > 1 else 0
            sharps = sf if sf < 128 else sf - 256
            mode = "minor" if mi else "major"
            return f"Meta:KeySig sharps={sharps} mode={mode}"
        if mt == MetaType.SMPTE_OFFSET:
            return f"Meta:SMPTEOffset ({len(ev.data)}b)"
        if mt == MetaType.SEQUENCER_SPECIFIC:
            return f"Meta:SequencerSpecific ({len(ev.data)}b)"
        return f"Meta:{mt.name} ({len(ev.data)}b)"
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


def event_to_dict(ev: MidiEvent, seconds: Optional[float] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "tick": ev.tick,
        "track": ev.track,
        "delta": ev.delta,
        "event_type": ev.event_type.name,
    }
    if seconds is not None:
        d["seconds"] = round(seconds, 4)
    if ev.event_type in (EventType.NOTE_OFF, EventType.NOTE_ON, EventType.POLY_PRESSURE,
                        EventType.CONTROL_CHANGE, EventType.PITCH_BEND,
                        EventType.PROGRAM_CHANGE, EventType.CHANNEL_PRESSURE):
        d["channel"] = ev.channel
        if ev.event_type in (EventType.NOTE_OFF, EventType.NOTE_ON, EventType.POLY_PRESSURE,
                            EventType.CONTROL_CHANGE, EventType.PITCH_BEND):
            if len(ev.data) >= 1:
                if ev.event_type in (EventType.NOTE_OFF, EventType.NOTE_ON, EventType.POLY_PRESSURE):
                    d["note"] = ev.data[0]
                elif ev.event_type == EventType.CONTROL_CHANGE:
                    d["controller"] = ev.data[0]
            if len(ev.data) >= 2:
                if ev.event_type in (EventType.NOTE_OFF, EventType.NOTE_ON, EventType.POLY_PRESSURE):
                    d["velocity"] = ev.data[1]
                elif ev.event_type == EventType.CONTROL_CHANGE:
                    d["value"] = ev.data[1]
                elif ev.event_type == EventType.PITCH_BEND:
                    d["value"] = (ev.data[1] << 7) | ev.data[0]
        elif ev.event_type == EventType.PROGRAM_CHANGE:
            if len(ev.data) >= 1:
                d["program"] = ev.data[0]
        elif ev.event_type == EventType.CHANNEL_PRESSURE:
            if len(ev.data) >= 1:
                d["pressure"] = ev.data[0]
    elif ev.event_type == EventType.META:
        if ev.meta_type is not None:
            d["meta_type"] = ev.meta_type.name
            d["meta_type_code"] = ev.meta_type.value
        else:
            code = ev.meta_type_code if ev.meta_type_code is not None else 0
            d["meta_type"] = "UNKNOWN"
            d["meta_type_code"] = code
        if ev.meta_type == MetaType.TEMPO and len(ev.data) >= 3:
            us = struct.unpack(">I", b"\x00" + ev.data)[0]
            d["microseconds_per_beat"] = us
            d["bpm"] = round(60_000_000.0 / us, 2)
        elif ev.meta_type == MetaType.TIME_SIGNATURE and len(ev.data) >= 4:
            d["numerator"] = ev.data[0]
            d["denominator"] = 2 ** ev.data[1]
            d["signature"] = f"{ev.data[0]}/{2 ** ev.data[1]}"
        elif ev.meta_type == MetaType.KEY_SIGNATURE and len(ev.data) >= 2:
            sf = ev.data[0]
            mi = ev.data[1]
            d["sharps_flats"] = sf if sf < 128 else sf - 256
            d["mode"] = "minor" if mi else "major"
        if ev.text:
            d["text"] = ev.text
        d["data_hex"] = ev.data.hex()
    elif ev.event_type in (EventType.SYSEX, EventType.ESCAPE):
        d["data_hex"] = ev.data.hex()
        d["data_length"] = len(ev.data)
    return d


def note_event_to_dict(n: NoteEvent) -> Dict[str, Any]:
    return {
        "track": n.track,
        "channel": n.channel,
        "note": n.note,
        "velocity": n.velocity,
        "start_tick": n.start_tick,
        "end_tick": n.end_tick,
        "duration_ticks": n.duration_ticks,
        "start_seconds": round(n.start_seconds, 4),
        "end_seconds": round(n.end_seconds, 4),
        "duration_seconds": round(n.duration_seconds, 4),
    }


def export_events_to_json(midi: MidiFile, path: str) -> int:
    tmap = midi.tempo_map()
    events = midi.all_events_sorted()
    data = {
        "format": midi.format_type,
        "tracks_count": len(midi.tracks),
        "ticks_per_beat": midi.ticks_per_beat,
        "events": [event_to_dict(ev, midi.tick_to_seconds(ev.tick, tmap)) for ev in events],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return len(events)


def export_notes_to_json(midi: MidiFile, path: str, include_grouped: bool = True) -> Dict[str, int]:
    result = midi.extract_notes_extended()
    notes_data = {
        "total_notes": len(result.notes),
        "warnings_count": len(result.warnings),
        "unresolved_note_ons": result.unresolved_note_ons,
        "notes": [note_event_to_dict(n) for n in result.notes],
        "warnings": [{
            "tick": w.tick,
            "track": w.track,
            "channel": w.channel,
            "note": w.note,
            "action": w.action,
            "new_velocity": w.new_velocity,
            "old_start_tick": w.old_start_tick,
            "old_velocity": w.old_velocity,
        } for w in result.warnings],
    }
    if include_grouped:
        grouped = result.by_track_and_channel()
        grouped_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for t, ch_dict in grouped.items():
            grouped_data[str(t)] = {}
            for ch, notes in ch_dict.items():
                grouped_data[str(t)][str(ch)] = [note_event_to_dict(n) for n in notes]
        notes_data["notes_by_track_channel"] = grouped_data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notes_data, f, ensure_ascii=False, indent=2)
    return {
        "notes": len(result.notes),
        "warnings": len(result.warnings),
    }


def print_analysis(midi: MidiFile) -> None:
    analysis = midi.analyze()
    print("\n" + "=" * 70)
    print("SONG ANALYSIS")
    print("=" * 70)
    print(f"  Format:          {analysis['format_type']}")
    print(f"  Tracks:          {analysis['tracks_count']}")
    print(f"  Ticks/Beat:      {analysis['ticks_per_beat']}")
    print(f"  Total Duration:  {analysis['total_seconds']:.3f}s ({analysis['total_minutes']:.2f} min)")
    print(f"  Total Ticks:     {analysis['total_ticks']}")
    print(f"  Total Notes:     {analysis['total_notes']}")
    print(f"  Pitch Range:     {analysis['overall_pitch_range']['min']}-{analysis['overall_pitch_range']['max']} "
          f"({analysis['overall_range_semitones']} semitones)")
    print()
    print(f"  Tempo Changes:   {len(analysis['tempo_changes'])}")
    for t in analysis["tempo_changes"][:10]:
        print(f"    tick={t['tick']:>6d}  {t['seconds']:>8.3f}s  {t['bpm']:.2f} BPM")
    if len(analysis["tempo_changes"]) > 10:
        print(f"    ... ({len(analysis['tempo_changes']) - 10} more)")
    print()
    print(f"  Time Sig Changes: {len(analysis['time_signature_changes'])}")
    for ts in analysis["time_signature_changes"]:
        print(f"    tick={ts['tick']:>6d}  {ts['seconds']:>8.3f}s  {ts['signature']}")
    print()
    print(f"  Per-Track Details:")
    for idx in sorted(analysis["tracks"].keys()):
        tinfo = analysis["tracks"][idx]
        pr = tinfo["pitch_range"]
        print(f"    Track {idx}: \"{tinfo['name']}\"")
        print(f"      Events: {tinfo['total_events']}  Notes: {tinfo['note_count']}")
        if tinfo["note_count"] > 0:
            print(f"      Pitch Range: {pr['min']}-{pr['max']} ({tinfo['note_range_semitones']} semitones)")
        else:
            print(f"      (conductor track / no notes)")
    print()
    if analysis["warnings_count"] > 0:
        print(f"  ⚠  Warnings: {analysis['warnings_count']} total")
        if analysis["overlaps_count"] > 0:
            print(f"     - Premature note closures (overlaps): {analysis['overlaps_count']}")
        if analysis["orphaned_note_offs"] > 0:
            print(f"     - Orphaned note-offs: {analysis['orphaned_note_offs']}")
        if analysis["unresolved_note_ons"] > 0:
            print(f"     - Unresolved note-ons: {analysis['unresolved_note_ons']}")
    else:
        print(f"  ✓ No warnings detected")


def print_grouped_notes(midi: MidiFile) -> None:
    result = midi.extract_notes_extended()
    grouped = result.by_track_and_channel()
    print("\n" + "=" * 70)
    print("NOTES GROUPED BY TRACK & CHANNEL")
    print("=" * 70)
    for t in sorted(grouped.keys()):
        for ch in sorted(grouped[t].keys()):
            notes = grouped[t][ch]
            print(f"\n  Track {t}, Channel {ch} — {len(notes)} notes")
            pitches = sorted(set(n.note for n in notes))
            vels = [n.velocity for n in notes]
            avg_vel = sum(vels) / len(vels) if vels else 0
            print(f"    Unique pitches: {len(pitches)}  |  Avg velocity: {avg_vel:.1f}")
            print(f"    Pitch range: {min(pitches)}-{max(pitches)} ({max(pitches) - min(pitches)} semitones)")
            for n in notes[:10]:
                print(f"      {n.note:>3d}  vel={n.velocity:>3d}  "
                      f"ticks={n.start_tick}-{n.end_tick} ({n.duration_ticks:>4d})  "
                      f"{n.start_seconds:>7.3f}-{n.end_seconds:>7.3f}s ({n.duration_seconds:.3f}s)")
            if len(notes) > 10:
                print(f"      ... ({len(notes) - 10} more)")
    if result.warnings:
        print("\n" + "-" * 70)
        print(f"  WARNINGS ({len(result.warnings)}):")
        for w in result.warnings[:20]:
            if w.action == "note-off with no matching note-on (orphaned)":
                print(f"    tick={w.tick:>6d}  Track {w.track} Ch {w.channel:>2d}: "
                      f"Orphaned NoteOff for note {w.note}")
            elif w.action == "unresolved note-on (never received note-off)":
                print(f"    tick={w.tick:>6d}  Track {w.track} Ch {w.channel:>2d}: "
                      f"Unresolved NoteOn for note {w.note} (vel={w.new_velocity})")
            else:
                print(f"    tick={w.tick:>6d}  Track {w.track} Ch {w.channel:>2d}: "
                      f"Note {w.note} overlap: closed note-on from tick {w.old_start_tick} "
                      f"(vel={w.old_velocity}) before new note-on (vel={w.new_velocity})")
        if len(result.warnings) > 20:
            print(f"    ... ({len(result.warnings) - 20} more warnings)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI File Parser & Event Processing Engine")
    parser.add_argument("file", help="Path to MIDI file (.mid)")
    parser.add_argument("--analysis", action="store_true", help="Show detailed song analysis")
    parser.add_argument("--grouped", action="store_true", help="Show notes grouped by track and channel")
    parser.add_argument("--export-events", help="Export full event stream to JSON file")
    parser.add_argument("--export-notes", help="Export extracted notes to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress default summary output")

    args = parser.parse_args()

    midi = parse_midi(args.file)

    if not args.quiet:
        print_summary(midi)
        if args.analysis:
            print_analysis(midi)
        if args.grouped:
            print_grouped_notes(midi)

    if args.export_events:
        count = export_events_to_json(midi, args.export_events)
        print(f"\n✓ Exported {count} events to {args.export_events}")

    if args.export_notes:
        counts = export_notes_to_json(midi, args.export_notes)
        print(f"\n✓ Exported {counts['notes']} notes and {counts['warnings']} warnings to {args.export_notes}")
