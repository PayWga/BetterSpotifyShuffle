"""Offline JSON-lines sequencing session: python -m shufflerec.sequence --help."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

from .audit import read_catalog
from .inference import InferenceEngine
from .sequencing import Sequencer, SequencerConfig
from .playback import Snapshot
from .playback_session import PlaybackSession


def read_ids(path):
    frame = pd.read_csv(path, dtype={'track_id': 'string'})
    if 'track_id' not in frame:
        raise ValueError("Playlist/context CSV requires 'track_id'")
    values = frame.track_id.dropna().str.strip()
    return values[values.ne('')].tolist()


def emit(event, **data):
    print(json.dumps({'event': event, **data}, allow_nan=False), flush=True)


def handle_request(seq, request, *, session=None):
    if not isinstance(request, dict):
        raise ValueError('Request must be a JSON object')
    command = request.get('command')
    if command in ('submit', 'observe', 'interrupt'):
        if session is None:
            raise ValueError('Playback commands require a PlaybackSession')
        events = []
        if command == 'submit':
            session.submit_prefix(request.get('count'))
            event = 'submitted'
        elif command == 'observe':
            data = request.get('snapshot')
            if not isinstance(data, dict):
                raise ValueError('observe requires a snapshot object')
            events = session.observe(Snapshot(**data), continuity=request.get('continuity', False))
            event = 'observed'
        else:
            session.interrupt()
            event = 'interrupted'
        return event, {**seq.status(), **session.status(), 'events': [asdict(e) for e in events]}
    if command in ('start', 'feedback') and session and session.tracker.active:
        raise ValueError('Cannot mix manual feedback with an active playback observation')
    if command == 'status':
        return 'status', {**seq.status(), **(session.status() if session else {})}
    if command == 'plan':
        seq.plan()
        return 'planned', seq.status()
    if command == 'start':
        seq.start_next()
        return 'started', seq.status()
    if command == 'feedback':
        if 'selection_id' not in request or 'completion_rate' not in request:
            raise ValueError('Feedback requires selection_id and completion_rate')
        outcome = seq.feedback(request['selection_id'], request['completion_rate'],
                               reason_end=request.get('reason_end', 'track_change'))
        return 'feedback', {'outcome': outcome, **seq.status()}
    raise ValueError('command must be status, plan, start, feedback, submit, observe, or interrupt')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Selected private model checkpoint')
    parser.add_argument('--catalog', required=True, help='Master catalog CSV (id column)')
    parser.add_argument('--playlist', required=True, help='Playlist CSV (track_id column)')
    parser.add_argument('--context', help='Ordered accepted context CSV (track_id column), not raw history')
    parser.add_argument('--config', help='JSON object of SequencerConfig overrides')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1024, help='Candidate embedding cache batch size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--once', action='store_true', help='Print initial queue and exit')
    args = parser.parse_args(argv)
    try:
        options = json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else {}
        if not isinstance(options, dict):
            raise ValueError('Configuration must be a JSON object')
        config = SequencerConfig(**options)
        playlist = read_ids(args.playlist)
        context = read_ids(args.context) if args.context else []
        # Load only playlist and context tracks into the embedding cache.
        # The rest of the master catalog is not needed for this session.
        catalog = read_catalog(args.catalog, track_ids=set(playlist) | set(context))
        engine = InferenceEngine.from_checkpoint(args.checkpoint, catalog,
                                                  device=args.device, batch_size=args.batch_size)
        seq = Sequencer(engine, playlist, context_ids=context, config=config, seed=args.seed)
        session = PlaybackSession(seq)
        seq.plan()
    except (ValueError, TypeError) as exc:
        parser.exit(2, f'Invalid sequencing inputs: {exc}\n')
    except (OSError, RuntimeError, KeyError, FloatingPointError):
        parser.exit(2, 'Could not initialize inference. Check checkpoint, catalog, and device compatibility.\n')
    emit('ready', **seq.status())
    if args.once:
        return 0
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            event, data = handle_request(seq, request, session=session)
            emit(event, **data)
        except (ValueError, TypeError) as exc:
            # Input lines can contain private IDs, so parsing errors leave them out.
            message = 'Invalid JSON request' if isinstance(exc, json.JSONDecodeError) else str(exc)
            emit('error', message=message)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
