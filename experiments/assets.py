#!/usr/bin/env python3
"""Where the pretrained model assets live.

eWaSR's clone and checkpoint and MobileSAM's clone and checkpoint are
several hundred megabytes of third-party weights, so they are not in
this repository. They were originally unpacked into a session scratch
directory, whose path was hard-coded in three places and made the tree
unrunnable anywhere else; this resolves it instead.

Order of preference:
  1. $HORIZONATOR_ASSETS
  2. ~/.horizonator/assets          (what setup_local.sh creates)
  3. the legacy session scratchpad, if it still exists

Expected layout:
    <assets>/eWaSR/                 the patched clone
    <assets>/ewasr_resnet18.pth     MaSTr1325 weights
    <assets>/MobileSAM/             the clone, with weights/mobile_sam.pt
"""

import os

_LEGACY = ('/tmp/claude-0/-home-user/'
           '792503f9-74c5-5111-83ca-eeeda63e838d/scratchpad')


def assets_dir():
    env = os.environ.get('HORIZONATOR_ASSETS')
    if env:
        return os.path.expanduser(env)
    home = os.path.expanduser('~/.horizonator/assets')
    if os.path.isdir(home):
        return home
    if os.path.isdir(_LEGACY):
        return _LEGACY
    return home            # report the intended path in the error message


def need(*parts):
    """Path under the asset root, with a useful error when it is absent."""
    p = os.path.join(assets_dir(), *parts)
    if not os.path.exists(p):
        raise SystemExit(
            'missing model asset: %s\n'
            'Run experiments/setup_local.sh, or point $HORIZONATOR_ASSETS '
            'at a directory that has it.' % p)
    return p
