#!/usr/bin/env python3
"""Merge each phase's native FP8 implementation and its shared headers.

Apply from upstream ccc7cad: --apply REPO
Verify the committed tree: --verify TARGET
"""
from pathlib import Path
import argparse
import sys

BASE_COMMIT = 'ccc7cadf3e3a55f5caedd4f3be32df8616eb6329'
HEADER = 'csrc/ppu/sparse_native_fp8_attention.cuh'
INCLUDE = '#include "ppu/sparse_native_fp8_attention.cuh"\n'


def transform(root: Path) -> None:
    removed = []
    for phase in ('prefill', 'decode'):
        folder = root / 'csrc/ppu' / phase / 'sparse'
        target = folder / 'native_fp8.cu'
        parts = [target.read_text().rstrip()]
        for schedule in ('std', 'ws'):
            source = folder / f'native_fp8_{phase}_{schedule}.cu'
            text = source.read_text()
            assert text.count(INCLUDE) == 1, source
            parts.append(text.replace(INCLUDE, '', 1).strip())
            removed.append(source.relative_to(root).as_posix())
            source.unlink()
        text = '\n\n'.join(parts) + '\n'
        text = text.replace('Warp-specialized attention kernels are implemented in native_fp8_decode_ws.cu.',
                            'Standard CTA and warp-specialized kernels share this translation unit.')
        text = text.replace('// WS launchers are implemented in native_fp8_decode_ws.cu.',
                            '// Launch declarations for the kernels below.')
        target.write_text(text)

    header = root / HEADER
    config = root / 'csrc/sparse_native_fp8_config.h'
    arch = root / 'csrc/ppu/sparse_native_fp8_arch.h'
    text = header.read_text()
    text = text.replace('#include <hggc_runtime.h>\n',
                        '#include <hggc_runtime.h>\n\n' + config.read_text().removeprefix('#pragma once\n').strip() + '\n', 1)
    text = text.replace('#include "ppu/sparse_native_fp8_arch.h"',
                        arch.read_text().removeprefix('#pragma once\n').strip(), 1)
    text = text.replace('#include "sparse_native_fp8_config.h"\n', '', 1)
    text = text.replace('Host C++ includes only launch contracts.',
                        'Host C++ includes launch contracts and policy.')
    header.write_text(text)
    config.unlink()
    arch.unlink()

    api = root / 'csrc/api/sparse_quantized.h'
    text = api.read_text()
    assert text.count('#include "sparse_native_fp8_config.h"\n') == 1
    api.write_text(text.replace('#include "sparse_native_fp8_config.h"\n', '', 1))

    setup = root / 'setup.py'
    text = setup.read_text()
    for path in removed:
        line = f'        "{path}",\n'
        assert text.count(line) == 1, path
        text = text.replace(line, '', 1)
    setup.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--apply', type=Path)
    group.add_argument('--verify')
    args = parser.parse_args()
    if args.apply:
        transform(args.apply)
    else:
        import tempfile
        import subprocess
        sys.path.insert(0, '/root/.codex/skills/mechanical-refactor-verify')
        import mechanical_refactor_verify_utils as utils
        target = subprocess.check_output(['git', 'rev-parse', '--verify', args.verify], text=True).strip()
        # Use the skill's scaffold with the SSH PPU worktree path/branch policy.
        tempfile.tempdir = '/root/lfwang/Projects/t-head/worktrees'
        original_mkdtemp = tempfile.mkdtemp
        def policy_tempdir(*pos, **kw):
            kw['prefix'] = 'FlashMLA-for-sail-verify-sparse-merge-'
            return original_mkdtemp(*pos, **kw)
        tempfile.mkdtemp = policy_tempdir
        original_exec = utils.exec_command
        def policy_exec(command, **kwargs):
            return original_exec(command.replace(' -b verify-mechanical-', ' -b codex/verify-sparse-merge-final-'), **kwargs)
        utils.exec_command = policy_exec
        def committed_transform(root):
            transform(root)
            utils.git_add_and_commit('refactor(sparse): merge native FP8 files by phase', cwd=str(root))
        utils.verify_mechanical_refactor(base_commit=BASE_COMMIT, target_commit=target,
                                         transform=committed_transform)
