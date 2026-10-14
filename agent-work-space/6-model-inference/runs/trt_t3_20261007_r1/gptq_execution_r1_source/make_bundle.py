"""Bind the three engine files and the immutable preprocessing dependencies."""
import argparse
import json
import pathlib

from common import PREVIOUS, PREVIOUS_SHA, ROOT, verified, sha, write_json


def main(out, algorithm, build):
    assert not out.exists()
    engines = {}
    for name in ['vision', 'prefix', 'action_step']:
        if name == 'prefix' and algorithm != 'FLOAT':
            report = json.loads((build / 'build.json').read_text())
            path = build / 'prefix.engine'
            expected = report['engine_sha256']
            assert sha(path) == expected
        else:
            path = PREVIOUS / 't1_attempt_r6' / (name + '.engine')
            expected = verified(path)
        engines[name] = {'path': str(path), 'sha256': expected, 'bytes': path.stat().st_size}
    out.parent.mkdir(exist_ok=True)
    write_json(out, {'version': 'trt_t3_20261007_r1', 'algorithm': algorithm, 'engines': engines,
        'previous_release_manifest_sha256': PREVIOUS_SHA, 'TensorRT': '10.13.3.9',
        'runtime_image': 'sha256:cf96556ab8b55eb3f80770d2a684d6961c0128c8100660a8cc4c7e61ecc60369',
        'resident_engine_count': 3, 'total_engine_file_bytes': sum(x['bytes'] for x in engines.values()),
        'hardware_compatibility': 'NONE, development4090, actual4070 must rebuild/test',
        'preprocessing_source': '/app/src', 'source_checkpoint_loaded_at_inference': False,
        'target_device_accepted': False})
    out.chmod(0o444)
    print('ENGINE BUNDLE BOUND', algorithm, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--algorithm', choices=['FLOAT', 'AWQ', 'GPTQ'], required=True)
    p.add_argument('--build', type=pathlib.Path)
    a = p.parse_args()
    main(a.out, a.algorithm, a.build)
