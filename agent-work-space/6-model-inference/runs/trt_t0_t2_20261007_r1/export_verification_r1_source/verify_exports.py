"""Independently verify packed bundles against every external ONNX INT4 tensor."""
import argparse
import gc
import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto
import torch

from common import sha, write_json


def raw_bytes(t, folder):
    if t.raw_data:
        return t.raw_data
    external = {v.key: v.value for v in t.external_data}
    with (folder / external['location']).open('rb') as f:
        f.seek(int(external.get('offset', 0)))
        return f.read(int(external['length']))


def signed_nibbles(raw):
    packed = np.frombuffer(raw, dtype=np.uint8)
    q = np.empty(packed.size * 2, dtype=np.int8)
    q[0::2] = packed & 15
    q[1::2] = packed >> 4
    q[q >= 8] -= 16
    return q


def main(out, root):
    assert not out.exists(), out
    results = {}
    for algorithm in ['AWQ', 'GPTQ', 'RTN']:
        folder = root / (algorithm.lower() + '_r1')
        manifest = json.loads((folder / 'packed_manifest.json').read_text())
        parser = json.loads((folder / 'parser_validation.json').read_text())
        graph = onnx.load(folder / 'prefix_int4.onnx', load_external_data=False)
        initializers = {t.name: t for t in graph.graph.initializer}
        dq = {n.input[0]: n for n in graph.graph.node if n.op_type == 'DequantizeLinear'}
        verified = []
        for entry in manifest['layers']:
            path = pathlib.Path(entry['file'])
            assert sha(path) == entry['sha256']
            bundle = torch.load(path, weights_only=True)
            shape = bundle['shape']
            assert bundle['packed'].dtype == torch.uint8
            assert bundle['scale'].dtype == torch.bfloat16
            assert bundle['input_multiplier'].dtype == torch.bfloat16
            assert bundle['group_size'] == 128 and shape[1] % 128 == 0
            assert list(bundle['scale'].shape) == [shape[0], shape[1] // 128]
            assert list(bundle['input_multiplier'].shape) == [shape[1]]
            q = signed_nibbles(bundle['packed'].numpy().tobytes()).reshape(shape)
            param = entry['module'].removeprefix('paligemma_with_expert.paligemma.model.') + '.weight'
            qname = param + '.int4'
            if qname in initializers:
                t = initializers[qname]
                assert t.data_type == TensorProto.INT4 and list(t.dims) == shape[::-1]
                exported = signed_nibbles(raw_bytes(t, folder)).reshape(shape[::-1])
                assert np.array_equal(exported, q.T), entry['module']
                s = initializers[param + '.scale']
                assert s.data_type == TensorProto.BFLOAT16
                assert list(s.dims) == [shape[1] // 128, shape[0]]
                expected = bundle['scale'].T.contiguous().view(torch.int16).numpy().tobytes()
                assert raw_bytes(s, folder) == expected
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in dq[qname].attribute}
                assert attrs['axis'] == 0 and attrs['block_size'] == 128
                assert len(dq[qname].input) == 2  # Omitted zero point is signed zero.
                pname = param + '.input_multiplier'
                if pname in initializers:
                    expected = bundle['input_multiplier'].view(torch.int16).numpy().tobytes()
                    assert raw_bytes(initializers[pname], folder) == expected
                else:
                    assert torch.equal(bundle['input_multiplier'], torch.ones_like(bundle['input_multiplier']))
                verified.append(entry['module'])
            del bundle, q
            gc.collect()
        assert len(manifest['layers']) == 126 and len(verified) == 121
        assert parser['passed'] and not parser['engine_built']
        results[algorithm] = {
            'passed': True, 'packed_layers_verified': 126, 'ONNX_INT4_tensors_verified_bytewise': 121,
            'exported_codes_scales_input_multipliers_match_packed_bundle': True,
            'TensorRT_parser_passed': True, 'full_INT4_engine_built': False,
            'prefix_ONNX_bytes': (folder / 'prefix_int4.onnx').stat().st_size,
            'prefix_external_data_bytes': (folder / 'prefix_int4.data').stat().st_size,
            'prefix_ONNX_sha256': sha(folder / 'prefix_int4.onnx'),
            'prefix_external_data_sha256': sha(folder / 'prefix_int4.data'),
            'packed_manifest_sha256': sha(folder / 'packed_manifest.json'),
        }
        print('BYTEWISE EXPORT VERIFIED', algorithm, flush=True)
    write_json(out, {'passed': True, 'algorithms': results,
                     'performance_memory_closed_loop_acceptance': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    main(args.out, args.root)
