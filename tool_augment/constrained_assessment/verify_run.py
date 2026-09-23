#!/usr/bin/env python3
"""Independently check recorded call/dispatch/return provenance, not accuracy."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as pilot


def main(out):
    protocol = pilot.read(out / 'PROTOCOL.json')
    summary = pilot.read(out / 'SUMMARY.json')
    if summary['status'] != 'complete' or summary['cases_completed'] != len(protocol['cases']):
        raise ValueError('Not a complete experiment')
    checks = []
    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'mllm_client_seconds': 0., 'tool_seconds': 0.}
    for item in protocol['cases']:
        case = out / item['case_id']
        tool_request = pilot.read(case / 'MODEL_TOOL_REQUEST.json')
        request_response = pilot.read(case / 'calls/02_tool_request.response.json')
        emitted = json.loads(request_response['choices'][0]['message']['content'])
        assert tool_request == emitted, 'Dispatcher request differs from actual model output'
        pilot.validate_call(tool_request, item)
        execution = pilot.read(case / 'tool_execution/EXECUTION_REQUEST.json')
        assert execution['validated_model_request'] == emitted
        assert execution['cache_used'] is False
        result = pilot.read(case / 'tool_execution/TOOL_RESULT.json')
        assert result['cache_used'] is False
        assert result['full_measurements_sha256'] == pilot.sha(case / 'tool_execution/MEASUREMENTS.json')
        raw_receipt = pilot.read(case / 'tool_execution/raw/RUN_RECEIPT.json')
        assert raw_receipt['input_sha256'] == item['source_sha256'] == pilot.sha(item['source'])
        assert raw_receipt['frames_processed'] == item['frames']
        final_request = pilot.read(case / 'calls/03_tool_judgment.request.json')
        assistant_turns = [m for m in final_request['messages'] if m['role'] == 'assistant']
        assert len(assistant_turns) == 1 and json.loads(assistant_turns[0]['content']) == emitted
        texts = []
        for m in final_request['messages']:
            texts += [m['content']] if isinstance(m['content'], str) else [b['text'] for b in m['content'] if b['type'] == 'text']
        tool_messages = [t for t in texts if 'TOOL_RESULT:\n' in t]
        assert len(tool_messages) == 1
        envelope = tool_messages[0].split('TOOL_RESULT:\n', 1)[1].split('\nUsing this returned evidence', 1)[0]
        assert json.loads(envelope) == result, 'Returned result differs from actual tool evidence'
        initial_ids = [im['evidence_id'] for im in item['images']]
        baseline = pilot.read(case / 'BASELINE.json')
        judgment = pilot.read(case / 'TOOL_JUDGMENT.json')
        pilot.validate_judgment(baseline, item, initial_ids)
        artifacts = result.get('artifacts', [])
        pilot.validate_judgment(judgment, item, initial_ids + [e['evidence_id'] for e in result['evidence']] + [a['evidence_id'] for a in artifacts])
        image_sets = []
        for prefix in ['01_baseline', '02_tool_request', '03_tool_judgment']:
            req = pilot.read(case / f'calls/{prefix}.request.json')
            resp = pilot.read(case / f'calls/{prefix}.response.json')
            assert resp['choices'][0]['finish_reason'] == 'stop'
            if prefix != '02_tool_request':
                stored = baseline if prefix == '01_baseline' else judgment
                assert json.loads(resp['choices'][0]['message']['content']) == stored
            ims = []
            for message in req['messages']:
                if isinstance(message['content'], list):
                    ims += [x['image_url']['url'] for x in message['content'] if x['type'] == 'image_url']
            assert len(ims) == 6 + (len(artifacts) if prefix == '03_tool_judgment' else 0)
            if prefix == '03_tool_judgment':
                for im, artifact in zip(ims[6:], artifacts):
                    expected = (out / artifact['path']).read_bytes()
                    assert hashlib.sha256(expected).hexdigest() == artifact['sha256']
                    assert base64.b64decode(im.split(';base64,', 1)[1]) == expected
            image_sets.append(ims)
            # Check outcome-bearing source names were not included as text.
            request_text = json.dumps(req, ensure_ascii=False)
            assert Path(item['source']).name not in request_text
            for k in ['prompt_tokens', 'completion_tokens']:
                total_usage[k] += resp['usage'][k]
            total_usage['mllm_client_seconds'] += resp['client_elapsed_s']
        assert image_sets[0] == image_sets[1] == image_sets[2][:6]
        total_usage['tool_seconds'] += result['elapsed_s']
        checks.append({'case_id': item['case_id'], 'model_request_executed_exactly': True,
                       'fresh_tool_execution': True, 'exact_result_returned_to_model': True,
                       'same_six_original_images_in_all_calls': True, 'tool_diagram_return_verified': True,
                       'references_exist': True,
                       'input_source_sha256_verified': True})
    report = {'status': 'passed', 'cases': checks, 'formal_model_calls': 3 * len(checks), 'tool_invocations': len(checks),
              'aggregate_usage': total_usage,
              'what_this_does_not_test': ['Semantic entailment of cited evidence', 'Clinical correctness',
                                         'Accuracy against independent labels', 'Expert approval of formalized SOP']}
    pilot.save(out / 'VERIFICATION.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output.resolve())
