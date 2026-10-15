from pathlib import Path
import json
import shutil
import tempfile
from .agent import execute, intent_constexpr_options, intent_tensor_types, materialize_language
from .tasks import description

def finish_trial(arguments, result, measured) -> dict:
    destination = arguments.output / result['task'] / result['language']
    (destination / 'measurement.json').write_text(json.dumps(measured, indent=2) + '\n')
    result.update(measured)
    print(json.dumps({key: result.get(key) for key in ('task', 'language', 'status', 'candidate_ms', 'reference_ms', 'ratio')}), flush=True)
    return result

def generate_trial(arguments, row, language, *, generation_method='codex') -> dict:
    if generation_method not in {'codex', 'rag'}:
        raise ValueError('generation_method must be codex or rag')
    if generation_method == 'rag' and language != 'intent':
        raise NotImplementedError('public-manual RAG generation currently supports Intent only')
    directory = Path(tempfile.mkdtemp(prefix=f"{row['task']}-{language}-", dir=arguments.state_root / 'candidates')).resolve()
    materials = materialize_language(arguments.project, arguments.triton_ref, directory / 'materials', language)
    task_text = description(arguments.reference, row)
    task_text += '\n\nCallable parameter signature (names, order and defaults):\n' + row['reference_signature']
    task_text += '\n\nFixed profile parameters (reference defaults included; tensor values are not disclosed):\n' + json.dumps(row['invocation'], indent=2)
    task_text += '\nParameters at their defaults may be omitted by the caller. Preserve optional parameter defaults in the returned callable.'
    task_text += '\n\nExpected return structure, shape and dtype:\n' + json.dumps(row['return_contract'], indent=2)
    if language == 'intent':
        task_text += '\n\nIntent dtype/shape syntax for the same disclosed external tensors. When passing a tensor unchanged to a kernel, use these components in I.In[dtype, shape], I.Out[dtype, shape] or I.InOut[dtype, shape] according to its access direction. Choose your own internal interfaces and kernel stages:\n'
        task_text += '\nTensor | dtype expression | shape expression\n--- | --- | ---\n'
        for (name, components) in intent_tensor_types({'arguments': row['invocation'], 'result': row['return_contract']}).items():
            task_text += f"{name} | `{components['dtype']}` | `{components['shape']}`\n"
        options = intent_constexpr_options(row['invocation'])
        if options:
            task_text += '\nOptional compile-time spellings for the already disclosed Python profile options. If an option is declared as a kernel constexpr, bind it in context.compile inside build and omit it from artifact launch arguments. These options do not describe tensor contents or view-shape symbols.\n\nOption | parameter annotation | entry in constexprs\n--- | --- | ---\n'
            for (name, option) in options.items():
                task_text += f"{name} | `{option['annotation']}` | `{option['binding']}`\n"
    task_text += '\n\nTolerance: ' + json.dumps(arguments.suite['tolerances'][row['tolerance']])
    task_text += '\n\nTiming: ' + row['timing']
    task_text += '\n\nProfile note: ' + row['reason']
    (directory / 'TASK.md').write_text(task_text)
    prompt = f'Implement TASK.md correctly and efficiently using {language}. Submit one complete candidate.py.\n'
    prompt += '\nComplete task and fixed invocation (identical to TASK.md):\n\n' + task_text + '\n'
    if generation_method == 'rag':
        from .retrieval import execute as execute_retrieval
        agent = execute_retrieval(directory, arguments.suite, prompt, state_root=arguments.state_root, stop=arguments.stop)
    else:
        agent = execute(directory, arguments.suite, prompt, executable=arguments.codex, state_root=arguments.state_root, language=language, stop=arguments.stop)
    destination = arguments.output / row['task'] / language
    destination.mkdir(parents=True)
    agent['language_materials'] = materials
    (destination / 'agent.json').write_text(json.dumps(agent, indent=2) + '\n')
    program = destination / 'candidate.py'
    result = {'task': row['task'], 'profile': row['input_index'], 'language': language, 'program': ''}
    if agent['action'] != 'submit':
        measured = {'status': agent['status'], 'error': agent['error']}
    elif not (directory / 'candidate.py').exists():
        measured = {'status': 'agent_program_error', 'error': 'No candidate.py was submitted'}
    else:
        shutil.copyfile(directory / 'candidate.py', program)
        return {**result, 'program': str(program.relative_to(arguments.output)), 'status': 'submitted'}
    return finish_trial(arguments, result, measured)
