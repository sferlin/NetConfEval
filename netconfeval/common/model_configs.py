def _build_llama2_prompt(messages):
    start_prompt = "<s>[INST] "
    end_prompt = " [/INST]"

    conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() == 'user':
            conversation.append(content)
        elif role.lower() == 'ai':
            conversation.append(f"{content}")
        elif role.lower() == 'function':
            raise ValueError('Llama 2 does not support function calls.')
        elif role.lower() == 'system':
            conversation.append(f"<<SYS>>\n{content}\n<</SYS>>\n\n")
        else:
            raise ValueError(f"Invalid message role: {role}")

    return start_prompt + ''.join(conversation) + end_prompt


def _build_phind_prompt(messages):
    system_token = "### System Prompt \n"
    user_token = "\n ### User Message \n"
    assistant_token = "\n ### Assistant \n"

    system_conversation = []
    user_conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() == 'user':
            user_conversation.append(content)
        elif role.lower() == 'function' or role.lower() == 'ai':
            raise ValueError('Llama 2 does not support function and ai calls.')
        elif role.lower() == 'system':
            system_conversation.append(content)
        else:
            raise ValueError(f"Invalid message role: {role}")

    return system_token + ''.join(system_conversation) + user_token + ''.join(user_conversation) + assistant_token


def _build_mistral_instruct_prompt(messages):
    start_prompt = "<s>[INST] "
    end_prompt = " [/INST]"

    conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() in ['user', 'system']:  # 'user' and 'system' are treated the same.
            conversation.append(content)
        elif role.lower() in ['ai', 'function']:  # 'ai' and 'function' roles are not supported.
            raise ValueError(f"Mistral model does not support the role: {role}")
        else:
            raise ValueError(f"Unexpected role: {role}")

    # Assemble the prompt with the start and end tokens.
    prompt = start_prompt + ' '.join(conversation) + end_prompt

    return prompt


def _build_mistral_lite_prompt(messages):
    start_prompt = "<|prompter|> "
    end_prompt = " </s><|assistant|>"

    conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() in ['user', 'system']:  # 'user' and 'system' are treated the same.
            conversation.append(content)
        elif role.lower() in ['ai', 'function']:  # 'ai' and 'function' roles are not supported.
            raise ValueError(f"Mistral model does not support the role: {role}")
        else:
            raise ValueError(f"Unexpected role: {role}")

    # Assemble the prompt with the start and end tokens.
    prompt = start_prompt + ' '.join(conversation) + end_prompt

    return prompt


model_configurations = {
    'gpt-3.5-turbo': {
        'model_name': 'gpt-3.5-turbo',
        'type': 'openai',
        'args': {}
    },
    'gpt-3.5-1106': {
        'model_name': 'gpt-3.5-turbo-1106',
        'type': 'openai',
        'args': {
            'response_format': {'type': 'json_object'},
            'seed': 5000,
        }
    },
    'gpt-3.5-0613': {
        'model_name': 'gpt-3.5-turbo-0613',
        'type': 'openai',
        'args': {}
    },
    'gpt-4': {
        'model_name': 'gpt-4',
        'type': 'openai',
        'args': {}
    },
    'gpt-4-turbo': {
        'model_name': 'gpt-4-turbo',
        'type': 'openai',
        'args': {}
    },
    'gpt-4-1106': {
        'model_name': 'gpt-4-1106-preview',
        'type': 'openai',
        'args': {
            'response_format': {'type': 'json_object'},
            'seed': 5000,
        }
    },
    'llama2-7b-chat': {
        'model_name': 'meta-llama/Llama-2-7b-chat-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'llama2-13b-chat': {
        'model_name': 'meta-llama/Llama-2-13b-chat-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'codellama-7b-instruct': {
        'model_name': 'codellama/CodeLlama-7b-Instruct-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'codellama-13b-instruct': {
        'model_name': 'codellama/CodeLlama-13b-Instruct-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'codellama-34b-instruct': {
        'model_name': 'codellama/CodeLlama-34b-Instruct-hf',
        'prompt_builder': _build_llama2_prompt,
        'use_quantization': True,
        'max_length': 4096,
        'type': 'HF'
    },
    'phind-34b-v2': {
        'model_name': 'Phind/Phind-CodeLlama-34B-v2',
        'prompt_builder': _build_phind_prompt,
        'use_quantization': True,
        'max_length': 4096,
        'type': 'HF'
    },
    'mistral-7b-instruct': {
        'model_name': 'mistralai/Mistral-7B-Instruct-v0.1',
        'prompt_builder': _build_mistral_instruct_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'mistral-lite': {
        'model_name': 'amazon/MistralLite',
        'prompt_builder': _build_mistral_lite_prompt,
        'max_length': 32768,
        'type': 'HF',
        'use_quantization': False
    },
}
