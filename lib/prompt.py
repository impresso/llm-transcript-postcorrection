# -*- coding: utf-8 -*-
import os.path

import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch import nn


class Prompt:

    def __call__(self, prompt, options=None, device='cpu'):
        return self.prediction(prompt, options)

    def prediction(self, prompt, options=None):
        pass


class GPTPrompt(Prompt):
    '''
    Prompting for GPT-2 anf GPT-3 through their API
    '''

    def __init__(self, api_key=None, model='text-davinci-003', device='cpu'):
        openai.api_key = api_key
        self.device = device
        self.options = {
            'engine': model,
            'temperature': 0.7,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'max_tokens': 512
        }
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

    def prediction(self, prompt, options=None):
        if not options:
            options = self.options

        # max_tokens = len(self.tokenizer(prompt, return_tensors="pt").to(self.device))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_model_tokens = 6144
        if 'engine' in options:
            if 'davinci' in options['engine']:
                max_model_tokens = 2049
            elif '2' in options['engine']:
                max_model_tokens = 1024
            else:
                max_model_tokens = 4096
            if '3' in options['engine']:
                max_model_tokens = 2049

        elif 'model' in options['model']:
            if 'davinci' in options['model']:
                max_model_tokens = 2049
            elif '2' in options['model']:
                max_model_tokens = 1024
            else:
                max_model_tokens = 4096
            if '3' in options['model']:
                max_model_tokens = 2049

        if inputs['input_ids'].shape[1] > max_model_tokens:
            inputs['input_ids'] = inputs['input_ids'][:, -
                                                      (max_model_tokens // 2):]
            options['max_tokens'] = max_model_tokens

        # print('max_model_tokens', max_model_tokens, inputs['input_ids'].shape[1])
        if inputs['input_ids'].shape[1] > max_model_tokens // 2:
            inputs['input_ids'] = inputs['input_ids'][:, -
                                                      (max_model_tokens // 2):]
            options['max_tokens'] = max_model_tokens // 2

            # print('---', inputs['input_ids'].shape[1])
            prompt = self.tokenizer.decode(*inputs['input_ids'])
        else:
            options['max_tokens'] = inputs['input_ids'].shape[1]

        # print(inputs['input_ids'].shape[1], max_model_tokens)
        # import pdb;pdb.set_trace()
        if ('3' in options['engine']) or ('4' in options['engine']):
            options.update({'model': options['engine']})
            if 'engine' in options:
                options.pop('engine')

            prompt = self.tokenizer.decode(inputs['input_ids'][0])
            # import pdb;pdb.set_trace()
            result = openai.ChatCompletion.create(
                **options, messages=[{"role": "user", "content": prompt}])['choices'][0]['message']['content']
        else:
            prompt = self.tokenizer.decode(inputs['input_ids'][0])
            result = openai.Completion.create(
                prompt=prompt, **options)['choices'][0]['text']

        return result


class HFPrompt(Prompt):
    '''
    Prompting for HuggingFace models that can be loaded with AutoForCausalLM
    '''

    def __init__(self, api_key=None, model="bigscience/bloom", device='cpu'):

        self.model_name = model
        self.device = device
        from llama import Llama
        if 'llama-2' in model.lower():
            print('tokenizer path:', '/'.join(model.split('/')[:-2]))
            self.model = Llama.build(
                ckpt_dir=model,
                tokenizer_path=os.path.join('/'.join(model.split('/')[:-2]), 'tokenizer.model'),
                max_seq_len=512,
                max_batch_size=1,
            )
        elif 'llama' in model.lower():

            self.tokenizer = LlamaTokenizer.from_pretrained(model)
            self.model = LlamaForCausalLM.from_pretrained(model).to(device)

        else:
            if 'llama' in model.lower():
                self.tokenizer = LlamaTokenizer.from_pretrained(model)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            if torch.cuda.device_count() > 1:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model, device_map='auto')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model).to(device)  # , device_map='auto')#.to(device)
                # self.tokenizer = AutoTokenizer.from_pretrained(model)

                # self.model = nn.DataParallel(self.model)
                # import torch
        self.model.to_bettertransformer()
        # print(torch.cuda.device_count())

    def prediction(self, prompt, options=None, search='topk'):

        if 'llama-2' in self.model_name.lower():
            if len(prompt.split()) >= 512:
                max_gen_len = 512
            else:
                max_gen_len = len(prompt.split())
            results = self.model.text_completion(
                [prompt], max_gen_len=max_gen_len, temperature=0.6, top_p=0.9)
            # import pdb;
            # pdb.set_trace()
            # for prompt, result in zip([prompt], results):
            #     print(prompt)
            #     print(f"> {result['generation']}")
            #     print("\n==================================\n")
            return results[0]['generation']
        else:
            inputs = self.tokenizer(
                prompt, return_tensors="pt").to(
                self.device)

            options['max_tokens'] = inputs['input_ids'].shape[1] * 2 + 1
            try:
                max_model_tokens = self.model.config.max_position_embeddings
            except BaseException:
                max_model_tokens = 2048

            if options['max_tokens'] > max_model_tokens:
                inputs['input_ids'] = inputs['input_ids'][:,
                                                          :max_model_tokens // 2]
                options['max_tokens'] = max_model_tokens

            if search == 'greedy':
                # Greedy Search
                with torch.no_grad():
                    result = self.tokenizer.decode(
                        self.model.generate(
                            inputs["input_ids"],
                            pad_token_id=self.tokenizer.eos_token_id,
                            temperature=options['temperature'],
                            max_length=options['max_tokens']
                        )[0])

            elif search == 'beam':
                # Beam Search
                with torch.no_grad():
                    result = self.tokenizer.decode(
                        self.model.generate(
                            inputs["input_ids"],
                            max_length=options['max_tokens'],
                            pad_token_id=self.tokenizer.eos_token_id,
                            num_beams=2,
                            no_repeat_ngram_size=2,
                            early_stopping=True)[0])
            else:
                # in case no specific request, apply the best one: top-k
                # Sampling Top-k + Top-p

                with torch.no_grad():
                    result = self.tokenizer.decode(
                        self.model.generate(
                            input_ids=inputs["input_ids"],
                            max_length=options['max_tokens'],
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=True, top_k=50, top_p=0.9)[0])

            # OPT adds the prompt in the response, so we are removing it
            last_comment_in_prompt = prompt.split('\n')[-1]
            if last_comment_in_prompt in result:
                result = result[result.index(
                    last_comment_in_prompt) + len(last_comment_in_prompt) + 1:]

        return result

